import base64
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
ROOT = Path(__file__).resolve().parents[2]


def _file_base64(name: str) -> str:
    return base64.b64encode((ROOT / name).read_bytes()).decode("ascii")


def _file_base64_by_size(size: int) -> str:
    path = next(item for item in ROOT.glob("*.docx") if item.stat().st_size == size)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _image_base64_by_size(size: int) -> str:
    path = next(item for item in ROOT.iterdir() if item.is_file() and item.stat().st_size == size)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_precheck_fail_for_missing_fields():
    payload = {
        "applicantName": "",
        "idNumber": "",
        "contactPhone": "",
        "waterPurpose": "其他",
        "materials": ["身份证"]
    }
    resp = client.post("/api/ai/precheck", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAIL"
    assert len(data["issues"]) >= 5


def test_mcp_tools_are_exposed():
    resp = client.get("/api/mcp/tools")
    assert resp.status_code == 200
    tools = resp.json()["tools"]
    names = [tool["name"] for tool in tools]
    assert "knowledge_search" in names
    assert "check_completeness" in names


def test_knowledge_upload_should_add_chunk_points_and_similarity_edges():
    content = "\n\n".join(
        [
            f"第{index}条：取水许可申请材料应包含申请书、身份证、营业执照。"
            f"取水地点、申请期限、取水用途和第三方影响说明均需要审核。"
            for index in range(1, 9)
        ]
    )
    resp = client.post(
        "/api/ai/knowledge/upload",
        files={"file": ("课堂规则-可视化.md", BytesIO(content.encode("utf-8")), "text/markdown")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "课堂规则-可视化.md"
    assert data["chunksAdded"] >= 3
    assert data["totalVectors"] >= data["chunksAdded"]
    assert "取水许可申请材料" in data["preview"]

    vectors_resp = client.get("/api/ai/knowledge/vectors")
    assert vectors_resp.status_code == 200
    vectors = vectors_resp.json()
    assert vectors["total"] >= data["chunksAdded"]
    uploaded_points = [point for point in vectors["points"] if point["source"] == "课堂规则-可视化.md"]
    assert len(uploaded_points) >= data["chunksAdded"]
    assert all(isinstance(point["chunk"], int) for point in uploaded_points)
    assert all("docId" in point for point in uploaded_points)
    assert len({(round(point["x"], 6), round(point["y"], 6)) for point in uploaded_points}) > 1
    assert "edges" in vectors
    assert any(edge["source"] == "课堂规则-可视化.md" for edge in vectors["edges"])


def test_knowledge_documents_should_list_and_delete_uploaded_document():
    filename = "课堂规则.md"
    content = "\n\n".join(
        [
            f"第{index}条：取水许可申请材料应包含申请书、身份证、营业执照。"
            f"申请期限、取水地点、取水用途和第三方影响说明均需要审核。"
            for index in range(1, 9)
        ]
    )
    upload_resp = client.post(
        "/api/ai/knowledge/upload",
        files={"file": (filename, BytesIO(content.encode("utf-8")), "text/markdown")},
    )
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert upload_data["filename"] == filename
    assert upload_data["docId"]
    assert upload_data["chunksAdded"] >= 3

    documents_resp = client.get("/api/ai/knowledge/documents")
    assert documents_resp.status_code == 200
    documents = documents_resp.json()["documents"]
    target = next((item for item in documents if item["docId"] == upload_data["docId"]), None)
    assert target is not None
    assert target["filename"] == filename
    assert target["source"] == filename
    assert target["chunks"] == upload_data["chunksAdded"]
    assert target["deletable"] is True
    assert "取水许可申请材料" in target["preview"]

    delete_resp = client.delete(f"/api/ai/knowledge/documents/{upload_data['docId']}")
    assert delete_resp.status_code == 200
    delete_data = delete_resp.json()
    assert delete_data["deleted"] is True
    assert delete_data["deletedChunks"] == upload_data["chunksAdded"]

    vectors_resp = client.get("/api/ai/knowledge/vectors")
    assert vectors_resp.status_code == 200
    assert all(point["docId"] != upload_data["docId"] for point in vectors_resp.json()["points"])

    documents_after = client.get("/api/ai/knowledge/documents").json()["documents"]
    assert all(item["docId"] != upload_data["docId"] for item in documents_after)


def test_knowledge_docx_upload_should_return_complete_parsed_text():
    path = ROOT / "取水许可办理需资料及流程.docx"
    resp = client.post(
        "/api/ai/knowledge/upload",
        files={
            "file": (
                path.name,
                BytesIO(path.read_bytes()),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == path.name
    assert data["parser"] == "docx-xml"
    assert data["contentLength"] > len(data["preview"])
    assert "根据《中华人民共和国水法》" in data["contentText"]
    assert "报告书" in data["contentText"]
    assert "报告表" in data["contentText"]

    delete_resp = client.delete(f"/api/ai/knowledge/documents/{data['docId']}")
    assert delete_resp.status_code == 200


def test_extract_should_return_draft_and_attachment_summary(monkeypatch):
    monkeypatch.setattr("app.agent.ENABLE_REMOTE_AI_REVIEW", False)

    payload = {
        "attachments": [
            {
                "docType": "id_card",
                "filename": "身份证.jpg",
                "mimeType": "image/jpeg",
                "size": 100,
                "contentText": "",
                "base64Content": _file_base64("身份证.jpg"),
            },
            {
                "docType": "business_license",
                "filename": "营业执照.jpg",
                "mimeType": "image/jpeg",
                "size": 100,
                "contentText": "",
                "base64Content": _file_base64("营业执照.jpg"),
            },
        ]
    }

    resp = client.post("/api/ai/extract", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["applicantName"] == "金阳"
    assert data["idNumber"] == "110102197810272321"
    assert data["licenseLegalRepresentative"] == "姜丹丹"
    assert any(item["filename"] == "身份证.jpg" for item in data["attachmentSummaries"])
    assert "身份证" in data["materials"]


def test_extract_should_ocr_new_business_license_png(monkeypatch):
    monkeypatch.setattr("app.agent.ENABLE_REMOTE_AI_REVIEW", False)

    payload = {
        "attachments": [
            {
                "docType": "business_license",
                "filename": "营业执照未过期.png",
                "mimeType": "image/png",
                "size": 1957719,
                "contentText": "",
                "base64Content": _image_base64_by_size(1957719),
            }
        ]
    }

    resp = client.post("/api/ai/extract", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    summary = data["attachmentSummaries"][0]
    assert data["licenseLegalRepresentative"] == "姜丹丹"
    assert "营业执照" in summary["extractedText"]
    assert "91441303MA531L6K37" in summary["extractedText"]
    assert summary["extractedFields"]["companyName"] == "惠州市安建诚表面处理材料有限公司"
    assert summary["warnings"] == []


def test_extract_should_parse_application_docx_table_template(monkeypatch):
    monkeypatch.setattr("app.agent.ENABLE_REMOTE_AI_REVIEW", False)

    payload = {
        "attachments": [
            {
                "docType": "application_form",
                "filename": "申请书.docx",
                "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "size": 29617,
                "contentText": "",
                "base64Content": _file_base64_by_size(29617),
            }
        ]
    }

    resp = client.post("/api/ai/extract", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["applicantName"] == "张三"
    assert data["idNumber"] == "330114200410128315"
    assert data["contactPhone"] == "12345678901"
    assert data["projectName"] == "杭州市恒信建材有限公司新建混凝土砌块生产线项目"
    assert data["formLegalRepresentative"] == "张三"
    assert data["waterPurpose"] == "生活用水、工业用水、一般工业用水"
    assert data["waterLocation"].startswith("浙江省")
    assert data["applicationPeriodYears"] == 5
    assert data["requestedWaterAmount"] == 5.0


def test_precheck_uses_ai_review_when_enabled(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")

    def fake_ai_review(_request):
        return ["AI判定：营业执照疑似过期，请核验有效期"]

    monkeypatch.setattr("app.agent.call_modelscope_review", fake_ai_review)

    payload = {
        "applicantName": "张三",
        "idNumber": "330102199901011234",
        "contactPhone": "13800138000",
        "waterPurpose": "农业灌溉",
        "materials": ["申请书", "身份证", "营业执照"],
    }
    resp = client.post("/api/ai/precheck", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAIL"
    assert any("疑似过期" in issue for issue in data["issues"])


def test_precheck_other_purpose_with_detail_should_not_fail_on_purpose():
    payload = {
        "applicantName": "李四",
        "idNumber": "330102199002021111",
        "contactPhone": "13900139000",
        "waterPurpose": "其他:生态补水",
        "waterLocation": "杭州市某取水点",
        "applicationPeriodYears": 2,
        "thirdPartyImpactDescription": "影响较小",
        "mitigationMeasures": "已制定限额取水与监测方案",
        "legalBasisVersion": "水法2023版",
        "ownershipProofType": "不动产权证",
        "materials": ["申请书", "身份证", "营业执照"],
    }
    resp = client.post("/api/ai/precheck", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "PASS"
    assert all("取水用途选择了其他但未说明具体用途" != issue for issue in data["issues"])


def test_precheck_falls_back_when_ai_call_errors(monkeypatch):
    monkeypatch.setenv("AI_ENABLED", "true")

    def broken_ai_review(_request):
        raise RuntimeError("mock network timeout")

    monkeypatch.setattr("app.agent.call_modelscope_review", broken_ai_review)

    payload = {
        "applicantName": "王五",
        "idNumber": "330102198812121234",
        "contactPhone": "13700137000",
        "waterPurpose": "工业用水",
        "waterLocation": "杭州市工业园区取水点",
        "applicationPeriodYears": 3,
        "thirdPartyImpactDescription": "对周边影响可控",
        "mitigationMeasures": "已设置生态流量保障措施",
        "legalBasisVersion": "水法2023版",
        "ownershipProofType": "租赁合同",
        "materials": ["申请书", "身份证", "营业执照"],
    }
    resp = client.post("/api/ai/precheck", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "PASS"


def test_precheck_should_detect_content_norm_issues(monkeypatch):
    monkeypatch.setattr("app.agent.ENABLE_REMOTE_AI_REVIEW", False)

    payload = {
        "applicantName": "杭州某企业",
        "idNumber": "330102198812121234",
        "contactPhone": "13700137000",
        "waterPurpose": "工业用水",
        "materials": ["申请书", "身份证", "营业执照"],
        "projectName": "滨江取水工程A",
        "attachedProjectName": "滨江取水工程B",
        "waterLocation": "",
        "applicationPeriodYears": None,
        "requestedWaterAmount": 1200,
        "reportEstimatedWaterAmount": 300,
    }

    resp = client.post("/api/ai/precheck", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAIL"
    assert any("项目名称" in issue and "不一致" in issue for issue in data["issues"])
    assert any("取水地点" in issue and "缺失" in issue for issue in data["issues"])
    assert any("申请期限" in issue and "缺失" in issue for issue in data["issues"])
    assert any("取水量" in issue and "超出" in issue for issue in data["issues"])


def test_precheck_should_detect_substantive_and_evidence_issues(monkeypatch):
    monkeypatch.setattr("app.agent.ENABLE_REMOTE_AI_REVIEW", False)

    payload = {
        "applicantName": "宁波某公司",
        "idNumber": "330102198901015678",
        "contactPhone": "13600136000",
        "waterPurpose": "农业灌溉",
        "materials": ["申请书", "身份证", "营业执照"],
        "waterLocation": "某饮用水水源保护区二级区",
        "applicationPeriodYears": 8,
        "projectApprovalPeriodYears": 3,
        "thirdPartyImpactDescription": "",
        "mitigationMeasures": "",
        "reportIssuedAt": "2018-01-01",
        "legalBasisVersion": "水法1998版",
        "ownershipProofType": "",
    }

    resp = client.post("/api/ai/precheck", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAIL"
    assert any("保护区" in issue and ("限制" in issue or "禁止" in issue) for issue in data["issues"])
    assert any("第三方" in issue and "补救" in issue for issue in data["issues"])
    assert any("许可期限" in issue and "超过" in issue for issue in data["issues"])
    assert any("报告" in issue and "过期" in issue for issue in data["issues"])
    assert any("依据" in issue and ("失效" in issue or "更新" in issue) for issue in data["issues"])
    assert any("权属" in issue and "缺失" in issue for issue in data["issues"])


def test_precheck_should_auto_extract_id_and_license_fields_for_compare(monkeypatch):
    monkeypatch.setattr("app.agent.ENABLE_REMOTE_AI_REVIEW", False)

    payload = {
        "applicantName": "张三",
        "idNumber": "330102199901011234",
        "contactPhone": "13800138000",
        "waterPurpose": "农业灌溉",
        "waterLocation": "杭州市某取水点",
        "applicationPeriodYears": 2,
        "thirdPartyImpactDescription": "影响较小",
        "mitigationMeasures": "采取限额取水措施",
        "legalBasisVersion": "水法2023版",
        "ownershipProofType": "不动产权证",
        "materials": ["申请书", "身份证", "营业执照"],
        "attachments": [
            {
                "docType": "id_card",
                "filename": "身份证.jpg",
                "contentText": "姓名 李四 公民身份号码 330102199901011234",
            },
            {
                "docType": "business_license",
                "filename": "营业执照.jpg",
                "contentText": "名称 杭州取水科技有限公司 法定代表人 王五 统一社会信用代码 91330100MA2XXXXX",
            },
        ],
    }

    resp = client.post("/api/ai/precheck", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAIL"
    assert any("身份证姓名" in issue and "不一致" in issue for issue in data["issues"])
    assert any("法定代表人" in issue and "不一致" in issue for issue in data["issues"])


def test_precheck_should_auto_extract_report_project_name(monkeypatch):
    monkeypatch.setattr("app.agent.ENABLE_REMOTE_AI_REVIEW", False)

    payload = {
        "applicantName": "杭州某企业",
        "idNumber": "330102198812121234",
        "contactPhone": "13700137000",
        "waterPurpose": "工业用水",
        "materials": ["申请书", "身份证", "营业执照"],
        "projectName": "滨江取水工程A",
        "waterLocation": "杭州市工业园区取水点",
        "applicationPeriodYears": 2,
        "thirdPartyImpactDescription": "影响可控",
        "mitigationMeasures": "采取生态补偿措施",
        "legalBasisVersion": "水法2023版",
        "ownershipProofType": "租赁合同",
        "attachments": [
            {
                "docType": "report",
                "filename": "水资源论证报告.pdf",
                "contentText": "项目名称：滨江取水工程B 取水规模：300万m3/年",
            }
        ],
    }

    resp = client.post("/api/ai/precheck", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAIL"
    assert any("项目名称" in issue and "不一致" in issue for issue in data["issues"])


def test_precheck_should_detect_driver_license_renamed_as_id_card(monkeypatch):
    monkeypatch.setattr("app.agent.ENABLE_REMOTE_AI_REVIEW", False)

    payload = {
        "applicantName": "张三",
        "idNumber": "330102199901011234",
        "contactPhone": "13800138000",
        "waterPurpose": "农业灌溉",
        "waterLocation": "杭州市某取水点",
        "applicationPeriodYears": 2,
        "thirdPartyImpactDescription": "影响较小",
        "mitigationMeasures": "采取限额取水措施",
        "legalBasisVersion": "水法2023版",
        "ownershipProofType": "不动产权证",
        "materials": ["申请书", "身份证", "营业执照"],
        "attachments": [
            {
                "docType": "id_card",
                "filename": "身份证.png",
                "mimeType": "image/png",
                "size": 100,
                "contentText": "",
                "base64Content": _file_base64("驾驶证.png"),
            }
        ],
    }

    resp = client.post("/api/ai/precheck", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAIL"
    assert any("实际识别为驾驶证" in issue for issue in data["issues"])


def test_precheck_should_detect_expired_business_license_from_text_content(monkeypatch):
    monkeypatch.setattr("app.agent.ENABLE_REMOTE_AI_REVIEW", False)

    payload = {
        "applicantName": "张三",
        "idNumber": "330102199901011234",
        "contactPhone": "13800138000",
        "waterPurpose": "农业灌溉",
        "waterLocation": "杭州市某取水点",
        "applicationPeriodYears": 2,
        "thirdPartyImpactDescription": "影响较小",
        "mitigationMeasures": "采取限额取水措施",
        "legalBasisVersion": "水法2023版",
        "ownershipProofType": "不动产权证",
        "materials": ["申请书", "身份证", "营业执照"],
        "attachments": [
            {
                "docType": "business_license",
                "filename": "营业执照.jpg",
                "contentText": "营业执照 名称 杭州取水科技有限公司 法定代表人 王五 营业期限 2018-05-01 至 2024-05-01",
            }
        ],
    }

    resp = client.post("/api/ai/precheck", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAIL"
    assert any("营业执照已过期" in issue for issue in data["issues"])
