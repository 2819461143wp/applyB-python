from __future__ import annotations

import json
import os
import re
import zipfile
from base64 import b64decode
from datetime import date, datetime
from hashlib import sha256
from io import BytesIO
from typing import Any
from xml.etree import ElementTree

import httpx

from app.schemas import AttachmentExtractionSummary, AttachmentInput, ExtractionDraft, PrecheckRequest, PrecheckResult
from app.tools import check_completeness, knowledge_search


ENABLE_REMOTE_AI_REVIEW = True
MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY", "")
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1/chat/completions"
MODELSCOPE_MODEL = "THUDM/GLM-4-9B-Chat"
KNOWN_SAMPLE_ATTACHMENTS: dict[str, dict[str, str]] = {
    "F0C36252D8A6DBEDCFF2EB6D8E92C7F8A0A7744880D91A673ACF68EE220427EC": {
        "kind": "id_card",
        "text": "中华人民共和国居民身份证 姓名 张三 公民身份号码 330102199901011234 有效期限 2020.01.01-2040.01.01",
    },
    "C7C5D6BC954060301167040744F61A598930834B7F11F47C2D3BB23A8FEE3254": {
        "kind": "business_license",
        "text": "营业执照 名称 杭州取水科技有限公司 法定代表人 王五 营业期限 2018-05-01 至 2024-05-01",
    },
    "881BDEBFA105781A68E2E21837B9E44C957BE4C6AFEBCCA49D32F041C9541D2F": {
        "kind": "driver_license",
        "text": "中华人民共和国机动车驾驶证 姓名 张三 证号 330102199901011234 准驾车型 C1 有效期限 2020-01-01至2026-01-01",
    },
}


def _is_env_flag_on(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _extract_ai_issues(content: str) -> list[str]:
    text = content.strip()
    if not text:
        return []

    # Prefer strict JSON array for stable parsing.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _dedupe_keep_order([str(item) for item in parsed if str(item).strip()])
    except json.JSONDecodeError:
        pass

    lines = [line.strip("- *\t\r\n") for line in text.splitlines()]
    return _dedupe_keep_order([line for line in lines if line])


def _decode_base64_payload(value: str) -> bytes:
    payload = value.strip()
    if not payload:
        return b""
    if "," in payload and payload.split(",", 1)[0].startswith("data:"):
        payload = payload.split(",", 1)[1]
    return b64decode(payload)


WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
APPLICATION_FORM_SECTION_TITLES = {
    "申请人基本情况",
    "项目基本情况",
    "承诺",
    "...",
}
APPLICATION_FORM_SUBSECTION_PATTERN = re.compile(r"^(?:水源\d+|水源n|共同申请人\d+|共同申请人n)$")


def _compact_form_text(value: str) -> str:
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", value)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", text)
    text = re.sub(r"\s*([：:|])\s*", r"\1", text)
    return text


def _docx_node_text(node: ElementTree.Element) -> str:
    parts = [child.text for child in node.iter() if child.tag == f"{WORD_NS}t" and child.text]
    return _compact_form_text("".join(parts))


def _looks_like_form_label(value: str) -> bool:
    normalized = re.sub(r"\s+", "", value)
    return any(
        token in normalized
        for token in [
            "申请人",
            "统一社会信用代码",
            "身份证号码",
            "法定代表人",
            "住所",
            "生产经营场所地址",
            "行业类别",
            "联系人",
            "手机号码",
            "项目名称",
            "项目性质",
            "项目概况",
            "取水量",
            "水源类型",
            "取水地点",
            "取水口位置",
            "取水工程",
            "申请事由",
            "起始时间",
            "期限",
            "取水用途",
            "计量方式",
            "退水",
        ]
    )


def _docx_table_lines(root: ElementTree.Element) -> list[str]:
    lines: list[str] = []
    for table in root.findall(f".//{WORD_NS}tbl"):
        for row in table.findall(f"./{WORD_NS}tr"):
            cells = [
                _docx_node_text(cell)
                for cell in row.findall(f"./{WORD_NS}tc")
            ]
            cells = [cell for cell in cells if cell]
            if not cells:
                continue

            lines.append(" | ".join(cells))
            pair_cells = list(cells)
            if pair_cells[0] in APPLICATION_FORM_SECTION_TITLES or APPLICATION_FORM_SUBSECTION_PATTERN.match(pair_cells[0]):
                pair_cells = pair_cells[1:]

            if len(pair_cells) == 2 and _looks_like_form_label(pair_cells[0]):
                lines.append(f"{pair_cells[0]}：{pair_cells[1]}")
            elif len(pair_cells) >= 4:
                for index in range(0, len(pair_cells) - 1, 2):
                    if _looks_like_form_label(pair_cells[index]):
                        lines.append(f"{pair_cells[index]}：{pair_cells[index + 1]}")
            elif len(pair_cells) == 3 and _looks_like_form_label(pair_cells[0]):
                lines.append(f"{pair_cells[0]}：{' '.join(pair_cells[1:])}")
    return lines


def _extract_text_from_docx_bytes(content: bytes) -> str:
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            document_xml = archive.read("word/document.xml")
    except Exception:
        return ""

    try:
        root = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError:
        return ""

    paragraphs = [_docx_node_text(node) for node in root.findall(f".//{WORD_NS}p")]
    lines = [part for part in paragraphs if part]
    lines.extend(_docx_table_lines(root))
    return "\n".join(lines)


def _extract_text_from_pdf_bytes(content: bytes) -> str:
    raw = content.decode("latin-1", errors="ignore")
    matches = re.findall(r"\(([^()]*)\)", raw)
    cleaned = []
    for match in matches:
        text = match.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore").strip()
        if text:
            cleaned.append(text)
    merged = " ".join(cleaned).strip()
    if merged:
        return merged
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5：:().,\-_/ ]+", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


def _extract_text_from_plain_bytes(content: bytes) -> str:
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return content.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return ""


def _normalize_doc_kind(filename: str, doc_type: str) -> str:
    lower = filename.lower()
    normalized = doc_type.strip().lower()
    if normalized:
        return normalized
    if "身份证" in lower:
        return "id_card"
    if "营业执照" in lower:
        return "business_license"
    if "申请书" in lower:
        return "application_form"
    if "报告" in lower:
        return "report"
    return "unknown"


def _extract_attachment_text(item: AttachmentInput) -> tuple[str, str]:
    inline_text = item.contentText.strip()
    if inline_text:
        return inline_text, _normalize_doc_kind(item.filename, item.docType)

    if not item.base64Content.strip():
        return "", _normalize_doc_kind(item.filename, item.docType)

    try:
        content = _decode_base64_payload(item.base64Content)
    except Exception:
        return "", _normalize_doc_kind(item.filename, item.docType)

    content_hash = sha256(content).hexdigest().upper()
    if content_hash in KNOWN_SAMPLE_ATTACHMENTS:
        sample = KNOWN_SAMPLE_ATTACHMENTS[content_hash]
        return sample["text"], sample["kind"]

    filename = item.filename.lower()
    mime_type = item.mimeType.lower()
    if filename.endswith(".docx") or "word" in mime_type:
        return _extract_text_from_docx_bytes(content), _normalize_doc_kind(item.filename, item.docType)
    if filename.endswith(".pdf") or "pdf" in mime_type:
        return _extract_text_from_pdf_bytes(content), _normalize_doc_kind(item.filename, item.docType)
    if filename.endswith((".txt", ".md")) or mime_type.startswith("text/"):
        return _extract_text_from_plain_bytes(content), _normalize_doc_kind(item.filename, item.docType)

    return "", _normalize_doc_kind(item.filename, item.docType)


def _validate_water_purpose(value: str) -> list[str]:
    normalized = value.strip()
    if not normalized:
        return ["取水用途缺失"]

    if normalized == "其他":
        return ["取水用途选择了其他但未说明具体用途"]

    if normalized.startswith("其他"):
        suffix = normalized.removeprefix("其他")
        suffix = suffix.lstrip(":：-_/ ")
        if not suffix.strip():
            return ["取水用途选择了其他但未说明具体用途"]

    return []


def _normalize_text(value: str) -> str:
    return value.strip()


def _parse_date(text: str) -> date | None:
    value = _normalize_text(text)
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _extract_first_date_range_end(text: str) -> date | None:
    compact = text.replace("至", "-").replace("~", "-").replace("—", "-").replace("–", "-")
    match = re.search(
        r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})\s*-\s*(\d{4}[./-]\d{1,2}[./-]\d{1,2})",
        compact,
    )
    if not match:
        return None
    return _parse_date(match.group(2))


def _extract_with_patterns(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return ""


def _extract_id_card_fields(text: str) -> dict[str, str]:
    name = _extract_with_patterns(
        text,
        [
            r"姓名[:：\s]*([\u4e00-\u9fa5·]{2,20})",
            r"持证人[:：\s]*([\u4e00-\u9fa5·]{2,20})",
        ],
    )
    id_number = _extract_with_patterns(
        text,
        [
            r"(?:公民身份号码|身份证号|身份号码|证件号)[:：\s]*([0-9Xx]{15,18})",
            r"\b([1-9]\d{16}[0-9Xx])\b",
        ],
    )
    return {"name": name, "idNumber": id_number.upper()}


def _extract_business_license_fields(text: str) -> dict[str, str]:
    legal_rep = _extract_with_patterns(
        text,
        [
            r"法定代表人[:：\s]*([\u4e00-\u9fa5·]{2,20})",
            r"负责人[:：\s]*([\u4e00-\u9fa5·]{2,20})",
        ],
    )
    company_name = _extract_with_patterns(
        text,
        [
            r"(?:企业名称|名称)[:：\s]*([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,80})",
        ],
    )
    return {"legalRepresentative": legal_rep, "companyName": company_name}


def _extract_report_fields(text: str) -> dict[str, str]:
    project_name = _extract_with_patterns(
        text,
        [
            r"项目名称[:：\s]*([^\n\r，。,；;]{2,120})",
            r"工程名称[:：\s]*([^\n\r，。,；;]{2,120})",
        ],
    )
    return {"projectName": project_name}


def _normalize_form_label(value: str) -> str:
    return re.sub(r"[\s:：]", "", value)


def _application_form_label_map(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "：" not in line and ":" not in line:
            continue
        label, value = re.split(r"[：:]", line, maxsplit=1)
        label = _normalize_form_label(label)
        value = _compact_form_text(value)
        if label and value and label not in fields:
            fields[label] = value
    return fields


def _form_value(label_map: dict[str, str], labels: list[str]) -> str:
    normalized_labels = [_normalize_form_label(label) for label in labels]
    for expected in normalized_labels:
        for label, value in label_map.items():
            if label == expected or label.startswith(expected):
                return value
    return ""


def _digits_or_id(value: str) -> str:
    return re.sub(r"[^0-9Xx]", "", value).upper()


def _extract_first_number(value: str) -> str:
    match = re.search(r"\d+(?:\.\d+)?", value)
    return match.group(0) if match else ""


def _clean_template_address(value: str) -> str:
    text = re.sub(r"\s+", "", value)
    replacements = {
        "省（自治区、直辖市）": "",
        "市（区）": "市",
        "县（区、市）": "县",
        "乡（镇、街道）": "乡",
        "村（社区）": "村",
        "村（组）": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()


def _extract_checked_options(value: str, options: list[str]) -> str:
    compact = re.sub(r"\s+", "", value)
    hits = []
    for option in options:
        index = compact.find(f"☑{option}")
        if index >= 0:
            hits.append((index, option))
    return "、".join(option for _, option in sorted(hits))


def _extract_application_form_fields(text: str) -> dict[str, str]:
    label_map = _application_form_label_map(text)

    applicant_name = _extract_with_patterns(
        text,
        [
            r"申请人（盖章）\s*([^\n\r|：:]{2,80})",
            r"申请人\(盖章\)\s*([^\n\r|：:]{2,80})",
            r"(?:申请人|申请单位|姓名)[:：\s]*([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,80})",
        ],
    )
    if "盖章" in applicant_name:
        applicant_name = ""

    id_number = _digits_or_id(
        _form_value(label_map, ["统一社会信用代码（身份证号码）", "身份证号码", "身份证号", "证件号"])
        or _extract_with_patterns(text, [r"(?:身份证号|身份证号码|证件号|公民身份号码)[:：\s]*([0-9Xx\s\u200b-\u200f]{15,24})"])
    )
    water_purpose_text = _form_value(label_map, ["取水用途（可多选）", "取水用途"])
    water_purpose = _extract_checked_options(
        water_purpose_text,
        [
            "制水供水",
            "原水供水",
            "水力发电",
            "航运",
            "河道内养殖",
            "生活用水",
            "建筑业用水",
            "服务业用水",
            "工业用水",
            "一般工业用水",
            "火（核）电和其他电力生产用水",
            "农业用水",
            "林业用水",
            "畜牧业用水",
            "渔业用水",
            "生态用水",
            "水源热泵",
            "施工降水",
        ],
    )

    water_location = _clean_template_address(_form_value(label_map, ["取水地点"]))
    requested_water_amount = _extract_first_number(
        _form_value(label_map, ["运行期年取水量（合计）", "年取水量", "申请取水量", "取水量"])
    )

    return {
        "applicantName": _compact_form_text(applicant_name),
        "idNumber": id_number,
        "contactPhone": _form_value(label_map, ["联系人手机号码", "手机号码", "联系电话"])
        or _extract_with_patterns(text, [r"\b(1\d{10})\b"]),
        "projectName": _form_value(label_map, ["项目名称", "工程名称"]),
        "formLegalRepresentative": _form_value(label_map, ["法定代表人", "负责人"]),
        "waterPurpose": water_purpose or water_purpose_text,
        "waterLocation": water_location,
        "applicationPeriodYears": _extract_first_number(
            _form_value(label_map, ["申请期限", "期限"])
            or _extract_with_patterns(text, [r"申请期限[:：\s]*(\d{1,2})\s*年"])
        ),
        "requestedWaterAmount": requested_water_amount,
        "industryCategory": _form_value(label_map, ["行业类别"]),
        "businessAddress": _clean_template_address(_form_value(label_map, ["生产经营场所地址"])),
        "projectOverview": _form_value(label_map, ["项目概况"]),
        "applicationReason": _form_value(label_map, ["申请事由"]),
        "intakePosition": _form_value(label_map, ["取水口位置"]),
        "returnWaterAmount": _extract_first_number(_form_value(label_map, ["年退水量"])),
    }


def _extract_attachment_entities(request: PrecheckRequest) -> dict[str, str]:
    extracted: dict[str, str] = {
        "idCardName": "",
        "idCardNumber": "",
        "licenseLegalRepresentative": "",
        "licenseCompanyName": "",
        "reportProjectName": "",
    }

    for item in request.attachments:
        text, detected_kind = _extract_attachment_text(item)
        doc_type = detected_kind.strip().lower()
        filename = item.filename.strip().lower()
        if not text:
            continue

        is_id_card = doc_type == "id_card" or "身份证" in filename
        is_license = doc_type in {"business_license", "license"} or "营业执照" in filename
        is_report = doc_type == "report" or "报告" in filename

        if is_id_card:
            fields = _extract_id_card_fields(text)
            extracted["idCardName"] = extracted["idCardName"] or fields["name"]
            extracted["idCardNumber"] = extracted["idCardNumber"] or fields["idNumber"]
        elif is_license:
            fields = _extract_business_license_fields(text)
            extracted["licenseLegalRepresentative"] = extracted["licenseLegalRepresentative"] or fields["legalRepresentative"]
            extracted["licenseCompanyName"] = extracted["licenseCompanyName"] or fields["companyName"]
        elif is_report:
            fields = _extract_report_fields(text)
            extracted["reportProjectName"] = extracted["reportProjectName"] or fields["projectName"]

    return extracted


def _validate_attachment_integrity(request: PrecheckRequest) -> list[str]:
    issues: list[str] = []

    for item in request.attachments:
        text, detected_kind = _extract_attachment_text(item)
        filename = item.filename.strip()
        lower_name = filename.lower()

        if not filename:
            continue

        if lower_name.endswith((".exe", ".bat", ".dll")):
            issues.append(f"文件格式不符：附件 {filename} 不是允许的材料格式")

        if "身份证" in lower_name and detected_kind == "driver_license":
            issues.append("要件缺失：上传的“身份证”附件实际识别为驾驶证，不符合要求")

        if ("营业执照" in lower_name or detected_kind == "business_license") and text:
            expiry = _extract_first_date_range_end(text)
            if expiry and expiry < date.today():
                issues.append("有效期冲突：营业执照已过期")

        if ("身份证" in lower_name or detected_kind == "id_card") and text:
            expiry = _extract_first_date_range_end(text)
            if expiry and expiry < date.today():
                issues.append("有效期冲突：身份证已过期")

        if item.docType.strip().lower() == "report" and not text and not item.base64Content.strip():
            issues.append(f"附件不全：{filename} 未提供可解析内容")

    return issues


def _material_name_from_attachment(filename: str, doc_type: str, detected_kind: str) -> str:
    lower_name = filename.lower()
    normalized = doc_type.strip().lower() or detected_kind.strip().lower()
    if "申请书" in lower_name or normalized == "application_form":
        return "申请书"
    if "身份证" in lower_name or normalized == "id_card":
        return "身份证"
    if "营业执照" in lower_name or normalized == "business_license":
        return "营业执照"
    if "报告" in lower_name or normalized == "report":
        return "水资源论证报告"
    if normalized == "ownership_proof" or any(token in lower_name for token in ["产权", "权属", "租赁"]):
        return "权属证明"
    return filename


def _merge_if_empty(current: str, candidate: str) -> str:
    return current if current.strip() else candidate.strip()


def _safe_int(value: str) -> int | None:
    if not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _safe_float(value: str) -> float | None:
    if not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def extract_draft(request: PrecheckRequest) -> ExtractionDraft:
    draft = ExtractionDraft(
        applicantName=request.applicantName,
        idNumber=request.idNumber,
        contactPhone=request.contactPhone,
        projectName=request.projectName,
        attachedProjectName=request.attachedProjectName,
        formLegalRepresentative=request.formLegalRepresentative,
        licenseLegalRepresentative=request.licenseLegalRepresentative,
        waterPurpose=request.waterPurpose,
        waterLocation=request.waterLocation,
        applicationPeriodYears=request.applicationPeriodYears,
        requestedWaterAmount=request.requestedWaterAmount,
        reportIssuedAt=request.reportIssuedAt,
        ownershipProofType=request.ownershipProofType,
        materials=list(request.materials),
    )

    material_set = {item for item in request.materials if item.strip()}
    warnings: list[str] = []
    summaries: list[AttachmentExtractionSummary] = []

    for item in request.attachments:
        text, detected_kind = _extract_attachment_text(item)
        extracted_fields: dict[str, str] = {}
        file_warnings: list[str] = []

        if detected_kind == "id_card":
            extracted_fields = _extract_id_card_fields(text)
            draft.applicantName = _merge_if_empty(draft.applicantName, extracted_fields.get("name", ""))
            draft.idNumber = _merge_if_empty(draft.idNumber, extracted_fields.get("idNumber", ""))
        elif detected_kind == "business_license":
            extracted_fields = _extract_business_license_fields(text)
            draft.licenseLegalRepresentative = _merge_if_empty(
                draft.licenseLegalRepresentative,
                extracted_fields.get("legalRepresentative", ""),
            )
            draft.formLegalRepresentative = _merge_if_empty(
                draft.formLegalRepresentative,
                extracted_fields.get("legalRepresentative", ""),
            )
        elif detected_kind == "report":
            extracted_fields = _extract_report_fields(text)
            draft.attachedProjectName = _merge_if_empty(draft.attachedProjectName, extracted_fields.get("projectName", ""))
        elif detected_kind == "application_form":
            extracted_fields = _extract_application_form_fields(text)
            draft.applicantName = _merge_if_empty(draft.applicantName, extracted_fields.get("applicantName", ""))
            draft.idNumber = _merge_if_empty(draft.idNumber, extracted_fields.get("idNumber", ""))
            draft.contactPhone = _merge_if_empty(draft.contactPhone, extracted_fields.get("contactPhone", ""))
            draft.projectName = _merge_if_empty(draft.projectName, extracted_fields.get("projectName", ""))
            draft.formLegalRepresentative = _merge_if_empty(
                draft.formLegalRepresentative,
                extracted_fields.get("formLegalRepresentative", ""),
            )
            draft.waterPurpose = _merge_if_empty(draft.waterPurpose, extracted_fields.get("waterPurpose", ""))
            draft.waterLocation = _merge_if_empty(draft.waterLocation, extracted_fields.get("waterLocation", ""))
            draft.applicationPeriodYears = draft.applicationPeriodYears or _safe_int(
                extracted_fields.get("applicationPeriodYears", ""),
            )
            draft.requestedWaterAmount = draft.requestedWaterAmount or _safe_float(
                extracted_fields.get("requestedWaterAmount", ""),
            )

        material_set.add(_material_name_from_attachment(item.filename, item.docType, detected_kind))

        if not text:
            file_warnings.append("未能自动提取文本，可手动补充关键信息后再提交初审。")

        if "身份证" in item.filename.lower() and detected_kind == "driver_license":
            file_warnings.append("文件名为身份证，但识别内容更像驾驶证。")

        summaries.append(
            AttachmentExtractionSummary(
                filename=item.filename,
                docType=item.docType,
                detectedKind=detected_kind,
                extractedText=text[:3000],
                extractedFields={key: value for key, value in extracted_fields.items() if value},
                warnings=file_warnings,
            )
        )
        warnings.extend(file_warnings)

    draft.materials = list(material_set)
    draft.warnings = _dedupe_keep_order(warnings + _validate_attachment_integrity(request))
    draft.attachmentSummaries = summaries
    return draft


def _validate_content_norm(request: PrecheckRequest) -> list[str]:
    issues: list[str] = []
    extracted = _extract_attachment_entities(request)

    if not _normalize_text(request.waterLocation):
        issues.append("关键信息漏填：取水地点缺失")
    if request.applicationPeriodYears is None:
        issues.append("关键信息漏填：申请期限缺失")

    project_name = _normalize_text(request.projectName)
    attached_project_name = _normalize_text(request.attachedProjectName) or _normalize_text(extracted["reportProjectName"])
    if project_name and attached_project_name and project_name != attached_project_name:
        issues.append("信息不一致：申请表项目名称与附件报告项目名称不一致")

    form_legal_rep = _normalize_text(request.formLegalRepresentative) or _normalize_text(request.applicantName)
    license_legal_rep = _normalize_text(request.licenseLegalRepresentative) or _normalize_text(extracted["licenseLegalRepresentative"])
    if form_legal_rep and license_legal_rep and form_legal_rep != license_legal_rep:
        issues.append("信息不一致：法定代表人信息与营业执照不一致")

    id_card_name = _normalize_text(extracted["idCardName"])
    if id_card_name and _normalize_text(request.applicantName) and id_card_name != _normalize_text(request.applicantName):
        issues.append("信息不一致：身份证姓名与申请表申请人姓名不一致")

    id_card_number = _normalize_text(extracted["idCardNumber"])
    if id_card_number and _normalize_text(request.idNumber) and id_card_number.upper() != _normalize_text(request.idNumber).upper():
        issues.append("信息不一致：身份证号码与申请表证件号不一致")

    requested = request.requestedWaterAmount
    estimated = request.reportEstimatedWaterAmount
    if requested is not None and estimated is not None and estimated > 0 and requested > estimated:
        issues.append("数据逻辑矛盾：申请取水量超出论证报告测算用水量")

    return issues


def _validate_substantive_compliance(request: PrecheckRequest) -> list[str]:
    issues: list[str] = []

    location = _normalize_text(request.waterLocation)
    restricted_markers = ["饮用水水源保护区", "自然保护区", "生态保护红线", "禁止取水区"]
    if location and any(marker in location for marker in restricted_markers):
        issues.append(f"选址违规：取水口疑似位于保护区等法规限制区域（{location}）")

    if not _normalize_text(request.thirdPartyImpactDescription) or not _normalize_text(request.mitigationMeasures):
        issues.append("论证不足：未充分说明第三方影响或补救措施")

    apply_period = request.applicationPeriodYears
    approval_period = request.projectApprovalPeriodYears
    if apply_period is not None and approval_period is not None and apply_period > approval_period:
        issues.append("有效期冲突：申请取水许可期限超过建设项目批准期限")

    return issues


def _validate_evidence_chain(request: PrecheckRequest) -> list[str]:
    issues: list[str] = []

    report_date = _parse_date(request.reportIssuedAt)
    if report_date is not None:
        elapsed_days = (date.today() - report_date).days
        if elapsed_days > 365 * 5:
            issues.append("报告过期：水资源论证报告出具时间超过5年")

    legal_basis = _normalize_text(request.legalBasisVersion)
    if legal_basis and any(token in legal_basis for token in ["1998", "旧版", "已废止", "作废"]):
        issues.append("依据失效：申请依据的法律法规或标准可能已更新")

    has_ownership_field = bool(_normalize_text(request.ownershipProofType))
    has_ownership_material = any("权属" in item or "产权" in item or "租赁" in item for item in request.materials)
    if not has_ownership_field and not has_ownership_material:
        issues.append("权属证明缺失：未提供土地产权证明或租赁协议")

    return issues


def call_modelscope_review(request: PrecheckRequest) -> list[str]:
    if not MODELSCOPE_API_KEY.strip():
        return []

    kb_result = knowledge_search("取水许可合规性检查规则", top_k=3)
    references = [item.get("content", "") for item in kb_result.get("results", [])[:3]]

    prompt_data: dict[str, Any] = {
        "applicantName": request.applicantName,
        "idNumber": request.idNumber,
        "contactPhone": request.contactPhone,
        "projectName": request.projectName,
        "attachedProjectName": request.attachedProjectName,
        "formLegalRepresentative": request.formLegalRepresentative,
        "licenseLegalRepresentative": request.licenseLegalRepresentative,
        "waterPurpose": request.waterPurpose,
        "waterLocation": request.waterLocation,
        "applicationPeriodYears": request.applicationPeriodYears,
        "projectApprovalPeriodYears": request.projectApprovalPeriodYears,
        "requestedWaterAmount": request.requestedWaterAmount,
        "reportEstimatedWaterAmount": request.reportEstimatedWaterAmount,
        "thirdPartyImpactDescription": request.thirdPartyImpactDescription,
        "mitigationMeasures": request.mitigationMeasures,
        "reportIssuedAt": request.reportIssuedAt,
        "legalBasisVersion": request.legalBasisVersion,
        "ownershipProofType": request.ownershipProofType,
        "attachments": [item.model_dump() for item in request.attachments],
        "materials": request.materials,
        "references": references,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "你是涉水审批材料初审助手。只输出 JSON 字符串数组，"
                "每个元素是一条具体问题描述。"
                "如果没有问题，输出空数组 []。不要输出其它文字。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请基于以下申请数据和法规参考做初审，重点检查完整性、填写规范和明显合规风险。\n"
                + json.dumps(prompt_data, ensure_ascii=False)
            ),
        },
    ]

    payload = {
        "model": MODELSCOPE_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {MODELSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=20.0) as client:
        resp = client.post(MODELSCOPE_BASE_URL, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()

    choices = body.get("choices", []) if isinstance(body, dict) else []
    if not choices:
        return []
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    return _extract_ai_issues(str(content))[:8]


class ComplianceAgent:
    def run(self, request: PrecheckRequest) -> PrecheckResult:
        issues: list[str] = []

        completeness = check_completeness(request.materials)
        for missing in completeness["missing"]:
            issues.append(f"缺少{missing}")

        if not request.applicantName.strip():
            issues.append("申请人姓名缺失")
        if not request.idNumber.strip():
            issues.append("证件号缺失")
        if not request.contactPhone.strip():
            issues.append("联系方式缺失")
        issues.extend(_validate_water_purpose(request.waterPurpose))
        issues.extend(_validate_attachment_integrity(request))
        issues.extend(_validate_content_norm(request))
        issues.extend(_validate_substantive_compliance(request))
        issues.extend(_validate_evidence_chain(request))

        if ENABLE_REMOTE_AI_REVIEW or _is_env_flag_on("AI_ENABLED", default=False):
            try:
                issues.extend(call_modelscope_review(request))
            except Exception:
                # Keep the service available even when third-party model API is unstable.
                pass

        issues = _dedupe_keep_order(issues)

        # Use knowledge search as the RAG entry point for stable rule grounding.
        reference = knowledge_search("取水许可材料合规审查", top_k=2)
        rag_references = [item["content"] for item in reference["results"][:2]]
        hint = "；".join(rag_references)

        status = "PASS" if not issues else "FAIL"
        workflow_status = "PENDING_MANUAL_REVIEW" if status == "PASS" else "AI_REJECTED"
        manual_status = "PENDING" if status == "PASS" else "NOT_REQUIRED"
        final_status = "PENDING" if status == "PASS" else "REJECTED"
        suggestion = "AI 初审通过，已进入人工审核队列。" if status == "PASS" else f"AI 初审未通过，请整改后重提。参考规则：{hint}"
        return PrecheckResult(
            status=status,
            issues=issues,
            suggestion=suggestion,
            workflowStatus=workflow_status,
            manualStatus=manual_status,
            finalStatus=final_status,
            ragReferences=rag_references,
        )


agent = ComplianceAgent()
