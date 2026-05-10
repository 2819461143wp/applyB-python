from __future__ import annotations

from pydantic import BaseModel, Field


class AttachmentInput(BaseModel):
    docType: str = ""
    filename: str = ""
    mimeType: str = ""
    size: int | None = None
    contentText: str = ""
    base64Content: str = ""


class PrecheckRequest(BaseModel):
    applicantName: str = ""
    idNumber: str = ""
    contactPhone: str = ""
    waterPurpose: str = ""
    materials: list[str] = Field(default_factory=list)
    projectName: str = ""
    attachedProjectName: str = ""
    formLegalRepresentative: str = ""
    licenseLegalRepresentative: str = ""
    waterLocation: str = ""
    applicationPeriodYears: int | None = None
    projectApprovalPeriodYears: int | None = None
    requestedWaterAmount: float | None = None
    reportEstimatedWaterAmount: float | None = None
    thirdPartyImpactDescription: str = ""
    mitigationMeasures: str = ""
    reportIssuedAt: str = ""
    legalBasisVersion: str = ""
    ownershipProofType: str = ""
    attachments: list[AttachmentInput] = Field(default_factory=list)


class PrecheckResult(BaseModel):
    status: str
    issues: list[str]
    suggestion: str
    workflowStatus: str = ""
    manualStatus: str = ""
    finalStatus: str = ""
    ragReferences: list[str] = Field(default_factory=list)


class AttachmentExtractionSummary(BaseModel):
    filename: str
    docType: str
    detectedKind: str
    extractedText: str
    extractedFields: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ExtractionDraft(BaseModel):
    applicantName: str = ""
    idNumber: str = ""
    contactPhone: str = ""
    projectName: str = ""
    attachedProjectName: str = ""
    formLegalRepresentative: str = ""
    licenseLegalRepresentative: str = ""
    waterPurpose: str = ""
    waterLocation: str = ""
    applicationPeriodYears: int | None = None
    requestedWaterAmount: float | None = None
    reportIssuedAt: str = ""
    ownershipProofType: str = ""
    materials: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    attachmentSummaries: list[AttachmentExtractionSummary] = Field(default_factory=list)


class KnowledgeQuery(BaseModel):
    query: str
    top_k: int = 3


class CompletenessCheckRequest(BaseModel):
    materials: list[str]
