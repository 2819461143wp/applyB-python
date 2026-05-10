# FastAPI AI Service

## Run

```bash
conda activate nba
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## ModelScope AI Review

Current code calls ModelScope chat-completions directly in `app/agent.py` and uses GLM model for review.

Notes:

- If remote AI call fails, the service falls back to local rules and still returns a stable response.
- AI output is parsed into issue list and merged with local rule checks.

## Precheck Key Fields

The precheck endpoint now supports extra fields for content consistency, substantive compliance and evidence-chain checks:

- projectName / attachedProjectName
- formLegalRepresentative / licenseLegalRepresentative
- waterLocation
- applicationPeriodYears / projectApprovalPeriodYears
- requestedWaterAmount / reportEstimatedWaterAmount
- thirdPartyImpactDescription / mitigationMeasures
- reportIssuedAt
- legalBasisVersion
- ownershipProofType
- attachments (for OCR extraction)

Attachment payload example:

- docType: id_card | business_license | report | unknown
- filename: attachment file name
- contentText: OCR text content

## Test

```bash
conda activate nba
pytest -q
```
