import weave
from typing import Annotated, List, Optional, Tuple

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from zenml import pipeline, step, log_metadata
from zenml.types import HTMLString

llm = OpenAI(model="gpt-4o")

WANDB_PROJECT = "zenml-document-processing-llms"

CONTRACT_EXTRACT_PROMPT = PromptTemplate(
    """\
You are given contract data below. \
Please extract out relevant information from the contract into the defined schema - the schema is defined as a function call.

{contract_data}
"""
)

CONTRACT_MATCH_PROMPT = PromptTemplate(
    """\
Given the following contract clause and the corresponding relevant guideline text, evaluate the compliance \
and provide a JSON object that matches the ClauseComplianceCheck schema.

**Contract Clause:**
{clause_text}

**Matched Guideline Text(s):**
{guideline_text}
"""
)

COMPLIANCE_REPORT_SYSTEM_PROMPT = """\
You are a compliance reporting assistant. Your task is to generate a final compliance report \
based on the results of clause compliance checks against \
a given set of guidelines.

Analyze the provided compliance results and produce a structured report according to the specified schema. \
Ensure that if there are no noncompliant clauses, the report clearly indicates full compliance.
"""

COMPLIANCE_REPORT_USER_PROMPT = """\
A set of clauses within a contract were checked against GDPR compliance guidelines for the following vendor: {vendor_name}. \
The set of noncompliant clauses are given below.

Each section includes:
- **Clause:** The exact text of the contract clause.
- **Guideline:** The relevant GDPR guideline text.
- **Compliance Status:** Should be `False` for noncompliant clauses.
- **Notes:** Additional information or explanations.

{compliance_results}

Based on the above compliance results, generate a final compliance report following the `ComplianceReport` schema below. \
If there are no noncompliant clauses, the report should indicate that the contract is fully compliant.
"""


# Define Pydantic models
class ContractClause(BaseModel):
    clause_text: str = Field(..., description="The exact text of the clause.")
    mentions_data_processing: bool = Field(
        False,
        description="True if the clause involves personal data collection or usage.",
    )
    mentions_data_transfer: bool = Field(
        False, description="True if the clause involves transferring personal data."
    )
    requires_consent: bool = Field(
        False,
        description="True if the clause explicitly states that user consent is needed.",
    )
    specifies_purpose: bool = Field(
        False,
        description="True if the clause specifies a clear purpose for data handling.",
    )
    mentions_safeguards: bool = Field(
        False, description="True if the clause mentions security measures."
    )


class ContractExtraction(BaseModel):
    vendor_name: Optional[str] = Field(
        None, description="The vendor's name if identifiable."
    )
    effective_date: Optional[str] = Field(
        None, description="The effective date of the agreement."
    )
    governing_law: Optional[str] = Field(
        None, description="The governing law of the contract."
    )
    clauses: List[ContractClause] = Field(..., description="List of extracted clauses.")


class GuidelineMatch(BaseModel):
    guideline_text: str = Field(..., description="Relevant guideline excerpt.")
    similarity_score: float = Field(
        ..., description="Similarity score between 0 and 1."
    )
    relevance_explanation: Optional[str] = Field(
        None, description="Explanation of relevance."
    )


class ClauseComplianceCheck(BaseModel):
    clause_text: str = Field(..., description="The exact text of the clause.")
    matched_guideline: Optional[GuidelineMatch] = Field(
        None, description="Matched guideline."
    )
    compliant: bool = Field(..., description="Indicates compliance status.")
    notes: Optional[str] = Field(None, description="Additional commentary.")


class ComplianceReport(BaseModel):
    vendor_name: Optional[str] = Field(None, description="Vendor's name if identified.")
    overall_compliant: bool = Field(..., description="Overall compliance status.")
    summary_notes: Optional[str] = Field(None, description="Compliance summary.")


# ZenML steps
@step(experiment_tracker="wandb_weave")
def ingest_contract(
    contract_path: str,
) -> Annotated[List[Document], "contract_documents"]:
    """Load and parse contract document using local parser"""
    weave.init(project_name=WANDB_PROJECT)
    documents = SimpleDirectoryReader(input_files=[contract_path]).load_data()
    # Log the document metadata
    log_metadata(
        metadata={
            "num_documents": len(documents),
            "total_chars": sum(len(d.text) for d in documents),
        },
        infer_artifact=True,
    )
    return documents


@step(experiment_tracker="wandb_weave")
def ingest_guidelines(
    guidelines_path: str = "data/gdpr.pdf",
) -> Annotated[VectorStoreIndex, "guidelines_index"]:
    """Create local vector store for guidelines"""
    weave.init(project_name=WANDB_PROJECT)
    documents = SimpleDirectoryReader(input_files=[guidelines_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index


@step(experiment_tracker="wandb_weave")
def extract_clauses(
    documents: Annotated[List[Document], "contract_documents"],
) -> Annotated[ContractExtraction, "extracted_contract_data"]:
    """Extract structured clauses from contract text"""
    weave.init(project_name=WANDB_PROJECT)
    contract_text = "\n".join([d.text for d in documents])
    # Log the prompt being used
    log_metadata(
        metadata={
            "prompt_template": CONTRACT_EXTRACT_PROMPT.template,
            "input_length": len(contract_text),
        },
        infer_artifact=True,
    )
    prediction = llm.structured_predict(
        ContractExtraction, CONTRACT_EXTRACT_PROMPT, contract_data=contract_text
    )
    # Log extraction results
    log_metadata(
        metadata={
            "num_clauses": len(prediction.clauses),
        },
        infer_artifact=True,
    )
    return prediction


@step(experiment_tracker="wandb_weave")
def process_clauses(
    extraction: ContractExtraction, index: VectorStoreIndex, similarity_top_k: int = 2
) -> Annotated[List[ClauseComplianceCheck], "compliance_check_results"]:
    """Process each clause through compliance checks using local vector store"""
    weave.init(project_name=WANDB_PROJECT)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)

    # Log compliance check prompt
    log_metadata(
        metadata={"prompt_template": CONTRACT_MATCH_PROMPT.template},
        infer_artifact=True,
    )

    results = []
    clause_stats = {
        "total_clauses": len(extraction.clauses),
        "processed_clauses": 0,
        "non_compliant_clauses": 0,
    }

    for clause in extraction.clauses:
        try:
            # Retrieve relevant guidelines
            guideline_docs = retriever.retrieve(clause.clause_text)
            guidelines = "\n".join([d.text for d in guideline_docs])

            # Check compliance
            compliance_check = llm.structured_predict(
                ClauseComplianceCheck,
                CONTRACT_MATCH_PROMPT,
                clause_text=clause.clause_text,
                guideline_text=guidelines,
            )

            # Validate that we got a proper ClauseComplianceCheck object
            if not isinstance(compliance_check, ClauseComplianceCheck):
                # Create a default object if the prediction failed
                compliance_check = ClauseComplianceCheck(
                    clause_text=clause.clause_text,
                    compliant=False,
                    notes="Failed to properly analyze compliance",
                )

            results.append(compliance_check)

            # Update stats
            clause_stats["processed_clauses"] += 1
            if not compliance_check.compliant:
                clause_stats["non_compliant_clauses"] += 1

        except Exception as e:
            # Handle any errors gracefully
            print(f"Error processing clause: {str(e)}")
            compliance_check = ClauseComplianceCheck(
                clause_text=clause.clause_text,
                compliant=False,
                notes=f"Error during compliance check: {str(e)}",
            )
            results.append(compliance_check)
            clause_stats["processed_clauses"] += 1
            clause_stats["non_compliant_clauses"] += 1

    # Log clause processing stats
    log_metadata(
        metadata={"clause_processing_stats": clause_stats}, infer_artifact=True
    )
    return results


def generate_html_report(
    report: ComplianceReport, checks: List[ClauseComplianceCheck]
) -> HTMLString:
    """Generate interactive HTML report"""
    html_content = """
    <html><body>
        <h1>Contract Compliance Report</h1>
    """

    # Add overall summary section
    html_content += f"""
        <div class='summary'>
            <h2>Summary</h2>
            <p><strong>Vendor:</strong> {report.vendor_name or "Not specified"}</p>
            <p><strong>Overall Status:</strong> {"✅ Compliant" if report.overall_compliant else "❌ Non-compliant"}</p>
            <p><strong>Notes:</strong> {report.summary_notes or "No additional notes"}</p>
        </div>
        <h2>Detailed Clause Analysis</h2>
        <div class='dashboard'>
    """

    for check in checks:
        html_content += f"""
        <div class='clause'>
            <h3>{check.clause_text[:50]}...</h3>
            <p>Status: {"✅ Compliant" if check.compliant else "❌ Non-compliant"}</p>
            {f"<p>Guideline: {check.matched_guideline.guideline_text[:100]}...</p>" if check.matched_guideline else ""}
            {f"<p>Notes: {check.notes}</p>" if check.notes else ""}
        </div>
        """

    html_content += """
        </div>
        <style>
            .summary { padding: 1rem; margin-bottom: 2rem; background: #f5f5f5; border-radius: 4px; }
            .dashboard { display: grid; gap: 1rem; }
            .clause { padding: 1rem; border: 1px solid #ccc; border-radius: 4px; }
        </style>
    </body></html>
    """

    return HTMLString(html_content)


@step
def generate_report(
    checks: Annotated[List[ClauseComplianceCheck], "compliance_check_results"],
    extraction: Annotated[ContractExtraction, "extracted_contract_data"],
) -> Tuple[
    Annotated[ComplianceReport, "final_compliance_report"],
    Annotated[HTMLString, "html_compliance_report"],
]:
    """Generate final compliance report"""
    non_compliant = [c for c in checks if not c.compliant]
    report = ComplianceReport(
        vendor_name=extraction.vendor_name,
        overall_compliant=len(non_compliant) == 0,
        summary_notes=f"Found {len(non_compliant)} non-compliant clauses",
    )

    return report, generate_html_report(report, checks)


# Define pipeline
@pipeline
def contract_review_pipeline():
    guideline_index = ingest_guidelines()
    contract_docs = ingest_contract("data/vendor_agreement.md")
    extraction = extract_clauses(contract_docs)
    checks = process_clauses(extraction, guideline_index)
    report, html = generate_report(checks, extraction)
    return report, html


if __name__ == "__main__":
    contract_review_pipeline.with_options(
        config_path="configs/agent.yaml",
    )()
