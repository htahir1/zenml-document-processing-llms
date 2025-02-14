import base64
import io
import logging
import time
from datetime import datetime
from typing import Annotated, List, Optional, Tuple

import torch
import weave
from huggingface_hub import HfApi
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from openai import OpenAI
from pdf2image import convert_from_path
from pydantic import BaseModel, Field
from rich import print
from zenml import log_metadata, pipeline, step
from zenml.client import Client
from zenml.types import HTMLString

from constants import COMPUTED_RESULTS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

llm = LlamaIndexOpenAI(model="gpt-4")

WANDB_PROJECT = "zenml_llms"
# WANDB_PROJECT = "zenml-document-processing-llms"
ENDPOINT_NAME = "llama-3-2-11b-vision-instruc-egg"


def execute_colpali_rag(soc2_path: str):
    from byaldi import RAGMultiModalModel

    # Initialize Colpali RAG and index the PDF
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    rag = RAGMultiModalModel.from_pretrained("vidore/colpali", device=device_type)
    rag.index(
        input_path=soc2_path,
        index_name="soc2_analysis",
        overwrite=True,
    )
    logger.info("Initialized RAG model and indexed document")
    return rag


def encode_image_to_base64(image) -> str:
    """
    Encode a PIL Image to base64.

    Args:
        image: PIL Image object

    Returns:
        str: Base64 encoded string of the image
    """
    # Create a buffer to store the image
    buffer = io.BytesIO()
    # Save the image as JPEG to the buffer with reduced quality
    image.save(buffer, format="JPEG", quality=10)
    # Get the bytes from the buffer
    image_bytes = buffer.getvalue()
    # Encode to base64
    return base64.b64encode(image_bytes).decode("utf-8")


class SOC2Finding(BaseModel):
    """Represents a single finding from SOC2 analysis"""

    area: str = Field(
        ..., description="Technical area (e.g., 'Data Encryption', 'Access Control')"
    )
    finding: str = Field(..., description="Description of the specific finding")
    risk_level: str = Field(..., description="Risk level: 'Low', 'Medium', 'High'")
    gdpr_relevance: str = Field(
        ..., description="How this finding relates to GDPR compliance"
    )


class SOC2AnalysisResult(BaseModel):
    """Complete SOC2 analysis results"""

    vendor_name: str = Field(..., description="Name of the vendor from SOC2 report")
    analysis_date: str = Field(..., description="Date of analysis")
    report_period: str = Field(..., description="Period covered by SOC2 report")
    key_findings: List[SOC2Finding] = Field(..., description="List of key findings")
    overall_risk_assessment: str = Field(
        ..., description="Overall risk assessment summary"
    )


# Add SOC2 Analysis Prompts
SOC2_FINDING_PROMPT = PromptTemplate(
    """\
Analyze the following findings about {area} from a SOC2 report and create a structured finding.
Focus on GDPR relevance and assess the risk level.

Findings from pages {pages}:
{findings}

Create a finding that includes:
1. A clear description of the control or measure
2. Its relevance to GDPR compliance
3. A risk assessment (Low/Medium/High)
"""
)


@step(experiment_tracker="wandb_weave")
def analyze_soc2_report(
    soc2_path: str,
    colpali_demo_mode: bool = True,
) -> Annotated[SOC2AnalysisResult, "soc2_analysis"]:
    """Analyze SOC2 report using Colpali RAG and HF-based vision inference endpoint"""
    weave.init(project_name=WANDB_PROJECT)

    logger.info(f"Starting SOC2 analysis for {soc2_path}")

    images = convert_from_path(soc2_path)
    logger.info(f"Converted PDF to {len(images)} images")

    rag = None
    if not colpali_demo_mode:
        rag = execute_colpali_rag(soc2_path)

    # Standard queries for SOC2 analysis
    queries = {
        "encryption": [
            "What encryption standards are used for data at rest and in transit? Are there any exceptions to encryption policies?",
        ],
        "access_control": [
            "What access control and authentication mechanisms are in place? How is privileged access managed?",
        ],
        "data_retention": [
            "What are the data retention and deletion policies? How is data deletion handled?",
        ],
        "monitoring": [
            "What security monitoring tools and incident detection mechanisms are in place?",
        ],
        "subprocessors": [
            "How are third-party service providers and subprocessors managed and monitored for compliance?",
        ],
    }

    zenml_client = Client()
    hf_token = zenml_client.get_secret("huggingface_creds").secret_values["token"]

    # Collect findings for each area
    findings = []
    for area, area_queries in queries.items():
        area_results = []
        for query in area_queries:
            try:
                if rag:
                    # Get relevant pages using RAG
                    results = rag.search(query, k=3)
                else:
                    # load from disk
                    results = COMPUTED_RESULTS[area]

                if results and len(results) > 0:
                    hf_api = HfApi()
                    endpoint = hf_api.get_inference_endpoint(
                        ENDPOINT_NAME, namespace="zenml"
                    )
                    while endpoint.status != "running":
                        logger.info(
                            f"Endpoint status is '{endpoint.status}'. Waiting for endpoint to be ready..."
                        )
                        time.sleep(10)
                        endpoint = hf_api.get_inference_endpoint(
                            ENDPOINT_NAME, namespace="zenml"
                        )
                    base_url = endpoint.url
                    vision_client = OpenAI(base_url=base_url + "/v1", api_key=hf_token)

                    for result in results:
                        # Get the page number and encode image as base64
                        page_num = result.page_num - 1  # Convert to 0-based index
                        resized_image = images[page_num].copy()
                        resized_image.thumbnail((128, 128))
                        base64_image = encode_image_to_base64(resized_image)

                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": f"Analyze this page from a SOC2 report and tell me: {query}",
                                    },
                                ],
                            }
                        ]

                        chat_completion = vision_client.chat.completions.create(
                            model="tgi",
                            messages=messages,
                            max_tokens=200,
                        )
                        response = chat_completion.choices[0].message.content

                        area_results.append(
                            {
                                "query": query,
                                "findings": response,
                                "page_num": result.page_num,
                            }
                        )
                        logger.debug(
                            f"Query response for {area}: {query} -> {response[:100]}..."
                        )

            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                continue

        # Synthesize findings for this area using GPT-4
        if area_results:
            try:
                # Format findings for the prompt
                pages = ", ".join(str(r["page_num"]) for r in area_results)
                findings_text = "\n".join(
                    f"- {r['findings']} (Page {r['page_num']})" for r in area_results
                )

                finding = llm.structured_predict(
                    SOC2Finding,
                    SOC2_FINDING_PROMPT,
                    area=area,
                    pages=pages,
                    findings=findings_text,
                )
                findings.append(finding)
                logger.info(f"Synthesized finding for {area}: {finding}")
            except Exception as e:
                logger.error(f"Error synthesizing findings for area '{area}': {str(e)}")
                findings.append(
                    SOC2Finding(
                        area=area,
                        finding="Unable to analyze findings",
                        risk_level="High",
                        gdpr_relevance="Analysis failed - manual review required",
                    )
                )

    # Create final analysis result
    analysis = SOC2AnalysisResult(
        vendor_name="Kolide",
        analysis_date=datetime.now().strftime("%Y-%m-%d"),
        report_period="2024",
        key_findings=findings,
        overall_risk_assessment=_generate_overall_assessment(findings),
    )

    logger.info(f"Completed SOC2 analysis with {len(findings)} findings")

    # Log analysis metrics
    log_metadata(
        metadata={
            "num_findings": len(findings),
            "areas_analyzed": list(queries.keys()),
            "vendor": analysis.vendor_name,
        },
        infer_artifact=True,
    )

    return analysis


def _generate_overall_assessment(findings: List[SOC2Finding]) -> str:
    """Generate an overall risk assessment based on individual findings"""
    risk_levels = [f.risk_level for f in findings]
    high_risks = risk_levels.count("High")
    medium_risks = risk_levels.count("Medium")

    if high_risks > 0:
        return f"High Risk: Found {high_risks} high-risk and {medium_risks} medium-risk findings that require attention."
    elif medium_risks > 0:
        return f"Medium Risk: Found {medium_risks} medium-risk findings that should be reviewed."
    else:
        return "Low Risk: No significant compliance concerns identified."


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
    report: ComplianceReport,
    checks: List[ClauseComplianceCheck],
    soc2_analysis: Optional[SOC2AnalysisResult] = None,
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
            <p><strong>Overall Status:</strong> {"✅ Compliant" if report.overall_compliant else "❌ Non-compliant"}</p>
            <p><strong>Notes:</strong> {report.summary_notes or "No additional notes"}</p>
        </div>
    """

    # Add SOC2 analysis section if available
    if soc2_analysis:
        html_content += f"""
            <div class='soc2-analysis'>
                <h2>SOC2 Technical Controls Analysis</h2>
                <div class='summary'>
                    <p><strong>Analysis Date:</strong> {soc2_analysis.analysis_date}</p>
                    <p><strong>Overall Risk Assessment:</strong> {soc2_analysis.overall_risk_assessment}</p>
                </div>
                <div class='findings'>
                    <h3>Key Findings</h3>
                    <div class='findings-grid'>
        """

        # Risk level color coding
        risk_colors = {"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}

        for finding in soc2_analysis.key_findings:
            risk_color = risk_colors.get(finding.risk_level, "#9E9E9E")
            html_content += f"""
                <div class='finding-card'>
                    <div class='finding-header' style='background-color: {risk_color}'>
                        <h4>{finding.area}</h4>
                        <span class='risk-level'>{finding.risk_level} Risk</span>
                    </div>
                    <div class='finding-content'>
                        <p><strong>Finding:</strong> {finding.finding}</p>
                        <p><strong>GDPR Relevance:</strong> {finding.gdpr_relevance}</p>
                    </div>
                </div>
            """

        html_content += """
                    </div>
                </div>
            </div>
        """

    # Add clause analysis section
    html_content += """
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
            body { font-family: system-ui, -apple-system, sans-serif; line-height: 1.5; }
            .summary { padding: 1rem; margin-bottom: 2rem; background: #f5f5f5; border-radius: 4px; }
            .dashboard { display: grid; gap: 1rem; }
            .clause { padding: 1rem; border: 1px solid #ccc; border-radius: 4px; }
            .soc2-analysis { margin: 2rem 0; }
            .findings-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin: 1rem 0; }
            .finding-card { border: 1px solid #ccc; border-radius: 4px; overflow: hidden; background: white; }
            .finding-header { padding: 0.75rem; color: white; display: flex; justify-content: space-between; align-items: center; }
            .finding-header h4 { margin: 0; }
            .finding-content { padding: 1rem; }
            .risk-level { font-weight: bold; }
        </style>
    </body></html>
    """

    return HTMLString(html_content)


@step(experiment_tracker="wandb_weave")
def generate_report(
    checks: Annotated[List[ClauseComplianceCheck], "compliance_check_results"],
    extraction: Annotated[ContractExtraction, "extracted_contract_data"],
    soc2_analysis: Optional[SOC2AnalysisResult] = None,
) -> Tuple[
    Annotated[ComplianceReport, "final_compliance_report"],
    Annotated[HTMLString, "html_report"],
]:
    """Generate final compliance report"""
    non_compliant = [c for c in checks if not c.compliant]
    report = ComplianceReport(
        vendor_name=extraction.vendor_name,
        overall_compliant=len(non_compliant) == 0,
        summary_notes=f"Found {len(non_compliant)} non-compliant clauses",
    )

    # Add SOC2 analysis to the report
    if soc2_analysis:
        report.vendor_name = soc2_analysis.vendor_name
        report.overall_compliant = len(non_compliant) == 0
        report.summary_notes = f"Found {len(non_compliant)} non-compliant clauses. SOC2 analysis: {soc2_analysis.overall_risk_assessment}"

    return report, generate_html_report(report, checks, soc2_analysis)


@pipeline(enable_cache=False)
def contract_review_pipeline(
    contract_path: str = "data/vendor_agreement.md",
    soc2_path: Optional[str] = None,
    colpali_demo_mode: bool = True,
):
    """Enhanced contract review pipeline with optional SOC2 analysis"""
    guideline_index = ingest_guidelines()
    contract_docs = ingest_contract(contract_path)
    extraction = extract_clauses(contract_docs)
    checks = process_clauses(extraction, guideline_index)

    soc2_analysis = None
    if soc2_path:
        soc2_analysis = analyze_soc2_report(soc2_path, colpali_demo_mode)

    report, html = generate_report(checks, extraction, soc2_analysis)
    return report, html


if __name__ == "__main__":
    contract_review_pipeline.with_options(config_path="configs/agent.yaml")(
        contract_path="data/vendor_agreement.md",
        soc2_path="data/kolide-soc2.pdf",
        colpali_demo_mode=True,
    )
