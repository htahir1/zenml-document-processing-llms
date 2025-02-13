from typing import Annotated, List, Optional, Dict, Any, Tuple
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from zenml import pipeline, step, log_metadata
from zenml.types import HTMLString
from huggingface_hub import InferenceClient
from PIL import Image
import pdf2image

llm = OpenAI(model="gpt-4o")

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


# Data Models
class FinancialMetrics(BaseModel):
    # Balance Sheet Metrics
    total_assets: Optional[float] = Field(None, description="Total assets")
    total_liabilities: Optional[float] = Field(None, description="Total liabilities")
    shareholders_equity: Optional[float] = Field(
        None, description="Total shareholder's equity"
    )

    # Income Statement Metrics
    revenue: Optional[float] = Field(None, description="Total revenue")
    operating_income: Optional[float] = Field(None, description="Operating income")
    net_income: Optional[float] = Field(None, description="Net income")

    # Cash Flow Metrics
    operating_cash_flow: Optional[float] = Field(
        None, description="Operating cash flow"
    )
    investing_cash_flow: Optional[float] = Field(
        None, description="Cash flow from investing"
    )
    financing_cash_flow: Optional[float] = Field(
        None, description="Cash flow from financing"
    )

    # Key Ratios
    current_ratio: Optional[float] = Field(None, description="Current ratio")
    debt_to_equity: Optional[float] = Field(None, description="Debt to equity ratio")
    profit_margin: Optional[float] = Field(None, description="Profit margin")


class FinancialAnalysisReport(BaseModel):
    company_name: str = Field(..., description="Name of the company")
    report_date: datetime = Field(..., description="Date of the report")
    metrics: FinancialMetrics = Field(..., description="Financial metrics")
    key_insights: List[str] = Field(
        default_factory=list, description="Key insights from the analysis"
    )
    red_flags: List[str] = Field(
        default_factory=list, description="Potential red flags"
    )
    trend_analysis: Dict[str, Any] = Field(
        default_factory=dict, description="Trend analysis results"
    )


# Initialize model clients
vision_client = InferenceClient(model="meta-llama/Llama-3.2-11B-Vision-Instruct")
analysis_client = InferenceClient(model="mistralai/Mistral-Small-24B-Instruct-2501")


# ZenML steps
@step
def ingest_contract(
    contract_path: str,
) -> Annotated[List[Document], "contract_documents"]:
    """Load and parse contract document using local parser"""
    # weave.init(project_name="zenml_llms")
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


@step
def ingest_guidelines(
    guidelines_path: str = "data/gdpr.pdf",
) -> Annotated[VectorStoreIndex, "guidelines_index"]:
    """Create local vector store for guidelines"""
    # weave.init(project_name="zenml_llms")
    documents = SimpleDirectoryReader(input_files=[guidelines_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index


@step
def extract_clauses(
    documents: Annotated[List[Document], "contract_documents"],
) -> Annotated[ContractExtraction, "extracted_contract_data"]:
    """Extract structured clauses from contract text"""
    # weave.init(project_name="zenml_llms")
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


@step
def process_clauses(
    extraction: ContractExtraction, index: VectorStoreIndex, similarity_top_k: int = 2
) -> Annotated[List[ClauseComplianceCheck], "compliance_check_results"]:
    """Process each clause through compliance checks using local vector store"""
    # weave.init(project_name="zenml_llms")
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
        results.append(compliance_check)

        # Update stats
        clause_stats["processed_clauses"] += 1
        if not compliance_check.compliant:
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


@step
def convert_pdf_to_images(
    pdf_path: str,
) -> Annotated[List[Image.Image], "document_images"]:
    """Convert PDF pages to images."""
    # weave.init(project_name="zenml_llms")

    # Convert PDF to images
    images = pdf2image.convert_from_path(pdf_path)

    # Log metadata about the conversion
    log_metadata(
        metadata={
            "num_pages": len(images),
            "image_sizes": [f"{img.size[0]}x{img.size[1]}" for img in images],
        },
        infer_artifact=True,
    )

    return images


@step
def process_all_pages(
    images: Annotated[List[Image.Image], "document_images"],
) -> Annotated[List[Dict[str, Any]], "all_page_data"]:
    """Process all pages of the document in a single step."""
    extracted_data = []

    for i, image in enumerate(images):
        # Define the extraction prompt - following DataCamp's more specific approach
        prompt = f"""You are a financial data extraction expert. For this page (page {i + 1}) of a financial report:

1. First, identify if this page contains any financial tables or statements.
2. If there are tables:
   - Extract all numerical data from tables
   - Identify the type of statement (Income Statement, Balance Sheet, Cash Flow)
   - Preserve the hierarchical structure of the data
3. For text content:
   - Extract any mentioned financial metrics
   - Note any forward-looking statements
   - Capture any significant footnotes

Format the output as JSON with this structure:
{{
    "page_type": "financial_statement|text|mixed",
    "financial_statements": [{{
        "type": "income_statement|balance_sheet|cash_flow",
        "data": {{
            "line_items": [{{
                "name": "item_name",
                "value": numeric_value,
                "period": "period_identifier"
            }}]
        }}
    }}],
    "metrics": {{
        "metric_name": value
    }},
    "forward_looking": ["statement1", "statement2"],
    "footnotes": ["note1", "note2"]
}}"""

        # Process with Llama Vision
        response = vision_client.post(
            json={"inputs": [{"image": image, "text": prompt}]}
        )

        # Log metadata about the extraction
        log_metadata(
            metadata={
                "page_number": i + 1,
                "response_length": len(str(response)),
            },
            infer_artifact=True,
        )

        extracted_data.append(response)

    return extracted_data


@step
def analyze_financial_data(
    extracted_data: Annotated[List[Dict[str, Any]], "all_page_data"],
) -> Annotated[Dict[str, Any], "analysis_results"]:
    """Analyze the extracted financial data using Mistral Small."""

    # First, organize the extracted data by statement type
    organized_data = {
        "income_statement": [],
        "balance_sheet": [],
        "cash_flow": [],
        "metrics": {},
        "forward_looking": [],
        "footnotes": [],
    }

    for page_data in extracted_data:
        if "financial_statements" in page_data:
            for statement in page_data["financial_statements"]:
                organized_data[statement["type"]].extend(
                    statement["data"]["line_items"]
                )
        if "metrics" in page_data:
            organized_data["metrics"].update(page_data["metrics"])
        if "forward_looking" in page_data:
            organized_data["forward_looking"].extend(page_data["forward_looking"])
        if "footnotes" in page_data:
            organized_data["footnotes"].extend(page_data["footnotes"])

    system_prompt = """You are a financial analyst expert. Given the organized financial data from a company report:

1. Analyze the Income Statement:
   - Calculate YoY growth rates
   - Identify margin trends
   - Flag any unusual changes

2. Analyze the Balance Sheet:
   - Assess liquidity ratios
   - Evaluate capital structure
   - Check for any concerning trends

3. Analyze Cash Flows:
   - Evaluate operational efficiency
   - Assess investment activities
   - Review financing decisions

4. Provide:
   - Key performance indicators
   - Risk factors
   - Forward-looking insights
   - Strategic recommendations

Format your response as JSON with clear sections for metrics, insights, risks, and recommendations."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Analyze this organized financial data: {organized_data}",
        },
    ]

    response = analysis_client.post(
        json={
            "model": "mistralai/Mistral-Small-24B-Instruct-2501",
            "messages": messages,
            "temperature": 0,
        }
    )

    # Log metadata about the analysis
    log_metadata(
        metadata={
            "num_pages_analyzed": len(extracted_data),
            "analysis_length": len(str(response)),
        },
        infer_artifact=True,
    )

    return response


@step
def generate_report(
    analysis: Annotated[Dict[str, Any], "analysis_results"],
) -> Tuple[
    Annotated[FinancialAnalysisReport, "final_report"],
    Annotated[HTMLString, "html_report"],
]:
    """Generate the final analysis report."""

    # Create the structured report
    report = FinancialAnalysisReport(
        company_name="Microsoft",  # We could extract this from the document
        report_date=datetime.now(),  # We could extract this from the document
        metrics=FinancialMetrics(**analysis.get("metrics", {})),
        key_insights=analysis.get("key_insights", []),
        red_flags=analysis.get("red_flags", []),
        trend_analysis=analysis.get("trend_analysis", {}),
    )

    # Generate HTML report
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ background: #f5f5f5; padding: 10px; margin: 5px; border-radius: 5px; }}
            .insight {{ color: #2c5282; }}
            .red-flag {{ color: #c53030; }}
        </style>
    </head>
    <body>
        <h1>{report.company_name} Financial Analysis</h1>
        <p>Report generated on: {report.report_date}</p>
        
        <h2>Key Metrics</h2>
        <div class="metrics">
            {_generate_metrics_html(report.metrics)}
        </div>
        
        <h2>Key Insights</h2>
        <ul>
            {_generate_list_html(report.key_insights, "insight")}
        </ul>
        
        <h2>Risk Factors</h2>
        <ul>
            {_generate_list_html(report.red_flags, "red-flag")}
        </ul>
        
        <h2>Trend Analysis</h2>
        <div class="trends">
            {_generate_trends_html(report.trend_analysis)}
        </div>
    </body>
    </html>
    """

    return report, HTMLString(html_content)


def _generate_metrics_html(metrics: FinancialMetrics) -> str:
    """Helper function to generate HTML for metrics."""
    html = ""
    for field_name, value in metrics.dict().items():
        if value is not None:
            html += f'<div class="metric"><strong>{field_name}:</strong> {value}</div>'
    return html


def _generate_list_html(items: List[str], class_name: str) -> str:
    """Helper function to generate HTML for lists."""
    return "\n".join([f'<li class="{class_name}">{item}</li>' for item in items])


def _generate_trends_html(trends: Dict[str, Any]) -> str:
    """Helper function to generate HTML for trend analysis."""
    html = ""
    for metric, trend in trends.items():
        html += f'<div class="trend"><strong>{metric}:</strong> {trend}</div>'
    return html


@pipeline
def financial_analysis_pipeline(pdf_path: str = "data/microsoft-q2-2025-small.pdf"):
    """Main pipeline for financial document analysis."""
    images = convert_pdf_to_images(pdf_path)

    # Process all pages in a single step
    extracted_data = process_all_pages(images)

    # Analyze the extracted data
    analysis = analyze_financial_data(extracted_data)

    # Generate the final report
    generate_report(analysis)


if __name__ == "__main__":
    financial_analysis_pipeline()
