"""
src/trustlens/llm/prompt.py
=============================
Prompt templates for structured methodology extraction.

Design:
- System prompt defines the schema strictly
- User prompt injects the paper segment
- Output must be valid JSON — no markdown, no preamble
"""

SYSTEM_PROMPT = """You are a research methodology extraction assistant.
Your job is to extract structured methodological information from academic paper text.

You must respond with ONLY a valid JSON object — no explanation, no markdown, no code blocks.
If a field cannot be determined from the text, use null.

The JSON schema you must follow exactly:
{
  "study_type": "quantitative" | "qualitative" | "mixed" | null,
  "sample_size": <integer or null>,
  "countries": [<list of country names as strings>],
  "trust_measure": <string describing how trust was measured, or null>,
  "statistical_models": [<list of model/method names as strings>],
  "key_variables": [<list of key variable names as strings>],
  "data_sources": [<list of dataset or survey names as strings>],
  "extraction_confidence": "high" | "medium" | "low"
}

Rules:
- study_type: quantitative if uses statistical models or surveys with numeric outcomes,
  qualitative if interviews/ethnography/discourse analysis, mixed if both
- sample_size: the number of participants, observations, or cases — integer only
- countries: where the study was conducted — use full country names
- trust_measure: be specific, e.g. "Rosenberg trust scale", "WVS single item", "custom survey"
- statistical_models: e.g. "OLS regression", "SEM", "multilevel modelling", "logistic regression"
- extraction_confidence: high if methods section is clear, medium if inferred, low if very uncertain
"""


def build_user_prompt(segment_text: str, abstract: str | None = None) -> str:
    """Build the user-turn prompt from a paper segment."""
    parts = []

    if abstract:
        parts.append(f"ABSTRACT:\n{abstract}\n")

    parts.append(f"PAPER TEXT:\n{segment_text}")
    parts.append("\nExtract the methodology information as JSON:")

    return "\n".join(parts)