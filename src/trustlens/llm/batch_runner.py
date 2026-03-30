"""
src/trustlens/llm/batch_runner.py
===================================
Send paper segments to the local LLM and parse structured JSON output.

Uses the OpenAI-compatible Ollama endpoint — same client as ChargeGPT.
Never raises — all errors returned as failed ExtractionResult.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

from src.trustlens.llm.prompt import SYSTEM_PROMPT, build_user_prompt
from src.trustlens.parse.segmenter import Segment

log = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    openalex_id: str
    success: bool
    data: dict = field(default_factory=dict)
    raw_response: str = ""
    error: str | None = None
    latency_seconds: float = 0.0


def extract_methodology(
    segment: Segment,
    client: OpenAI,
    model: str,
    retries: int = 2,
) -> ExtractionResult:
    """
    Extract methodology metadata from a single paper segment.

    Args:
        segment:  Segment from the segmenter
        client:   OpenAI-compatible client pointed at Ollama
        model:    Model name e.g. "llama3.1:8b"
        retries:  Number of retry attempts on JSON parse failure

    Returns:
        ExtractionResult — never raises
    """
    user_prompt = build_user_prompt(segment.methods_text, segment.abstract)

    for attempt in range(1, retries + 2):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,    # deterministic — we want consistent schema
                max_tokens=600,
            )
            latency = time.perf_counter() - start
            raw = response.choices[0].message.content or ""

            data = _parse_json(raw)
            if data is not None:
                # Downgrade confidence on fallback extractions with no abstract
                if (segment.extraction_method == "fallback"
                        and not segment.abstract
                        and data.get("extraction_confidence") == "high"):
                    data["extraction_confidence"] = "low"
                    log.warning(f"Confidence downgraded to low — fallback with no abstract [{segment.openalex_id}]")

                log.info(f"Extracted [{segment.openalex_id}] in {latency:.1f}s (attempt {attempt})")
                return ExtractionResult(
                    openalex_id=segment.openalex_id,
                    success=True,
                    data=data,
                    raw_response=raw,
                    latency_seconds=latency,
                )

            log.warning(f"JSON parse failed [{segment.openalex_id}] attempt {attempt} — retrying")

        except Exception as e:
            latency = time.perf_counter() - start
            log.error(f"LLM call failed [{segment.openalex_id}]: {e}")
            return ExtractionResult(
                openalex_id=segment.openalex_id,
                success=False,
                error=str(e),
                latency_seconds=latency,
            )

    return ExtractionResult(
        openalex_id=segment.openalex_id,
        success=False,
        error="JSON extraction failed after all retries",
        raw_response=raw,
    )


def batch_extract(
    segments: list[Segment],
    client: OpenAI,
    model: str,
    delay_seconds: float = 1.0,
) -> tuple[list[ExtractionResult], list[ExtractionResult]]:
    """
    Run extraction over a list of segments.

    Args:
        delay_seconds: Pause between calls — prevents Ollama overload

    Returns:
        (successes, failures)
    """
    successes, failures = [], []

    for i, segment in enumerate(segments, 1):
        log.info(f"[{i}/{len(segments)}] Extracting: {segment.openalex_id}")
        result = extract_methodology(segment, client, model)

        if result.success:
            successes.append(result)
        else:
            failures.append(result)

        if i < len(segments):
            time.sleep(delay_seconds)

    log.info(f"Batch extraction complete — {len(successes)} succeeded, {len(failures)} failed")
    return successes, failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict | None:
    """
    Robustly parse JSON from LLM output.
    Handles cases where the model wraps output in markdown code blocks.
    """
    # Strip markdown code fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Last resort: find first { ... } block in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None