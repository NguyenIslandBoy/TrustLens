"""Tests for LLM prompt building and JSON parsing."""
import pytest
from unittest.mock import MagicMock, patch
from src.trustlens.llm.prompt import build_user_prompt, SYSTEM_PROMPT
from src.trustlens.llm.batch_runner import _parse_json, extract_methodology, ExtractionResult
from src.trustlens.parse.segmenter import Segment


class TestPromptBuilding:
    def test_user_prompt_contains_segment_text(self):
        result = build_user_prompt("We surveyed 500 people.")
        assert "We surveyed 500 people." in result

    def test_user_prompt_includes_abstract_when_provided(self):
        result = build_user_prompt("methods text", abstract="This paper studies trust.")
        assert "This paper studies trust." in result

    def test_user_prompt_no_abstract(self):
        result = build_user_prompt("methods text", abstract=None)
        assert "ABSTRACT" not in result

    def test_system_prompt_mentions_json(self):
        assert "JSON" in SYSTEM_PROMPT

    def test_system_prompt_defines_all_fields(self):
        for field in ["study_type", "sample_size", "countries", "trust_measure",
                      "statistical_models", "key_variables", "data_sources",
                      "extraction_confidence"]:
            assert field in SYSTEM_PROMPT


class TestParseJson:
    def test_clean_json(self):
        raw = '{"study_type": "quantitative", "sample_size": 500}'
        result = _parse_json(raw)
        assert result["study_type"] == "quantitative"
        assert result["sample_size"] == 500

    def test_strips_markdown_fences(self):
        raw = '```json\n{"study_type": "qualitative"}\n```'
        result = _parse_json(raw)
        assert result["study_type"] == "qualitative"

    def test_strips_plain_code_fences(self):
        raw = '```\n{"study_type": "mixed"}\n```'
        result = _parse_json(raw)
        assert result["study_type"] == "mixed"

    def test_extracts_json_from_surrounding_text(self):
        raw = 'Here is the result: {"study_type": "quantitative"} Hope that helps!'
        result = _parse_json(raw)
        assert result["study_type"] == "quantitative"

    def test_returns_none_on_invalid_json(self):
        assert _parse_json("This is not JSON at all.") is None

    def test_returns_none_on_empty_string(self):
        assert _parse_json("") is None


class TestExtractMethodology:
    def _make_segment(self) -> Segment:
        return Segment(
            openalex_id="W_test",
            methods_text="We surveyed 1200 participants using OLS regression.",
            abstract="A study of social trust.",
            extraction_method="methods_section",
            char_count=100,
        )

    def test_successful_extraction(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"study_type": "quantitative", "sample_size": 1200, "countries": [], "trust_measure": null, "statistical_models": ["OLS regression"], "key_variables": [], "data_sources": [], "extraction_confidence": "high"}'
        mock_client.chat.completions.create.return_value = mock_response

        result = extract_methodology(self._make_segment(), mock_client, "llama3.1:8b")
        assert result.success is True
        assert result.data["study_type"] == "quantitative"
        assert result.data["sample_size"] == 1200

    def test_handles_llm_connection_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Ollama not running")

        result = extract_methodology(self._make_segment(), mock_client, "llama3.1:8b")
        assert result.success is False
        assert result.error is not None

    def test_never_raises(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("unexpected")

        try:
            result = extract_methodology(self._make_segment(), mock_client, "llama3.1:8b")
        except Exception as e:
            pytest.fail(f"extract_methodology raised: {e}")
    
    def test_downgrades_confidence_on_fallback_with_no_abstract(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"study_type": "qualitative", "sample_size": null, "countries": [], "trust_measure": null, "statistical_models": [], "key_variables": [], "data_sources": [], "extraction_confidence": "high"}'
        mock_client.chat.completions.create.return_value = mock_response

        # Fallback segment with no abstract — high-risk hallucination scenario
        seg = Segment(
            openalex_id="W_fallback",
            methods_text="Some minimal text.",
            abstract=None,
            extraction_method="fallback",
            char_count=20,
        )
        result = extract_methodology(seg, mock_client, "llama3.1:8b")
        assert result.success is True
        assert result.data["extraction_confidence"] == "low"