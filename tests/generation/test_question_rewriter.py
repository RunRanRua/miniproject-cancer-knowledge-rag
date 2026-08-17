from unittest.mock import patch
from langchain_core.language_models import LLM
from backend.generation.question_rewriter import (
    create_question_rewriter,
)


class MockLLM(LLM):
    """Mock LLM that returns a fixed response without making API calls"""
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _call(self, prompt, stop=None, **kwargs):
        return "What is the diagnosis process for bladder cancer?"


def test_question_rewriter():
    # Mock the get_llm function to return our mock LLM
    with patch("backend.generation.question_rewriter.get_llm") as mock_get_llm:
        mock_get_llm.return_value = MockLLM()
        
        rewriter = create_question_rewriter()

        result = rewriter.invoke(
            {
                "history": (
                    "Human: What are the symptoms of bladder cancer?\n"
                    "AI: Bladder cancer may cause several symptoms."
                ),
                "question": "What about diagnosis?",
            }
        )

        assert isinstance(result, str)
        assert result.strip()