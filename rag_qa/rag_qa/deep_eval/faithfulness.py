from typing import List
from pydantic import Field
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from leaf_playground.eval_tools.base import EvalTool

from .base import DeepEvalBasicEvalTool, DeepEvalEvalToolConfig

# https://docs.confident-ai.com/docs/metrics-faithfulness
class DeepEvalFaithfulnessEvalToolConfig(DeepEvalEvalToolConfig):
    include_reason: bool = Field(default=True)


class DeepEvalFaithfulnessEvalTool(DeepEvalBasicEvalTool[DeepEvalFaithfulnessEvalToolConfig], EvalTool):
    config_cls = DeepEvalFaithfulnessEvalToolConfig
    config: config_cls
    metric: FaithfulnessMetric

    def __init__(self, config: DeepEvalFaithfulnessEvalToolConfig):
        super().__init__(config)
        model = self._create_model_based_on_config(config)
        self.metric = FaithfulnessMetric(threshold=0.7, model=model, include_reason=config.include_reason)

    async def evaluate(
        self, input_text: str, actual_output: str, retrieval_context: List[str]
    ) -> dict:
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        )
        self.metric.measure(test_case)

        return {
            "score": self.metric.score,
            "reason": self.metric.reason,
        }


__all__ = ["DeepEvalFaithfulnessEvalTool", "DeepEvalFaithfulnessEvalToolConfig"]
