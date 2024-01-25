from typing import List
from pydantic import Field
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from leaf_playground.eval_tools.base import EvalTool

from .base import DeepEvalBasicEvalTool, DeepEvalEvalToolConfig

# https://docs.confident-ai.com/docs/metrics-contextual-relevancy
class DeepEvalContextualRelevancyEvalToolConfig(DeepEvalEvalToolConfig):
    include_reason: bool = Field(default=True)


class DeepEvalContextualRelevancyEvalTool(DeepEvalBasicEvalTool[DeepEvalContextualRelevancyEvalToolConfig], EvalTool):
    config_cls = DeepEvalContextualRelevancyEvalToolConfig
    config: config_cls
    metric: ContextualRelevancyMetric

    def __init__(self, config: DeepEvalContextualRelevancyEvalToolConfig):
        super().__init__(config)
        model = self._create_model_based_on_config(config)
        self.metric = ContextualRelevancyMetric(threshold=0.7, model=model, include_reason=config.include_reason)

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


__all__ = ["DeepEvalContextualRelevancyEvalTool", "DeepEvalContextualRelevancyEvalToolConfig"]
