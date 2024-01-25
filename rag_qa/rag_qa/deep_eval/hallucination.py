from typing import List
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from leaf_playground.eval_tools.base import EvalTool, EvalToolConfig
import numpy as np

from .base import DeepEvalBasicEvalTool


# https://docs.confident-ai.com/docs/metrics-hallucination
class DeepEvalHallucinationEvalToolConfig(EvalToolConfig):
    pass


class DeepEvalHallucinationEvalTool(DeepEvalBasicEvalTool[DeepEvalHallucinationEvalToolConfig], EvalTool):
    config_cls = DeepEvalHallucinationEvalToolConfig
    config: config_cls
    metric: HallucinationMetric

    def __init__(self, config: DeepEvalHallucinationEvalToolConfig):
        super().__init__(config)
        self.metric = HallucinationMetric(threshold=0.5)

    async def evaluate(
        self, input_text: str, actual_output: str, context: List[str]
    ) -> dict:
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            context=context,
        )
        self.metric.measure(test_case)

        is_numpy_dtype = isinstance(self.metric.score, np.generic)
        if not is_numpy_dtype:
            return {
                "score": self.metric.score
            }
        else:
            return {
                "score": self.metric.score.item()
            }


__all__ = ["DeepEvalHallucinationEvalTool", "DeepEvalHallucinationEvalToolConfig"]
