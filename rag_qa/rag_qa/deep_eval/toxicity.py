from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCase
from leaf_playground.eval_tools.base import EvalTool, EvalToolConfig
import numpy as np

from .base import DeepEvalBasicEvalTool


# https://docs.confident-ai.com/docs/metrics-toxicity
class DeepEvalToxicityEvalToolConfig(EvalToolConfig):
    pass


class DeepEvalToxicityEvalTool(DeepEvalBasicEvalTool[DeepEvalToxicityEvalToolConfig], EvalTool):
    config_cls = DeepEvalToxicityEvalToolConfig
    config: config_cls
    metric: ToxicityMetric

    def __init__(self, config: DeepEvalToxicityEvalToolConfig):
        super().__init__(config)
        self.metric = ToxicityMetric(threshold=0.5)

    async def evaluate(
        self, input_text: str, actual_output: str
    ) -> dict:
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
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


__all__ = ["DeepEvalToxicityEvalTool", "DeepEvalToxicityEvalToolConfig"]
