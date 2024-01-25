from typing import List
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from leaf_playground.eval_tools.base import EvalTool

from .base import DeepEvalBasicEvalTool, DeepEvalEvalToolConfig


# https://docs.confident-ai.com/docs/metrics-llm-evals
class DeepEvalGEvalEvalToolConfig(DeepEvalEvalToolConfig):
    pass


class DeepEvalGEvalEvalTool(DeepEvalBasicEvalTool[DeepEvalGEvalEvalToolConfig], EvalTool):
    config_cls = DeepEvalGEvalEvalToolConfig
    config: config_cls

    def __init__(self, config: DeepEvalGEvalEvalToolConfig):
        super().__init__(config)
        self.model = self._create_model_based_on_config(config)

    async def evaluate(
        self, name: str, criteria: str, evaluation_params: List[LLMTestCaseParams],
        input_text: str, actual_output: str,
        evaluation_steps: List[str] = None
    ) -> dict:
        metric_init_params = {
            "name": name,
            "criteria": criteria,
            "evaluation_params": evaluation_params,
            "threshold": 0.5,
            "model": self.model,
        }
        if evaluation_steps is not None:
            metric_init_params["evaluation_steps"] = evaluation_steps
        metric = GEval(**metric_init_params)
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
        )
        metric.measure(test_case)

        return {
            "score": metric.score,
            "reason": metric.reason,
        }


__all__ = ["DeepEvalGEvalEvalTool", "DeepEvalGEvalEvalToolConfig"]
