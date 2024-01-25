from typing import Optional, List
from pydantic import Field
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from leaf_playground.eval_tools.base import EvalTool

from .base import DeepEvalBasicEvalTool, DeepEvalEvalToolConfig


# https://docs.confident-ai.com/docs/metrics-summarization
class DeepEvalSummarizationEvalToolConfig(DeepEvalEvalToolConfig):
    assessment_questions: Optional[List[str]] = Field(
        default=None,
        description="a list of close-ended questions that can be answered with either a 'yes' or a 'no'. These are questions you want your summary to be able to ideally answer, and is especially helpful if you already know what a good summary for your use case looks like. If assessment_questions is not provided, we will generate a set of assessment_questions for you at evaluation time. The assessment_questions are used to calculate the inclusion_score."
    )
    n: Optional[int] = Field(
        default=None,
        description="the number of questions to generate when calculating the alignment_score and inclusion_score, defaulted to 5."
    )


class DeepEvalSummarizationEvalTool(DeepEvalBasicEvalTool[DeepEvalSummarizationEvalToolConfig], EvalTool):
    config_cls = DeepEvalSummarizationEvalToolConfig
    config: config_cls

    def __init__(self, config: DeepEvalSummarizationEvalToolConfig):
        super().__init__(config)
        self.model = self._create_model_based_on_config(config)

    async def evaluate(
        self, input_text: str, actual_output: str
    ) -> dict:
        metric_init_params = {
            "threshold": 0.5,
            "model": self.model,
        }
        if self.config.assessment_questions is not None:
            metric_init_params["assessment_questions"] = self.config.assessment_questions
        if self.config.n is not None:
            metric_init_params["n"] = self.config.n
        metric = SummarizationMetric(**metric_init_params)
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
        )
        metric.measure(test_case)

        return {
            "score": metric.score,
            "alignment_score": metric.alignment_score,
            "inclusion_score": metric.inclusion_score,
        }


__all__ = ["DeepEvalSummarizationEvalTool", "DeepEvalSummarizationEvalToolConfig"]
