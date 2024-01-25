from typing import List
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCase
from leaf_playground.eval_tools.base import EvalTool

from .base import DeepEvalBasicEvalTool, DeepEvalEvalToolConfig

# https://docs.confident-ai.com/docs/metrics-ragas
class DeepEvalRagasEvalToolConfig(DeepEvalEvalToolConfig):
    pass


class DeepEvalRagasEvalTool(DeepEvalBasicEvalTool[DeepEvalRagasEvalToolConfig], EvalTool):
    metric: RagasMetric

    def __init__(self, config: DeepEvalRagasEvalToolConfig):
        super().__init__(config)
        model = self._create_model_based_on_config(config)
        self.metric = RagasMetric(threshold=0.5, model=model)

    async def evaluate(
        self, input_text: str, actual_output: str, retrieval_context: List[str], expected_output: str
    ) -> dict:
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
            expected_output=expected_output
        )
        self.metric.measure(test_case)
        print(self.metric.score_breakdown)

        return {
            "answer_relevancy": self.metric.score_breakdown['Answer Relevancy (ragas)'],
            "context_precision": self.metric.score_breakdown['Contextual Precision (ragas)'],
            "context_recall": self.metric.score_breakdown['Contextual Recall (ragas)'],
            "faithfulness": self.metric.score_breakdown['Faithfulness (ragas)']
        }


__all__ = ["DeepEvalRagasEvalTool", "DeepEvalRagasEvalToolConfig"]
