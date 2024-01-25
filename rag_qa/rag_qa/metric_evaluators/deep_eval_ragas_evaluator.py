from typing import Any, Dict, Literal, List, Optional, Union
from pydantic import Field
from leaf_playground.core.workers import MetricEvaluatorConfig, MetricEvaluator
from leaf_playground.core.workers.evaluator import _MetricName, CompareOutput, RecordOutput
from leaf_playground.data.media import Json, Text
from leaf_playground.data.message import Message

from ..deep_eval.ragas import DeepEvalRagasEvalToolConfig, DeepEvalRagasEvalTool

from ..scene_definition import ExamineeAnswer, SCENE_DEFINITION

ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examinee")
SUPPORT_METRIC_NAMES = ["answer_relevancy", "context_precision", "context_recall", "faithfulness"]
SUPPORT_METRICS = [
    m for m in ROLE_DEFINITION.get_action_definition("answer_question").metrics
    if m.name in SUPPORT_METRIC_NAMES
] if ROLE_DEFINITION.get_action_definition("answer_question").metrics is not None else []


class DeepEvalRagasEvaluatorConfig(MetricEvaluatorConfig):
    ragasEvalToolConfig: DeepEvalRagasEvalToolConfig = Field(...)


class DeepEvalRagasEvaluator(
    MetricEvaluator,
    metric_definitions=SUPPORT_METRICS,
    cls_description="Retrieval augmented generation question answering examine scene evaluator powered by DeepEval.",
):
    config_cls = DeepEvalRagasEvaluatorConfig
    config: config_cls

    @staticmethod
    def _init_evaluator(
        config: config_cls,
        record_metrics: List[_MetricName],
        compare_metrics: List[_MetricName]
    ) -> Any:
        if isinstance(config.ragasEvalToolConfig, DeepEvalRagasEvalToolConfig):
            tool: DeepEvalRagasEvalTool = DeepEvalRagasEvalTool.from_config(
                config.ragasEvalToolConfig
            )
            return tool
        else:
            raise ValueError(f"Invalid config type {type(config)}")

    @staticmethod
    async def _record(
        response: Message,
        references: Optional[List[Message]],
        ground_truth: Optional[Union[Json, Text]],
        evaluator: Any,
        **kwargs
    ) -> Dict[_MetricName, RecordOutput]:
        result = {}
        if isinstance(response, ExamineeAnswer) and ground_truth:

            if isinstance(ground_truth, Json):
                data: dict = ground_truth.data
            else:
                return result

            input_text = references[0].content.text
            actual_output = response.content.data['answer']
            retrieval_context = response.content.data['contexts']
            expected_output = data.get('golden_answer', None)

            misc = {
                "input": input_text,
                "actual_output": actual_output,
                "retrieval_context": retrieval_context,
                "expected_output": expected_output
            }
            try:
                if isinstance(evaluator, DeepEvalRagasEvalTool):
                    output = await evaluator.evaluate(input_text, actual_output, retrieval_context, expected_output)
            except Exception as e:
                print(f"Deep Eval Ragas evaluate error: {e}")
                output = {}

            for metric, value in output.items():
                result[f"examinee.answer_question.{metric}"] = RecordOutput(
                    record_value=round(value, 4),
                    misc=misc
                )

        return result

    @staticmethod
    async def _compare(
        response: Message,
        references: Optional[List[Message]],
        ground_truth: Optional[Union[Json, Text]],
        evaluator: Any,
        **kwargs
    ) -> Dict[_MetricName, CompareOutput]:
        return {}


__all__ = [
    "DeepEvalRagasEvaluatorConfig",
    "DeepEvalRagasEvaluator"
]
