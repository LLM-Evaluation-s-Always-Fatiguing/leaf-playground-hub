from typing import Any, Dict, List, Optional, Union
from pydantic import Field
from leaf_playground.core.workers import MetricEvaluatorConfig, MetricEvaluator
from leaf_playground.core.workers.evaluator import _MetricName, CompareOutput, RecordOutput
from leaf_playground.data.media import Json, Text
from leaf_playground.data.message import Message

from ..deep_eval.toxicity import DeepEvalToxicityEvalToolConfig, DeepEvalToxicityEvalTool

from ..scene_definition import ExamineeAnswer, SCENE_DEFINITION

ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examinee")
SUPPORT_METRIC_NAMES = ["toxicity"]
SUPPORT_METRICS = [
    m for m in ROLE_DEFINITION.get_action_definition("answer_question").metrics
    if m.name in SUPPORT_METRIC_NAMES
] if ROLE_DEFINITION.get_action_definition("answer_question").metrics is not None else []


class DeepEvalToxicityEvaluatorConfig(MetricEvaluatorConfig):
    deepEvalToxicityEvalToolConfig: DeepEvalToxicityEvalToolConfig = Field(default=...)


class DeepEvalToxicityEvaluator(
    MetricEvaluator,
    metric_definitions=SUPPORT_METRICS,
    cls_description="Toxicity metric powered by DeepEval.",
):
    config_cls = DeepEvalToxicityEvaluatorConfig
    config: config_cls

    @staticmethod
    def _init_evaluator(
        config: config_cls,
        record_metrics: List[_MetricName],
        compare_metrics: List[_MetricName]
    ) -> Any:
        if isinstance(config.deepEvalToxicityEvalToolConfig, DeepEvalToxicityEvalToolConfig):
            tool: DeepEvalToxicityEvalTool = DeepEvalToxicityEvalTool.from_config(
                config.deepEvalToxicityEvalToolConfig
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
            input_text = references[0].content.text
            actual_output = response.content.data['answer']

            misc = {
                "input": input_text,
                "actual_output": actual_output,
            }
            try:
                if isinstance(evaluator, DeepEvalToxicityEvalTool):
                    output = await evaluator.evaluate(input_text, actual_output)
            except Exception as e:
                print(f"Deep Eval toxicity evaluate error: {e}")
                output = {}

            result[f"examinee.answer_question.toxicity"] = RecordOutput(
                record_value=round(output["score"], 4),
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
    "DeepEvalToxicityEvaluatorConfig",
    "DeepEvalToxicityEvaluator"
]
