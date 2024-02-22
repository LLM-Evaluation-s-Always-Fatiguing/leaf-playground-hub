from typing import Any, Dict, List, Optional

from leaf_playground.core.workers import MetricEvaluatorConfig, MetricEvaluator
from leaf_playground.core.workers.evaluator import _MetricName, CompareOutput, RecordOutput
from leaf_playground.data.media import Text
from leaf_playground.data.message import Message

from ..scene_definition import ExamineeAnswer, SCENE_DEFINITION


ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examinee")


class SimpleEvaluatorConfig(MetricEvaluatorConfig):
    pass


class SimpleEvaluator(
    MetricEvaluator,
    metric_definitions=ROLE_DEFINITION.get_action_definition("answer").metrics,
    cls_description="A simple evaluator (determines whether the answer starts with the "
                    "reference answer, ignoring case sensitivity).",
):
    config_cls = SimpleEvaluatorConfig
    config: config_cls

    @staticmethod
    def _init_eval_tools(
        config: MetricEvaluatorConfig,
        record_metrics: List[_MetricName],
        compare_metrics: List[_MetricName]
    ) -> Any:
        return

    @staticmethod
    async def _record(
        response: Message,
        references: Optional[List[Message]],
        ground_truth: Optional[Text],
        eval_tools: List[Any],
        **kwargs,
    ) -> Dict[_MetricName, RecordOutput]:
        result = {}
        if isinstance(response, ExamineeAnswer) and ground_truth:
            answer = response.content.text
            ground_truth = ground_truth.text
            result["examinee.answer.accurate"] = RecordOutput(
                record_value=answer.lower().startswith(ground_truth.lower()),
                misc={
                    "question": references[0].content.text,
                    "agent_answer": answer,
                    "ground_truth": ground_truth
                }
            )
        return result

    @staticmethod
    async def _compare(
        response: Message,
        references: Optional[List[Message]],
        ground_truth: Optional[Text],
        eval_tools: List[Any],
        **kwargs,
    ) -> Dict[_MetricName, CompareOutput]:
        return {}


__all__ = [
    "SimpleEvaluatorConfig",
    "SimpleEvaluator"
]
