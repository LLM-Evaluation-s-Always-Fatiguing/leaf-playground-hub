from typing import Any, Dict, List

from leaf_playground.core.workers import MetricEvaluatorConfig, MetricEvaluator
from leaf_playground.core.workers.evaluator import _MetricName, CompareOutput, RecordOutput
from leaf_playground.data.log_body import ActionLogBody
from leaf_playground.eval_tools.regex import RegexEvalTool, RegexEvalToolConfig
from pydantic import Field

from ..scene_definition import ExamineeAnswer, SCENE_DEFINITION


ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examinee")


class RegexEvaluatorConfig(MetricEvaluatorConfig):
    regexEvalToolConfig: RegexEvalToolConfig = Field(...)


class RegexEvaluator(
    MetricEvaluator,
    metric_definitions=ROLE_DEFINITION.get_action_definition(
        "answer_question").metrics,
    cls_description="Regex evaluator (uses the specified regular expression to extract the answer, then compares it to the reference answer).",
):
    config_cls = RegexEvaluatorConfig
    config: config_cls

    @staticmethod
    def _init_evaluator(
        config: MetricEvaluatorConfig,
        record_metrics: List[_MetricName],
        compare_metrics: List[_MetricName]
    ) -> Any:
        if isinstance(config, RegexEvaluatorConfig):
            regexEvalTool: RegexEvalTool = RegexEvalTool.from_config(
                config.regexEvalToolConfig)
            return regexEvalTool
        else:
            raise ValueError(f"Invalid config type {type(config)}")

    @staticmethod
    async def _record(log: ActionLogBody, evaluator: Any) -> Dict[_MetricName, RecordOutput]:
        result = {}
        if isinstance(log.response, ExamineeAnswer) and log.ground_truth:
            origin_answer = log.response.content.text
            ground_truth = log.ground_truth.text
            ignore_case = True
            misc = {
                "question": log.references[0].content.text,
                "agent_answer": origin_answer,
                "ground_truth": ground_truth
            }
            if isinstance(evaluator, RegexEvalTool):
                answer = evaluator.extract_answer(origin_answer)
                ignore_case = evaluator.ignore_case
                misc["extracted_answer"] = answer
            else:
                answer = origin_answer

            is_correct = answer.lower().startswith(
                ground_truth.lower()) if ignore_case else answer.startswith(ground_truth)

            result["examinee.answer_question.accurate"] = RecordOutput(
                record_value=is_correct,
                misc=misc
            )
        return result

    @staticmethod
    async def _compare(log: ActionLogBody, evaluator: Any) -> Dict[_MetricName, CompareOutput]:
        return {}


__all__ = [
    "RegexEvaluatorConfig",
    "RegexEvaluator"
]
