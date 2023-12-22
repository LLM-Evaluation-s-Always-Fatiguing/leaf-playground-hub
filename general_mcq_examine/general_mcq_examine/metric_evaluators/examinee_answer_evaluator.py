from typing import Any, Dict, List

from leaf_playground.core.workers import MetricEvaluatorConfig, MetricEvaluator
from leaf_playground.core.workers.evaluator import _MetricName, CompareOutput, RecordOutput
from leaf_playground.data.log_body import ActionLogBody
from leaf_playground.eval_tools.regex import RegexEvalTool, RegexEvalToolConfig
from pydantic import Field

from ..scene_definition import ExamineeAnswer, SCENE_DEFINITION


ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examinee")


class ExamineeAnswerEvaluatorConfig(MetricEvaluatorConfig):
    regexEvalToolConfig: RegexEvalToolConfig = Field(...)


class ExamineeAnswerEvaluator(
    MetricEvaluator,
    metric_definitions=ROLE_DEFINITION.get_action_definition("answer_question").metrics,
    cls_description="An evaluator that evaluate examinee's answers",
):
    config_cls = ExamineeAnswerEvaluatorConfig
    config: config_cls

    @staticmethod
    def _init_evaluator(
        config: MetricEvaluatorConfig,
        record_metrics: List[_MetricName],
        compare_metrics: List[_MetricName]
    ) -> Any:
        if isinstance(config, ExamineeAnswerEvaluatorConfig):
            regexEvalTool: RegexEvalTool = RegexEvalTool.from_config(config.regexEvalToolConfig)
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
            if isinstance(evaluator, RegexEvalTool):
                answer = evaluator.extract_answer(origin_answer)
                ignore_case = evaluator.ignore_case
            else:
                answer = origin_answer
            
            is_correct = answer.lower().startswith(ground_truth.lower()) if ignore_case else answer.startswith(ground_truth)

            result["examinee.answer_question.accurate"] = RecordOutput(
                record_value=is_correct,
                misc={
                    "question": log.references[0].content.text,
                    "agent_answer": answer,
                    "ground_truth": ground_truth
                }
            )
        return result

    @staticmethod
    async def _compare(log: ActionLogBody, evaluator: Any) -> Dict[_MetricName, CompareOutput]:
        return {}


__all__ = [
    "ExamineeAnswerEvaluatorConfig",
    "ExamineeAnswerEvaluator"
]
