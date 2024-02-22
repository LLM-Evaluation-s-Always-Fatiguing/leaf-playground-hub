from typing import Any, Dict, Literal, List, Optional, Union

from datasets import Dataset, Features, Value, Sequence
from ragas import evaluate

from leaf_eval_tools.ragas_eval_worker import RagasEvalWorker, RagasEvalWorkerConfig
from leaf_playground.core.workers import MetricEvaluatorConfig, MetricEvaluator
from leaf_playground.core.workers.evaluator import _MetricName, CompareOutput, RecordOutput
from leaf_playground.data.media import Json, Text
from leaf_playground.data.message import Message

from ..scene_definition import ExamineeAnswer, SCENE_DEFINITION

from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    context_relevancy,
    faithfulness
)

ragas_metrics_map = {
    "answer_correctness": answer_correctness,
    "answer_relevancy": answer_relevancy,
    "answer_similarity": answer_similarity,
    "context_precision": context_precision,
    "context_recall": context_recall,
    "context_relevancy": context_relevancy,
    "faithfulness": faithfulness
}

ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examinee")


class RagasEvaluatorConfig(MetricEvaluatorConfig):
    pass


class RagasEvaluator(
    MetricEvaluator,
    metric_definitions=ROLE_DEFINITION.get_action_definition("answer_question").metrics,
    cls_description="Retrieval augmented generation question answering examine scene evaluator powered by ragas.",
):
    config_cls = RagasEvaluatorConfig
    config: config_cls

    @staticmethod
    def _init_eval_tools(
        config: MetricEvaluatorConfig,
        record_metrics: List[_MetricName],
        compare_metrics: List[_MetricName]
    ) -> List[RagasEvalWorker]:
        eval_tool = RagasEvalWorker(
            config=RagasEvalWorkerConfig(),
            activated_metrics=[ragas_metrics_map[metric_name.split('.')[-1]] for metric_name in record_metrics]
        )
        return [eval_tool]

    @staticmethod
    async def _record(
        response: Message,
        references: Optional[List[Message]],
        ground_truth: Optional[Union[Json, Text]],
        eval_tools: List[RagasEvalWorker],
        **kwargs
    ) -> Dict[_MetricName, RecordOutput]:
        result = {}
        if isinstance(response, ExamineeAnswer) and ground_truth:

            if isinstance(ground_truth, Json):
                data: dict = ground_truth.data
            else:
                return result

            question = references[0].content.text
            ground_truths = data.get('ground_truths', None)
            golden_answer = data.get('golden_answer', None)
            agent_answer = response.content.data['answer']
            contexts = response.content.data['contexts']

            misc = {
                'question': question,
                'answer': agent_answer,
                'contexts': contexts,
                'ground_truths': ground_truths,
                'golden_answer': golden_answer
                # Actually, it’s not used here. The original answer from ragas is the evaluated answer,
                # and it is placed here for future reference when displaying in the log.
            }

            features = Features({
                'question': Value('string'),
                'answer': Value('string'),
                'contexts': Sequence(Value('string')),
                'ground_truths': Sequence(Value('string')),
                'golden_answer': Value('string')
            })

            def gen():
                yield misc

            dataset = Dataset.from_generator(gen, features=features)

            try:
                output = eval_tools[0](dataset)
            except Exception as e:
                print(f"Ragas evaluate error: {e}")
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
        eval_tools: List[RagasEvalWorker],
        **kwargs
    ) -> Dict[_MetricName, CompareOutput]:
        return {}


__all__ = [
    "RagasEvaluatorConfig",
    "RagasEvaluator"
]
