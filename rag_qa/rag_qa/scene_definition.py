from typing import List, Literal, Union, get_args
from typing_extensions import Annotated

from pydantic import Field

from leaf_playground.core.scene_definition import *
from leaf_playground.core.scene_definition.definitions.metric import _RecordData, AggregationMethodOutput
from leaf_playground.data.message import TextMessage, JsonMessage
from leaf_playground.data.profile import Profile


def avg_fn(records: List[_RecordData]) -> AggregationMethodOutput:
    avg = round(sum(record.value for record in records) / len(records), 4)
    return AggregationMethodOutput(value=avg)


class ExaminerQuestion(TextMessage):
    question_id: int = Field(default=...)
    msg_type: Literal["question"] = Field(default="question")


class ExamineeAnswer(JsonMessage):
    question_id: int = Field(default=...)
    msg_type: Literal["answer"] = Field(default="answer")


MessageType = Annotated[Union[ExaminerQuestion, ExamineeAnswer], Field(discriminator="msg_type")]

MetricType = Literal[
    "answer_correctness",
    "answer_relevancy",
    "answer_similarity",
    "context_precision",
    "context_recall",
    "context_relevancy",
    "faithfulness"
]

MetricDefinitionList = [MetricDefinition(
    name=t,
    description=f"{t} of examinee's answer and contexts",
    record_value_dtype=ValueDType.FLOAT,
    record_display_type=DisplayType.NUMBER_INPUT,
    expect_resp_msg_type=ExamineeAnswer,
    agg_method=DynamicAggregationFn.create_dynamic_fn(fn=avg_fn),
    is_comparison=False
) for t in get_args(MetricType)]

SCENE_DEFINITION = SceneDefinition(
    name="RAG QA Examine",
    description="Retrieval Augmented Generation Question Answering Examine Scene. The evaluator powered by ragas.",
    roles=[
        RoleDefinition(
            name="examiner",
            description="the one that participants in a rag based qa examine to monitor the examinees",
            num_agents_range=(1, 1),
            is_static=True,
            actions=[]
        ),
        RoleDefinition(
            name="examinee",
            description="the one that participants in a rag based qa examine to answer questions",
            num_agents_range=(1, -1),
            is_static=False,
            actions=[
                ActionDefinition(
                    name="answer_question",
                    description="answering the question sent by examiner",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="question",
                                annotation=ExaminerQuestion
                            ),
                            ActionSignatureParameterDefinition(
                                name="examiner",
                                annotation=Profile
                            )
                        ],
                        return_annotation=ExamineeAnswer,
                        is_static_method=False
                    ),
                    metrics=MetricDefinitionList,
                )
            ]
        )
    ],
    env_vars=[]
)

__all__ = [
    "ExaminerQuestion",
    "ExamineeAnswer",
    "MessageType",
    "SCENE_DEFINITION"
]
