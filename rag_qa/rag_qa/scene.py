import asyncio
from typing import List, Optional

from pydantic import Field

from leaf_playground.core.workers import Logger
from leaf_playground.core.scene import Scene
from leaf_playground.core.scene_definition import SceneConfig
from leaf_playground.data.log_body import ActionLogBody
from leaf_playground.data.media import Text, Json

from .agents.examiner import Examiner
from .agents.base_examinee import AIBaseExaminee
from .dataset_utils import DatasetConfig
from .scene_definition import ExamineeAnswer, ExaminerQuestion, MessageType, SCENE_DEFINITION


class RagSceneLogBody(ActionLogBody):
    ground_truth: Optional[Json] = Field(default=None)


RagSceneConfig = SceneConfig.create_config_model(
    SCENE_DEFINITION,
    additional_config_fields={
        "dataset_config": (DatasetConfig, Field(default=...)),
        "debug_mode": (bool, Field(default=False, exclude=True))
    }
)


class RagScene(Scene, scene_definition=SCENE_DEFINITION, log_body_class=RagSceneLogBody):
    config_cls = RagSceneConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self.examiner: Examiner = self.static_agents["examiner"][0]
        self.examinees: List[AIBaseExaminee] = self.agents["examinee"]

    async def _run(self):
        async def examinee_answer(examinee: AIBaseExaminee, q: ExaminerQuestion) -> None:
            try:
                answer: ExamineeAnswer = await examinee.answer_question(question=q, examiner=self.examiner.profile)
            except:
                if self.config.debug_mode:
                    raise
                answer: ExamineeAnswer = ExamineeAnswer(
                    sender=examinee.profile,
                    receivers=[self.examiner.profile],
                    content=Json(data={"answer": "", "contexts": []}),
                    question_id=q.question_id
                )
            self.message_pool.put_message(answer)
            ground_truth = self.examiner.get_golden_answer(q.question_id)
            log = self.log_body_class(
                references=[q.id],
                response=answer.id,
                ground_truth=Json(data=ground_truth) if ground_truth else None,
                log_msg=f"examinee [{examinee.name}] answered question [{q.question_id}]",
                action_belonged_chain=examinee.role_definition.get_action_definition("answer_question").belonged_chain
            )
            self.logger.add_log(log)
            self.notify_evaluators_record(log)

        self.examiner.prepare_questions(self.config.dataset_config)
        while not self.examiner.check_examine_finish():
            question: ExaminerQuestion = self.examiner.send_question(
                receivers=[examinee.profile for examinee in self.examinees]
            )
            self.message_pool.put_message(question)
            self.logger.add_log(
                self.log_body_class(
                    references=None,
                    response=question.id,
                    ground_truth=None,
                    log_msg=f"examiner sent question [{question.question_id}] to all examinees",
                    action_belonged_chain=None
                )
            )

            await asyncio.gather(
                *[examinee_answer(examinee, question) for examinee in self.examinees]
            )


__all__ = [
    "RagSceneConfig",
    "RagScene"
]
