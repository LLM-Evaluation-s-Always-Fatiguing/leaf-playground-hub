from enum import Enum

from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import Field

from leaf_playground.ai_backend.openai import OpenAIBackendConfig
from leaf_playground.data.media import Text
from leaf_playground.data.profile import Profile
from leaf_playground.utils.import_util import DynamicObject

from .base_examinee import (
    AIBaseExaminee,
    AIBaseExamineeConfig
)
from ..scene_definition import ExamineeAnswer, ExaminerSample

MODELS = Enum(
    "Models",
    {
        m: m for m in [
            "gpt-3.5-turbo-1106",
            "gpt-4-1106-preview",
            "gpt-3.5-turbo",
            "gpt-4-vision-preview",
            "gpt-4",
            "gpt-4-32k",
        ]
    }
)


class BackendConfig(OpenAIBackendConfig):
    model: MODELS = Field(default=...)


class OpenAIBasicExamineeConfig(AIBaseExamineeConfig):
    ai_backend_config: BackendConfig = Field(default=...)
    ai_backend_obj: DynamicObject = Field(
        default=DynamicObject(obj="OpenAIBackend", module="leaf_playground.ai_backend.openai"),
        exclude=True
    )


class OpenAIBasicExaminee(AIBaseExaminee, cls_description="Examinee agent using OpenAI API to answer questions"):
    config_cls = OpenAIBasicExamineeConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

    async def answer(self, sample: ExaminerSample, examiner: Profile) -> ExamineeAnswer:
        client: AsyncOpenAI = self.backend.async_client
        model = self.config.ai_backend_config.model.value

        system_msg = (
            f"Your name is {self.name}, an {self.profile.role.name}, {self.profile.role.description}. "
            f"You will receive a question and a list of choices from the examiner(user), the task you "
            f"need to do is to choose the correct answer from the given choices. Please only answer with "
            f"the index of the correct answer, do not explain your choice."
        )
        examiner_msg = sample.content.text
        try:
            resp = await client.chat.completions.create(
                messages=[
                    ChatCompletionSystemMessageParam(role="system", content=system_msg),
                    ChatCompletionUserMessageParam(role="user", content=examiner_msg),
                ],
                model=model,
                max_tokens=2
            )
        except:
            resp = None
        return ExamineeAnswer(
            sample_id=sample.sample_id,
            content=Text(text=resp.choices[0].message.content if resp else ""),
            sender=self.profile,
            receivers=[examiner]
        )


__all__ = [
    "OpenAIBasicExamineeConfig",
    "OpenAIBasicExaminee"
]
