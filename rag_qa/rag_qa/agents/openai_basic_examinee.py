import json
from typing import Literal, Type, Union

from openai.types.chat.chat_completion_message_param import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import Field

from leaf_ai_backends.openai import OpenAIBackendConfig, OpenAIBackend, OpenAIClientConfig, AzureOpenAIClientConfig
from leaf_playground.data.media import Json
from leaf_playground.data.profile import Profile

from .base_examinee import (
    AIBaseExaminee,
    AIBaseExamineeConfig
)
from ..scene_definition import ExamineeAnswer, ExaminerQuestion


class CustomOpenAIClientConfig(OpenAIClientConfig):
    chat_model: Literal[
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    ] = Field(default=...)


class CustomAzureOpenAIClientConfig(AzureOpenAIClientConfig):
    chat_model: str = Field(default=...)


class CustomOpenAIBackendConfig(OpenAIBackendConfig):
    client_config: Union[
        CustomOpenAIClientConfig,
        CustomAzureOpenAIClientConfig,
    ] = Field(default=..., union_mode="smart")


class OpenAIBasicExamineeConfig(AIBaseExamineeConfig):
    ai_backend_config: CustomOpenAIBackendConfig = Field(default=...)
    ai_backend_cls: Type[OpenAIBackend] = Field(default=OpenAIBackend)


class OpenAIBasicExaminee(AIBaseExaminee, cls_description="Examinee agent using OpenAI API to answer questions"):
    config_cls = OpenAIBasicExamineeConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

    async def answer_question(self, question: ExaminerQuestion, examiner: Profile) -> ExamineeAnswer:
        model = self.config.ai_backend_config.chat_model

        system_msg = (
            f"You are a meticulous scholar who, when faced with users' questions, not only accurately answers their "
            f"queries but also provides the references used. You always reply to all questions from users in the "
            f"following JSON format:\n\n"
            f"{{\n"
            f"    \"answer\": \"The answer to the original question\",\n"
            f"    \"contexts\": [\"Relevant citations\", \"Typically three to five\", \"Mainly based on objective facts and data\",...]\n"
            f"}}\n\n"
        )
        examiner_msg = question.content.text

        resp_format = {"type": "json_object"} if model.find("gpt-4") >= 0 else {"type": "text"}
        try:
            resp = await self.backend.async_client.chat.completions.create(
                messages=[
                    ChatCompletionSystemMessageParam(role="system", content=system_msg),
                    ChatCompletionUserMessageParam(role="user", content=examiner_msg),
                ],
                model=model,
                response_format=resp_format,
                max_tokens=2048
            )
        except Exception as e:
            print(e)
            resp = None

        try:
            obj = json.loads(resp.choices[0].message.content) if resp else {}
        except Exception as e:
            print(f'Response Not JSON: {e}')
            obj = {
                "answer": resp.choices[0].message.content,
                "contexts": ['nothing found']  # default for ragas data type validation
            }

        contexts = obj['contexts'] if resp else ['nothing found']
        answer = obj['answer'] if resp else ""

        json_data = {"answer": answer, "contexts": contexts}

        return ExamineeAnswer(
            question_id=question.question_id,
            content=Json(data=json_data, display_text=answer),
            sender=self.profile,
            receivers=[examiner]
        )


__all__ = [
    "OpenAIBasicExamineeConfig",
    "OpenAIBasicExaminee"
]
