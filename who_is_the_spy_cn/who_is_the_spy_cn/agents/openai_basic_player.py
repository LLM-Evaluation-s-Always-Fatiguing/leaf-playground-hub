import base64
from typing import List, Literal

import openai
from leaf_playground.ai_backend.openai import OpenAIBackendConfig
from leaf_playground.data.media import Text
from leaf_playground.data.profile import Profile
from leaf_playground.utils.import_util import DynamicObject
from pydantic import Field

from .player import BaseAIPlayer, BaseAIPlayerConfig
from ..scene_definition import *


def encode_local_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


ChatModels = Literal[
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
]


class BackendConfig(OpenAIBackendConfig):
    model: ChatModels = Field(default=...)


class OpenAIBasicPlayerConfig(BaseAIPlayerConfig):
    ai_backend_config: BackendConfig = Field(default=...)
    ai_backend_obj: DynamicObject = Field(
        default=DynamicObject(obj="OpenAIBackend", module="leaf_playground.ai_backend.openai"),
        exclude=True
    )


class OpenAIBasicPlayer(
    BaseAIPlayer,
    cls_description="An AI agent using OpenAI as backend, participants in the game Who is the Spy as a player"
):
    config_cls = OpenAIBasicPlayerConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self.language_model = self.config.ai_backend_config.model
        self.vision_model = "gpt-4-vision-preview"
        self.audio_model = "whisper-1"

        self.client: openai.AsyncOpenAI = self.backend.async_client

        self.key_transcript = ""

    def _prepare_chat_message(self, history: List[MessageTypes]) -> List[dict]:
        messages = [
            {
                "role": "system",
                "content": (
                    f"你是一名游戏高手，非常擅长分析推理和伪装。你正在参与一场游戏：谁是卧底。游戏的赢家可以瓜分1000美元奖金。"
                    f"你的名字是 {self.name}，在这场游戏中会有一名主持人和其他玩家。"
                    f"必须记住，你是 {self.name}， 不是主持人或者其他玩家。"
                )
            },
            {"role": "system", "content": history[0].content.text}
        ]
        for msg in history[1:]:
            content = msg.content.text
            if isinstance(msg, ModeratorKeyAssignment):
                content = content.replace(KEY_PLACEHOLDER, self.key_transcript)
            messages.append(
                {
                    "role": "user",
                    "content": content,
                    "name": msg.sender_name
                }
            )
        return messages

    async def _respond(self, history: List[MessageTypes]) -> str:
        resp = await self.client.chat.completions.create(
            messages=self._prepare_chat_message(history),
            model=self.language_model,
            max_tokens=256,
            temperature=0.9
        )
        response = resp.choices[0].message.content
        return response

    async def receive_key(self, key_assignment: ModeratorKeyAssignment) -> None:
        key_modality = self.env_var["key_modality"].current_value
        if not key_assignment.key:
            return
        if key_modality == KeyModalities.TEXT:
            self.key_transcript = key_assignment.key.text
        elif key_modality == KeyModalities.IMAGE:
            image_data = encode_local_image(key_assignment.key.url)
            response = await self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请最多用20个字生成一段对图片的描述"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=256,
            )
            self.key_transcript = response.choices[0].message.content
            print(self.key_transcript)
        # TODO: audio modal

    async def describe_key(self, history: List[MessageTypes], receivers: List[Profile]) -> PlayerDescription:
        try:
            description = await self._respond(history)
        except Exception as e:
            print(e)
            description = "我不知道该说什么了，你们自己看着办吧。"
        return PlayerDescription(
            sender=self.profile,
            receivers=receivers,
            content=Text(text=description, display_text=description)
        )

    async def predict_role(self, history: List[MessageTypes], moderator: Profile) -> PlayerPrediction:
        prediction = await self._respond(history)
        return PlayerPrediction(
            sender=self.profile,
            receivers=[moderator, self.profile],
            content=Text(text=prediction, display_text=prediction)
        )

    async def vote(self, history: List[MessageTypes], moderator: Profile) -> PlayerVote:
        vote = await self._respond(history)
        return PlayerVote(
            sender=self.profile,
            receivers=[moderator, self.profile],
            content=Text(text=vote, display_text=vote)
        )

    async def reset_inner_status(self):
        self.key_transcript = ""


__all__ = [
    "OpenAIBasicPlayerConfig",
    "OpenAIBasicPlayer"
]
