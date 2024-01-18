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
]


class BackendConfig(OpenAIBackendConfig):
    model: ChatModels = Field(default=...)


class OpenAIAdvancePlayerConfig(BaseAIPlayerConfig):
    ai_backend_config: BackendConfig = Field(default=...)
    ai_backend_obj: DynamicObject = Field(
        default=DynamicObject(obj="OpenAIBackend", module="leaf_playground.ai_backend.openai"),
        exclude=True
    )


class OpenAIAdvancePlayer(
    BaseAIPlayer,
    cls_description="An AI agent using OpenAI as backend, participants in the game Who is the Spy as a player"
):
    config_cls = OpenAIAdvancePlayerConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self.language_model = self.config.ai_backend_config.model
        self.vision_model = "gpt-4-vision-preview"
        self.audio_model = "whisper-1"

        self.client: openai.AsyncOpenAI = self.backend.async_client

        self.key_transcript = ""

    def _prepare_chat_message(self, history: List[MessageTypes], mode: Literal['description', 'prediction', 'vote']) -> \
        List[dict]:
        messages = [
            {
                "role": "system",
                "content": (
                    f"你是一名特别有经验的谁是卧底玩家，你的名字是 {self.name}，非常擅长分析推理和伪装，你现在正在参加一场高手云集的谁是卧底比赛。\n"
                    f"{history[0].content.text}"
                )
            },
        ]

        history_str = "以下是游戏进行的历史记录：\n"
        for msg in history[1:]:
            content = msg.content.text
            if isinstance(msg, ModeratorKeyAssignment):
                content = content.replace(KEY_PLACEHOLDER, self.key_transcript)
            history_str += f"{msg.sender_name}: {content}\n"

        description_following_str = (
            f"现在轮到你描述关键词了，你的关键词是{self.key_transcript}，你也可以根据之前的身份预测描述你认为正确的关键词。\n"
            "根据之前其他人的发言，你尝试使用模糊的方式描述，让别的玩家无法识别出你的身份。\n"
            "接下来请你严格按照主持人的要求直接回复。\n"
        )
        prediction_following_str = (
            f"现在轮到你预测身份了，你的关键词是{self.key_transcript}。\n"
            "根据之前其他人的发言，你简要地一步一步地分析每个玩家的身份和所持有的关键词。\n"
            f"如果超过{1 if len(history[0].receivers) < 7 else 2}个玩家的描述与你的关键词有矛盾，你自己很有可能就是卧底，请根据你的关键词和其他玩家的描述推测正确的关键词。\n"
            "接下来请你严格按照主持人的要求直接回复，且不超过200个字或字符。\n"
        )
        vote_following_str = f"现在轮到你投票了，接下来请你严格按照主持人的要求直接投票。"

        if mode == 'description':
            messages.append(
                {
                    "role": "user",
                    "content": f"{history_str}\n{description_following_str}",
                }
            )
        elif mode == 'prediction':
            messages.append(
                {
                    "role": "user",
                    "content": f"{history_str}\n{prediction_following_str}",
                }
            )
        elif mode == 'vote':
            messages.append(
                {
                    "role": "user",
                    "content": f"{history_str}\n{vote_following_str}",
                }
            )

        return messages

    async def _respond(self, history: List[MessageTypes], mode: Literal['description', 'prediction', 'vote']) -> str:
        resp = await self.client.chat.completions.create(
            messages=self._prepare_chat_message(history, mode),
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
        # TODO: image mode change description to image & support audio modal

    async def describe_key(self, history: List[MessageTypes], receivers: List[Profile]) -> PlayerDescription:
        try:
            description = await self._respond(history, 'description')
        except Exception as e:
            print(e)
            description = "我不知道该说什么了，你们自己看着办吧。"
        return PlayerDescription(
            sender=self.profile,
            receivers=receivers,
            content=Text(text=description, display_text=description)
        )

    async def predict_role(self, history: List[MessageTypes], moderator: Profile) -> PlayerPrediction:
        prediction = await self._respond(history, 'prediction')
        return PlayerPrediction(
            sender=self.profile,
            receivers=[moderator, self.profile],
            content=Text(text=prediction, display_text=prediction)
        )

    async def vote(self, history: List[MessageTypes], moderator: Profile) -> PlayerVote:
        vote = await self._respond(history, 'vote')
        return PlayerVote(
            sender=self.profile,
            receivers=[moderator, self.profile],
            content=Text(text=vote, display_text=vote)
        )

    async def reset_inner_status(self):
        self.key_transcript = ""


__all__ = [
    "OpenAIAdvancePlayerConfig",
    "OpenAIAdvancePlayer"
]
