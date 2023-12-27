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
                "content": f"Your name is {self.name}, a player who is playing the Who is the Spy game."
            },
            {"role": "system", "content": history[0].content.text}
        ]
        for msg in history[1:]:
            content = msg.content.text
            if isinstance(msg, ModeratorKeyAssignment):
                content = content.replace(KEY_PLACEHOLDER, self.key_transcript)
            messages.append(
                {
                    "role": "user" if msg.sender_role != "moderator" else "system",
                    "content": content,
                    "name": msg.sender_name
                }
            )
        return messages

    async def _respond(self, history: List[MessageTypes]) -> str:
        resp = await self.client.chat.completions.create(
            messages=self._prepare_chat_message(history),
            model=self.language_model,
            max_tokens=64,
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
                            {"type": "text", "text": "Describe this image using **AT MOST** 16 words."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=24,
            )
            self.key_transcript = response.choices[0].message.content
        # TODO: audio modal

    async def describe_key(self, history: List[MessageTypes], receivers: List[Profile]) -> PlayerDescription:
        description = await self._respond(history)
        return PlayerDescription(
            sender=self.profile,
            receivers=receivers,
            content=Text(text=description, display_text=description)
        )

    async def predict_role(self, history: List[MessageTypes], moderator: Profile) -> PlayerPrediction:
        prediction = await self._respond(history)
        return PlayerPrediction(
            sender=self.profile,
            receivers=[moderator],
            content=Text(text=prediction, display_text=prediction)
        )

    async def vote(self, history: List[MessageTypes], moderator: Profile) -> PlayerVote:
        vote = await self._respond(history)
        return PlayerVote(
            sender=self.profile,
            receivers=[moderator],
            content=Text(text=vote, display_text=vote)
        )

    def reset_inner_status(self):
        self.key_transcript = ""


__all__ = [
    "OpenAIBasicPlayerConfig",
    "OpenAIBasicPlayer"
]
