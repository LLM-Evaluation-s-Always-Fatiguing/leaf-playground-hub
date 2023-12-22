from typing import List

import openai
from leaf_playground.ai_backend.openai import OpenAIBackendConfig, CHAT_MODELS
from leaf_playground.data.media import Text
from leaf_playground.data.profile import Profile
from leaf_playground.utils.import_util import DynamicObject
from pydantic import Field

from .player import BaseAIPlayer, BaseAIPlayerConfig
from ..scene_definition import *


class OpenAIBasicPlayerConfig(BaseAIPlayerConfig):
    ai_backend_config: OpenAIBackendConfig = Field(default=...)
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

    def _prepare_completion_prompt(self, history: List[MessageTypes]) -> str:
        prompt = f"Your name is {self.name}, a player who is playing the Who is the Spy game.\n\n"
        for msg in history:
            content = msg.content.text
            if isinstance(msg, ModeratorKeyAssignment):
                content = content.replace(KEY_PLACEHOLDER, self.key_transcript)
            prompt += f"{msg.sender_name}: {content}\n\n"

        return prompt

    async def _respond(self, history: List[MessageTypes]) -> str:
        if self.language_model in CHAT_MODELS:
            resp = await self.client.chat.completions.create(
                messages=self._prepare_chat_message(history),
                model=self.language_model,
                max_tokens=64,
                temperature=0.9
            )
            response = resp.choices[0].message.content
        else:
            resp = await self.client.completions.create(
                prompt=self._prepare_completion_prompt(history),
                model=self.language_model,
                max_tokens=64,
                temperature=0.9
            )
            response = resp.choices[0].text
        return response

    async def receive_key(self, key_assignment: ModeratorKeyAssignment) -> None:
        if not key_assignment.key:
            return
        if self.env_var["key_modality"].current_value == KeyModalities.TEXT:
            self.key_transcript = key_assignment.key.text
            return
        # TODO: other modalities

    async def describe_key(self, history: List[MessageTypes], receivers: List[Profile]) -> PlayerDescription:
        description = await self._respond(history)
        return PlayerDescription(
            sender=self.profile,
            receivers=receivers,
            content=Text(text=description)
        )

    async def predict_role(self, history: List[MessageTypes], moderator: Profile) -> PlayerPrediction:
        prediction = await self._respond(history)
        return PlayerPrediction(
            sender=self.profile,
            receivers=[moderator],
            content=Text(text=prediction)
        )

    async def vote(self, history: List[MessageTypes], moderator: Profile) -> PlayerVote:
        vote = await self._respond(history)
        return PlayerVote(
            sender=self.profile,
            receivers=[moderator],
            content=Text(text=vote)
        )

    def reset_inner_status(self):
        self.key_transcript = ""


__all__ = [
    "OpenAIBasicPlayerConfig",
    "OpenAIBasicPlayer"
]
