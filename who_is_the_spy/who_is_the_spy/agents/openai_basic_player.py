import base64
from typing import List, Literal, Union, Type

from leaf_ai_backends.openai import OpenAIBackendConfig, OpenAIBackend, OpenAIClientConfig, AzureOpenAIClientConfig
from leaf_playground.data.media import Text
from leaf_playground.data.profile import Profile
from pydantic import Field

from .player import BaseAIPlayer, BaseAIPlayerConfig
from ..scene_definition import *


def encode_local_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class CustomOpenAIClientConfig(OpenAIClientConfig):
    chat_model: Literal[
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    ] = Field(default=...)
    vision_model: Literal["gpt-4-1106-vision-preview"] = Field(default="gpt-4-1106-vision-preview")


class CustomOpenAIBackendConfig(OpenAIBackendConfig):
    client_config: Union[CustomOpenAIClientConfig, AzureOpenAIClientConfig] = Field(default=..., union_mode="smart")


class OpenAIBasicPlayerConfig(BaseAIPlayerConfig):
    ai_backend_config: CustomOpenAIBackendConfig = Field(default=...)
    ai_backend_cls: Type[OpenAIBackend] = Field(default=OpenAIBackend, exclude=True)


class OpenAIBasicPlayer(
    BaseAIPlayer,
    cls_description="An AI agent using OpenAI as backend, participants in the game Who is the Spy as a player"
):
    config_cls = OpenAIBasicPlayerConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self.client = self.backend.async_client
        self.key_transcript = ""

    def _prepare_chat_message(self, history: List[MessageTypes]) -> List[dict]:
        messages = [
            {
                "role": "system",
                "content": (
                    f"I want you to be a player named {self.name}, who is playing the Who is the Spy game, "
                    f"where there will be one moderator and some other players. You need to ALWAYS REMEMBER "
                    f"that you are a player named {self.name}, not the moderator and other players.\n"
                    f"In the game, you will be asked to do three things by the moderator:\n"
                    f"- describe key: in your description, do not include any of the original content "
                    f"from your key, but instead reveal your key to the other players through as subtle a "
                    f"description as possible.\n"
                    f"- predict spy (and blank if has): the moderator will tell you how many spies and blank "
                    f"player the game will have, when prediction, for each role, you should give the same number "
                    f"of names, and if you think you are the spy or blank, you can include your name in the final "
                    f"prediction.\n"
                    f"- vote to eliminate one player: in the voting stage, you should vote one player to be "
                    f"eliminated, DO NOT vote yourself, your goal is to survive till the last round."
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
            model=self.config.ai_backend_config.chat_model,
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
                model=self.config.ai_backend_config.vision_model,
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
