from abc import abstractmethod, ABC
from typing import List

from leaf_playground.data.profile import Profile
from leaf_playground.core.scene_agent import SceneAIAgentConfig, SceneAIAgent

from ..scene_definition import *


ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("player")


class BaseAIPlayerConfig(SceneAIAgentConfig):
    pass


class BaseAIPlayer(
    SceneAIAgent,
    ABC,
    role_definition=ROLE_DEFINITION,
    cls_description="An AI agent who participants in the game Who is the Spy as a player"
):
    config_cls = BaseAIPlayerConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

    @abstractmethod
    async def receive_key(self, key_assignment: ModeratorKeyAssignment) -> None:
        pass

    @abstractmethod
    async def describe_key(self, history: List[MessageTypes], receivers: List[Profile]) -> PlayerDescription:
        pass

    @abstractmethod
    async def predict_role(self, history: List[MessageTypes], moderator: Profile) -> PlayerPrediction:
        pass

    @abstractmethod
    async def vote(self, history: List[MessageTypes], moderator: Profile, player_names: List[str]) -> PlayerVote:
        pass

    @abstractmethod
    async def reset_inner_status(self):
        pass


__all__ = [
    "BaseAIPlayerConfig",
    "BaseAIPlayer"
]
