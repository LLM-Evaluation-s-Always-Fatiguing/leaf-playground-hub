from typing import List

from leaf_playground.core.scene_agent import SceneHumanAgent, SceneHumanAgentConfig
from leaf_playground.data.profile import Profile
from leaf_playground.data.media import Text

from .player import ROLE_DEFINITION
from ..scene_definition import *


class HumanPlayerConfig(SceneHumanAgentConfig):
    pass


class HumanPlayer(
    SceneHumanAgent,
    role_definition=ROLE_DEFINITION,
    cls_description="参与谁是卧底游戏的人类玩家的代理"
):
    config_cls = HumanPlayerConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

    async def receive_key(self, key_assignment: ModeratorKeyAssignment) -> None:
        pass

    async def describe_key(self, history: List[MessageTypes], receivers: List[Profile]) -> PlayerDescription:
        description = (await self.wait_human_text_input()) or ""
        return PlayerDescription(
            sender=self.profile,
            receivers=receivers,
            content=Text(text=description, display_text=description)
        )

    async def predict_role(self, history: List[MessageTypes], moderator: Profile) -> PlayerPrediction:
        prediction = (await self.wait_human_text_input()) or ""
        return PlayerPrediction(
            sender=self.profile,
            receivers=[moderator, self.profile],
            content=Text(text=prediction, display_text=prediction)
        )

    async def vote(self, history: List[MessageTypes], moderator: Profile, player_names: List[str]) -> PlayerVote:
        vote = (await self.wait_human_text_input(PlayerVote.generate_data_schema(player_names))) or ""
        return PlayerVote(
            sender=self.profile,
            receivers=[moderator, self.profile],
            content=Text(text=vote, display_text=vote)
        )

    async def reset_inner_status(self):
        pass


__all__ = [
    "HumanPlayerConfig",
    "HumanPlayer"
]
