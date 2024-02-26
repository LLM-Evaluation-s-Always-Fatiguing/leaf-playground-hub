import json
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
        description = (await self.wait_human_text_input(PlayerDescription.generate_data_schema())) or ""
        try:
            description = json.loads(description)
        except:
            description = {"description": description}
        return PlayerDescription(
            sender=self.profile,
            receivers=receivers,
            content=Text(text=f"{description['description']}")
        )

    async def predict_role(
        self, history: List[MessageTypes], moderator: Profile, player_names: List[str]
    ) -> PlayerPrediction:
        prediction = (await self.wait_human_text_input(PlayerPrediction.generate_data_schema(player_names))) or ""
        try:
            prediction_data = json.loads(prediction)
        except:
            prediction_data = {"maybe_spy_list": prediction}
        return PlayerPrediction(
            sender=self.profile,
            receivers=[moderator, self.profile],
            content=Text(text=f"卧底：{','.join(prediction_data['maybe_spy_list'])}")
        )

    async def vote(self, history: List[MessageTypes], moderator: Profile, player_names: List[str]) -> PlayerVote:
        vote = (await self.wait_human_text_input(PlayerVote.generate_data_schema(player_names))) or ""
        try:
            vote_data = json.loads(vote)
        except:
            vote_data = {"vote_target": vote}

        return PlayerVote(
            sender=self.profile,
            receivers=[moderator, self.profile],
            content=Text(
                text=f"投票：{vote_data['vote_target']}<EOS>"
            )
        )

    async def reset_inner_status(self):
        pass


__all__ = [
    "HumanPlayerConfig",
    "HumanPlayer"
]
