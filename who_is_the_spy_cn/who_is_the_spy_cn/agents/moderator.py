import random
from itertools import chain
from typing import Dict, List, Optional, Union

from leaf_playground.data.profile import Profile
from leaf_playground.data.media import Audio, Image, Text
from leaf_playground.core.scene_agent import SceneStaticAgentConfig, SceneStaticAgent

from ..data_utils import *
from ..scene_definition import *


ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("moderator")


ModeratorConfig = SceneStaticAgentConfig.create_config_model(ROLE_DEFINITION)


class Moderator(
    SceneStaticAgent,
    role_definition=ROLE_DEFINITION,
    cls_description="An agent who moderate the game: Who is the Spy"
):
    config_cls = ModeratorConfig
    config: config_cls

    game_rule_with_blank = (
        "下面是游戏规则：\n\n"
        "###基础信息###\n\n"
        "游戏中有三个角色：平民，卧底和白板。在游戏的开始，每个人都会收到一条关键词，仅自己可见\n"
        "- 平民们会收到相同的正确的关键词。\n"
        "- 卧底们会收到相同的错误的关键词，但是跟正确的关键词较为相似。\n"
        "- 白板会被告知自己是白板角色。\n"
        "你不会被告知自己的角色，只能通过观察和分析进行猜测。\n\n"
        "###游戏阶段###\n\n"
        "这场游戏有四个阶段:\n"
        "1.描述阶段：所有玩家同时进行一句话描述，无先后顺序。描述的对象可以是你收到的关键词，也可以是任何你认为可能是正确的关键词。"
        "当你不确定自己的角色时，可以通过模糊的描述从而隐藏自己的角色和关键词。\n"
        "2.预测阶段：根据本场游戏的历史上下文，判断自己是否是卧底，并预测其他的卧底和白板。\n"
        "3.投票阶段：为了达到你的胜利条件，请投出你的选票。被投票最多的人将被淘汰。\n"
        "4.游戏结束：当卧底全部被淘汰，或者仅剩两位玩家时，游戏结束。\n\n"
        "###胜利条件###\n\n"
        "你的胜利条件取决于你的角色：\n"
        "- 作为平民，你需要找出白板和卧底并通过投票将他们淘汰。请记住，优先淘汰白板。\n"
        "- 作为卧底，你需要隐藏自己的角色并潜伏下去，通过投票淘汰其他角色，活到最后你就胜利了。\n"
        "- 作为白板，你需要隐藏自己的角色并潜伏下去，通过投票淘汰卧底。\n\n"
        "###限定规则###\n\n"
        "- 你的描述应该足够简短，并且不能直接包含你收到的关键词。\n"
        "- 你的描述不能与之前的描述重复。\n"
        "- 在投票阶段，你不能投自己或者已经被淘汰的人，一人只能投一票。\n"
        "- 每句话必须以<EOS>作为结束。"
    )
    game_rule_without_blank = (
        "下面是游戏规则：\n\n"
        "###基础信息###\n\n"
        "游戏中有两个角色：平民，卧底。在游戏的开始，每个人都会收到一条关键词，仅自己可见\n"
        "- 平民们会收到相同的正确的关键词。\n"
        "- 卧底们会收到相同的错误的关键词，但是跟正确的关键词较为相似。\n"
        "你不会被告知自己的角色，只能通过观察和分析进行猜测。\n\n"
        "###游戏阶段###\n\n"
        "这场游戏有四个阶段:\n"
        "1.描述阶段：所有玩家同时进行一句话描述，无先后顺序。描述的对象可以是你收到的关键词，也可以是任何你认为可能是正确的关键词。"
        "当你不确定自己的角色时，可以通过模糊的描述从而隐藏自己的角色和关键词。\n"
        "2.预测阶段：根据本场游戏的历史上下文，判断自己是否是卧底，并预测其他的卧底。\n"
        "3.投票阶段：为了达到你的胜利条件，请投出你的选票。被投票最多的人将被淘汰。\n"
        "4.游戏结束：当卧底全部被淘汰，或者仅剩{set.num}位玩家时，游戏结束。\n\n"
        "###胜利条件###\n\n"
        "你的胜利条件取决于你的角色：\n"
        "- 作为平民，你需要找出卧底并通过投票将他们淘汰。\n"
        "- 作为卧底，你需要隐藏自己的角色并潜伏下去，通过投票淘汰其他角色，活到最后你就胜利了。\n"
        "###限定规则###\n\n"
        "- 你的描述应该足够简短，并且不能直接包含你收到的关键词。\n"
        "- 你的描述不能与之前的描述重复。\n"
        "- 在投票阶段，你不能投自己或者已经被淘汰的人，一人只能投一票。\n"
        "- 每句话必须以<EOS>作为结束。"
    )

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self.id2role: Dict[str, PlayerRoles] = {}
        self.role2players: Dict[PlayerRoles, List[Profile]] = {
            PlayerRoles.CIVILIAN: [],
            PlayerRoles.SPY: [],
            PlayerRoles.BLANK: []
        }
        self.id2player: Dict[str, Profile] = {}
        self.id2status: Dict[str, PlayerStatus] = {}
        self.civilian_key: KeyTypes = None
        self.spy_key: KeyTypes = None

    async def registry_players(self, players: List[Profile]) -> None:
        for player in players:
            self.id2player[player.id] = player
            self.id2status[player.id] = PlayerStatus.ALIVE

    async def init_game(self) -> ModeratorInitGameSummary:
        num_players = len(self.id2player)
        has_blank = self.env_var["has_blank"].current_value
        key_modality = self.env_var["key_modality"].current_value
        roles_assignment_strategy = {
            4: {
                PlayerRoles.CIVILIAN: 3,
                PlayerRoles.SPY: 1,
                PlayerRoles.BLANK: 0
            },
            5: {
                PlayerRoles.CIVILIAN: 3 if has_blank else 4,
                PlayerRoles.SPY: 1,
                PlayerRoles.BLANK: 1 if has_blank else 0
            },
            6: {
                PlayerRoles.CIVILIAN: 4 if has_blank else 5,
                PlayerRoles.SPY: 1,
                PlayerRoles.BLANK: 1 if has_blank else 0
            },
            7: {
                PlayerRoles.CIVILIAN: 4 if has_blank else 5,
                PlayerRoles.SPY: 2,
                PlayerRoles.BLANK: 1 if has_blank else 0
            },
            8: {
                PlayerRoles.CIVILIAN: 5 if has_blank else 6,
                PlayerRoles.SPY: 2,
                PlayerRoles.BLANK: 1 if has_blank else 0
            },
            9: {
                PlayerRoles.CIVILIAN: 6 if has_blank else 7,
                PlayerRoles.SPY: 2,
                PlayerRoles.BLANK: 1 if has_blank else 0
            }
        }
        roles_agent_num = roles_assignment_strategy[num_players]

        roles = list(chain(*[[role] * agent_num for role, agent_num in roles_agent_num.items()]))
        random.shuffle(roles)  # shuffle to randomize the role assignment
        for player_id, role in zip(list(self.id2player.keys()), roles):
            self.role2players[role].append(self.id2player[player_id])
            self.id2role[player_id] = role

        if key_modality == KeyModalities.TEXT:
            keys = random.choice(load_textual_key())
            self.civilian_key, self.spy_key = Text(text=keys["Civilian"]), Text(text=keys["Spy"])
        elif key_modality == KeyModalities.IMAGE:
            keys = random.choice(load_image_key())
            self.civilian_key, self.spy_key = Image(url=keys["Civilian"]), Image(url=keys["Spy"])
        else:
            raise NotImplementedError(f"[{key_modality.value}] modal not supported yet.")

        role_assign_summary = "\n".join(
            [f"- {role.value} :: {[f'{player.name}({player.id})' for player in players]}"
             for role, players in self.role2players.items()]
        )
        key_assign_summary = (
            f"- {PlayerRoles.CIVILIAN.value} :: {self.civilian_key}\n"
            f"- {PlayerRoles.SPY.value} :: {self.spy_key}"
        )
        msg = (
            f"## 角色和关键词分配结果\n\n"
            f"### 角色\n{role_assign_summary}\n\n### 关键词\n{key_assign_summary}"
        )
        return ModeratorInitGameSummary(
            sender=self.profile,
            receivers=[self.profile],
            content=Text(text=msg, display_text=msg),
            role2players={role.value: [p.name for p in players] for role, players in self.role2players.items()},
            keys={
                PlayerRoles.CIVILIAN.value: self.civilian_key.display_text,
                PlayerRoles.SPY.value: self.spy_key.display_text
            }
        )

    async def introduce_game_rule(self) -> ModeratorSummary:
        has_blank = self.env_var["has_blank"].current_value
        msg = self.game_rule_with_blank if has_blank else self.game_rule_without_blank
        return ModeratorSummary(
            sender=self.profile,
            receivers=list(self.id2player.values()),
            content=Text(text=msg, display_text=msg)
        )

    async def announce_game_start(self) -> ModeratorSummary:
        num_players = len(self.id2player)
        role2word = {
            PlayerRoles.CIVILIAN: f"名{PlayerRoles.CIVILIAN.value}",
            PlayerRoles.SPY: f"名{PlayerRoles.SPY.value}",
            PlayerRoles.BLANK: f"名{PlayerRoles.BLANK.value}"
        }
        roles_num_description = ", ".join(
            [f"{len(role_players)} {role2word[role]}" for role, role_players in self.role2players.items()]
        )
        msg = (
            f"现在游戏开始！本场游戏有 {num_players} 名玩家, 包括 "
            f"{roles_num_description}."
        )
        return ModeratorSummary(
            sender=self.profile,
            receivers=list(self.id2player.values()),
            content=Text(text=msg, display_text=msg)
        )

    async def assign_keys(self, player: Profile) -> ModeratorKeyAssignment:
        role = self.id2role[player.id]
        if role == PlayerRoles.CIVILIAN:
            return ModeratorKeyAssignment.create_with_key(
                key=self.civilian_key, sender=self.profile, receiver=player
            )
        elif role == PlayerRoles.SPY:
            return ModeratorKeyAssignment.create_with_key(
                key=self.spy_key, sender=self.profile, receiver=player
            )
        else:
            return ModeratorKeyAssignment.create_without_key(
                sender=self.profile, receiver=player
            )

    async def ask_for_key_description(self) -> ModeratorAskForDescription:
        return ModeratorAskForDescription.create(
            sender=self.profile,
            receivers=[
                player for player in self.id2player.values()
            ]
        )

    async def valid_player_description(self, description: PlayerDescription) -> ModeratorWarning:
        player_id = description.sender_id
        player_role = self.id2role[player_id]
        if player_role != PlayerRoles.BLANK and self.env_var["key_modality"].current_value == KeyModalities.TEXT:
            warn_msg = (
                "你的描述中包含你的关键词，这是不被允许的，请重新进行描述。回复仅包含你对关键词的描述，不需要多余的回答。"
            )
            if (player_role == PlayerRoles.CIVILIAN and self.civilian_key.text.lower() in description.content.text.lower()) or \
                    (player_role == PlayerRoles.SPY and self.spy_key.text.lower() in description.content.text.lower()):
                return ModeratorWarning(
                    sender=self.profile,
                    receivers=[description.sender],
                    content=Text(text=warn_msg, display_text=warn_msg),
                    has_warn=True
                )
        return ModeratorWarning(
            sender=self.profile,
            receivers=[description.sender],
            content=Text(text="", display_text=""),
            has_warn=False
        )

    async def ask_for_role_prediction(self) -> ModeratorAskForRolePrediction:
        has_blank = self.env_var["has_blank"].current_value
        return ModeratorAskForRolePrediction.create(
            sender=self.profile,
            receivers=[
                player for player in self.id2player.values() if self.id2status[player.id] == PlayerStatus.ALIVE
            ],
            player_names=[
                player.name for player in self.id2player.values() if self.id2status[player.id] == PlayerStatus.ALIVE
            ],
            has_blank=has_blank
        )

    async def summarize_players_prediction(self, predictions: List[PlayerPrediction]) -> ModeratorPredictionSummary:
        has_blank = self.env_var["has_blank"].current_value
        summaries = []
        extracted_predictions = {}
        for prediction in predictions:
            preds = prediction.get_prediction(
                player_names=[player.name for player in self.id2player.values()],
                has_blank=has_blank
            )
            extracted_predictions[prediction.sender_name] = {role.value: list(names) for role, names in preds.items()}
            summary = (
                f"**{prediction.sender_name}({prediction.sender_id})'s prediction**\n"
                f"- {PlayerRoles.SPY.value} :: {list(preds[PlayerRoles.SPY])}"
            )
            if has_blank:
                summary += f"\n- {PlayerRoles.BLANK.value} :: {list(preds[PlayerRoles.BLANK])}"
            summaries.append(summary)

        alive_spies = [
            player.name for player in self.role2players[PlayerRoles.SPY]
            if self.id2status[player.id] == PlayerStatus.ALIVE
        ]
        label = (
            f"**Correct Answer**\n- {PlayerRoles.SPY.value} :: {alive_spies}"
        )
        ground_truth = {PlayerRoles.SPY.value: alive_spies}
        if has_blank:
            alive_blanks = [
                player.name for player in self.role2players[PlayerRoles.BLANK]
                if self.id2status[player.id] == PlayerStatus.ALIVE
            ]
            label += f"\n- {PlayerRoles.BLANK.value} :: {alive_blanks}"
            ground_truth[PlayerRoles.BLANK.value] = alive_blanks
        msg = "\n\n".join(summaries) + f"\n\n{label}"
        return ModeratorPredictionSummary(
            sender=self.profile,
            receivers=[self.profile],
            content=Text(text=msg, display_text=msg),
            predictions=extracted_predictions,
            ground_truth=ground_truth
        )

    async def ask_for_vote(self, targets: List[Profile]) -> ModeratorAskForVote:
        return ModeratorAskForVote.create(
            sender=self.profile,
            receivers=[
                player for player in self.id2player.values() if self.id2status[player.id] == PlayerStatus.ALIVE
            ],
            targets=targets
        )

    async def summarize_player_votes(
        self,
        votes: List[PlayerVote],
        patience: int,
        focused_players: Optional[List[Profile]]
    ) -> ModeratorVoteSummary:
        def get_most_voted_players() -> List[Profile]:
            eliminated_names = [
                player_name for player_name, num_be_voted in player2num_be_voted.items() if
                num_be_voted == max(player2num_be_voted.values())
            ]
            return [player for player in self.id2player.values() if player.name in eliminated_names]

        player2num_be_voted = {player.name: 0 for player in self.id2player.values()}
        player2votes = {}
        for vote in votes:
            vote_to = vote.get_vote([player.name for player in self.id2player.values()])
            if not vote_to:
                continue
            player2votes[vote.sender_name] = vote_to
            player2num_be_voted[vote_to] += 1
        if focused_players:
            focused_names = [p.name for p in focused_players]
            for player_name in player2num_be_voted:
                if player_name not in focused_names:
                    player2num_be_voted[player_name] = 0

        voting_detail = "\n".join([f"{voter} 投票给 {voted}" for voter, voted in player2votes.items()]) + "\n"
        if focused_players:
            voting_detail += (
                f"这是一次针对平票玩家的重新投票, 因此将仅统计上次投票中平票玩家 {[p.name for p in focused_players]} 的本次得票。\n"
            )
        most_voted_players = get_most_voted_players()
        if len(most_voted_players) > 1 and patience > 0:  # tied
            msg = (
                f"{voting_detail}\n玩家 {[p.name for p in most_voted_players]} 有相同的票数。"
                f"请这些玩家额外进行一次针对自己获得的关键词的一句话描述。"
            )
            return ModeratorVoteSummary(
                sender=self.profile,
                receivers=[player for player in self.id2player.values()],
                content=Text(text=msg, display_text=msg),
                tied_players=most_voted_players,
                player_received_votes=player2num_be_voted,
                players_voted_to=player2votes
            )
        else:  # eliminate
            for player in most_voted_players:
                self.id2status[player.id] = PlayerStatus.ELIMINATED
            msg = f"{voting_detail}\n玩家 {[p.name for p in most_voted_players]} 获得的票数最多，本轮被淘汰。"
            return ModeratorVoteSummary(
                sender=self.profile,
                receivers=[player for player in self.id2player.values()],
                content=Text(text=msg, display_text=msg),
                player_received_votes=player2num_be_voted,
                players_voted_to=player2votes
            )

    async def check_if_game_over(self) -> ModeratorCheckGameOverSummary:
        def return_game_over(role: PlayerRoles):
            winners = [
                player.name for player in self.role2players[role]
                if self.id2status[player.id] == PlayerStatus.ALIVE
            ]
            msg = f"游戏结束！ {role.value} 胜利, 赢家是: {winners}."
            return ModeratorCheckGameOverSummary(
                sender=self.profile,
                receivers=[player for player in self.id2player.values()],
                content=Text(text=msg, display_text=msg),
                is_game_over=True,
                winners=winners
            )

        has_blank = self.env_var["has_blank"].current_value
        num_players = len(self.id2player)
        num_alive_players = len(
            [player for player, status in self.id2status.items() if status == PlayerStatus.ALIVE]
        )
        num_alive_civilians = len(
            [
                player for player in self.role2players[PlayerRoles.CIVILIAN]
                if self.id2status[player.id] == PlayerStatus.ALIVE
            ]
        )
        num_alive_spies = len(
            [
                player for player in self.role2players[PlayerRoles.SPY]
                if self.id2status[player.id] == PlayerStatus.ALIVE
            ]
        )
        if num_alive_civilians == num_alive_players:  # civilians win
            return return_game_over(PlayerRoles.CIVILIAN)
        if (
            (num_players > 6 and num_alive_players <= 3 and num_alive_spies > 0) or
            (num_players <= 6 and num_alive_players <= 2 and num_alive_spies > 0)
        ):  # spies win
            return return_game_over(PlayerRoles.SPY)
        if has_blank and num_alive_spies == 0 and num_alive_civilians != num_alive_players:  # blank wins
            return return_game_over(PlayerRoles.BLANK)

        msg = f"没有任何角色获胜，游戏继续。"
        return ModeratorCheckGameOverSummary(
            sender=self.profile,
            receivers=[player for player in self.id2player.values()],
            content=Text(text=msg, display_text=msg),
            is_game_over=False,
            winners=None
        )

    async def reset_inner_status(self):
        self.id2role: Dict[str, PlayerRoles] = {}
        self.role2players: Dict[PlayerRoles, List[Profile]] = {
            PlayerRoles.CIVILIAN: [],
            PlayerRoles.SPY: [],
            PlayerRoles.BLANK: []
        }
        self.id2player: Dict[str, Profile] = {}
        self.id2status: Dict[str, PlayerStatus] = {}
        self.civilian_key: Union[Audio, Image, Text] = None
        self.spy_key: Union[Audio, Image, Text] = None


__all__ = ["ModeratorConfig", "Moderator"]
