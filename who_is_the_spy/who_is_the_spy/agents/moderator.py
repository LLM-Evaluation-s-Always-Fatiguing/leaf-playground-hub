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
        "You are playing a game of Who is the spy. Here are the game rules:\n\n"
        "## Information and roles\n\n"
        "There are three roles in \"Who is the Spy?\": Spy, Civilian, and Blank Slate.\n"
        "- Civilians are shown the correct key.\n"
        "- Spies see a key similar to the correct one but incorrect.\n"
        "- Blank Slates receive a blank clue.\n"
        "Your role is unknown to you, so careful listening and inference are crucial to identify the spy.\n\n"
        "## Objectives\n\n"
        "Your objectives vary based on your role:\n"
        "- As a Civilian, your aim is to identify and vote out the Spy and the Blank Slate, without revealing the "
        "correct key. Focus first on finding the Blank Slate.\n"
        "- If you're the Spy, your goal is to blend in, avoid detection, and survive the voting. Winning occurs if "
        "at least one Spy remains at the end.\n"
        "- As a Blank Slate, try to uncover and vote out the Spy without revealing your own role. You can guess and "
        "describe what you think is the correct key.\n\n"
        "## Stages\n\n"
        "The game has two main stages and one special scenario:\n"
        "1. Giving Clues Stage: Each player gives clues about their key. Blank Slates can describe anything "
        "they choose.\n"
        "2. Accusation Stage: Here, Civilians vote for who they suspect is the Spy or Blank Slate. Spies vote "
        "for a likely Civilian or Blank Slate. Blank Slates vote for their suspected Spy.\n"
        "3. Tiebreaker Scenario: In the event of a tie, those with the most votes will re-describe their key, "
        "and a new vote takes place among them.\n\n"
        "## Code of Conduct\n\n"
        "Here are five rules of behavior you need to follow:\n"
        "- Your clues should be brief and not include the key.\n"
        "- Your clues can't duplicate the previous one.\n"
        "- Do not pretend you are other players or the moderator.\n"
        "- You cannot vote for yourself.\n"
        "- Always end your response with <EOS>."
    )
    game_rule_without_blank = (
        "You are playing a game of Who is the spy. Here are the game rules:\n\n"
        "## Information and roles\n\n"
        "There are two roles in \"Who is the Spy?\": Spy and Civilian.\n"
        "- Civilians are shown the correct key.\n"
        "- Spies see a key similar to the correct one but incorrect.\n"
        "Your role is unknown to you, so careful listening and inference are crucial to identify the spy.\n\n"
        "## Objectives\n\n"
        "Your objectives vary based on your role:\n"
        "- As a Civilian, your aim is to identify and vote out the Spy, without revealing the correct key.\n"
        "- If you're the Spy, your goal is to blend in, avoid detection, and survive the voting. Winning occurs "
        "if at least one Spy remains at the end.\n\n"
        "## Stages\n\n"
        "The game has two main stages and one special scenario:\n"
        "1. Giving Clues Stage: Each player gives clues about their key.\n"
        "2. Accusation Stage: Here, Civilians vote for who they suspect is the Spy or Blank Slate.Spies vote for "
        "a likely Civilian or Blank Slate.\n"
        "3. Tiebreaker Scenario: In the event of a tie, those with the most votes will re-describe their key, and "
        "a new vote takes place among them.\n\n"
        "## Code of Conduct\n\n"
        "Here are five rules of behavior you need to follow:\n"
        "- Your clues should be brief and not include the key.\n"
        "- Your clues can't duplicate the previous one.\n"
        "- Do not pretend you are other players or the moderator.\n"
        "- You cannot vote for yourself.\n"
        "- Always end your response with <EOS>."
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

    def registry_players(self, players: List[Profile]) -> None:
        for player in players:
            self.id2player[player.id] = player
            self.id2status[player.id] = PlayerStatus.ALIVE

    def init_game(self) -> ModeratorSummary:
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
        key_assign_summary = f"- civilian :: {self.civilian_key}\n- spy :: {self.spy_key}"
        msg = (
            f"## Roles and Keys assignment Results\n\n"
            f"### Roles\n{role_assign_summary}\n\n### Keys\n{key_assign_summary}"
        )
        return ModeratorSummary(
            sender=self.profile,
            receivers=[self.profile],
            content=Text(text=msg, display_text=msg)
        )

    def introduce_game_rule(self) -> ModeratorSummary:
        has_blank = self.env_var["has_blank"].current_value
        msg = self.game_rule_with_blank if has_blank else self.game_rule_without_blank
        return ModeratorSummary(
            sender=self.profile,
            receivers=list(self.id2player.values()),
            content=Text(text=msg, display_text=msg)
        )

    def announce_game_start(self) -> ModeratorSummary:
        num_players = len(self.id2player)
        role2word = {
            PlayerRoles.CIVILIAN: "civilians",
            PlayerRoles.SPY: "spies",
            PlayerRoles.BLANK: "blanks"
        }
        roles_num_description = ", ".join(
            [f"{len(role_players)} {role2word[role]}" for role, role_players in self.role2players.items()]
        )
        msg = (
            f"Now the game begins! There are {num_players} players in this game, including "
            f"{roles_num_description}."
        )
        return ModeratorSummary(
            sender=self.profile,
            receivers=list(self.id2player.values()),
            content=Text(text=msg, display_text=msg)
        )

    def assign_keys(self, player: Profile) -> ModeratorKeyAssignment:
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

    def ask_for_key_description(self) -> ModeratorAskForDescription:
        return ModeratorAskForDescription.create(
            sender=self.profile,
            receivers=[
                player for player in self.id2player.values() if self.id2status[player.id] == PlayerStatus.ALIVE
            ]
        )

    def valid_player_description(self, description: PlayerDescription) -> ModeratorWarning:
        player_id = description.sender_id
        player_role = self.id2role[player_id]
        if player_role != PlayerRoles.BLANK and self.env_var["key_modality"].current_value == KeyModalities.TEXT:
            warn_msg = "Your description contains your key, which is not allowed, please redo the description."
            if (player_role == PlayerRoles.CIVILIAN and self.civilian_key.text in description.content.text) or \
                    (player_role == PlayerRoles.SPY and self.spy_key.text in description.content.text):
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

    def ask_for_role_prediction(self) -> ModeratorAskForRolePrediction:
        has_blank = self.env_var["has_blank"].current_value
        return ModeratorAskForRolePrediction.create(
            sender=self.profile,
            receivers=[
                player for player in self.id2player.values() if self.id2status[player.id] == PlayerStatus.ALIVE
            ],
            player_names=[
                player.name for player in self.id2player.values() if self.id2status[player.id] == PlayerStatus.ALIVE
            ],
            has_blank_slate=has_blank
        )

    def summarize_players_prediction(self, predictions: List[PlayerPrediction]) -> ModeratorSummary:
        has_blank = self.env_var["has_blank"].current_value
        summaries = []
        for prediction in predictions:
            preds = prediction.get_prediction(
                player_names=[player.name for player in self.id2player.values()],
                has_blank_slate=has_blank
            )
            summary = (
                f"### {prediction.sender_name}({prediction.sender_id})'s prediction\n"
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
            f"### Correct Answer\n- {PlayerRoles.SPY.value} :: {alive_spies}"
        )
        if has_blank:
            alive_blanks = [
                player.name for player in self.role2players[PlayerRoles.BLANK]
                if self.id2status[player.id] == PlayerStatus.ALIVE
            ]
            label += f"\n- {PlayerRoles.BLANK.value} :: {alive_blanks}"
        msg = "\n\n".join(summaries) + f"\n\n{label}"
        return ModeratorSummary(
            sender=self.profile,
            receivers=[self.profile],
            content=Text(text=msg, display_text=msg)
        )

    def ask_for_vote(self) -> ModeratorAskForVote:
        has_blank = self.env_var["has_blank"].current_value
        return ModeratorAskForVote.create(
            sender=self.profile,
            receivers=[
                player for player in self.id2player.values() if self.id2status[player.id] == PlayerStatus.ALIVE
            ],
            has_blank_slate=has_blank
        )

    def summarize_player_votes(
        self,
        votes: List[PlayerVote],
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

        voting_detail = "\n".join([f"{voter} votes to {voted}" for voter, voted in player2votes.items()]) + "\n"
        if focused_players:
            voting_detail += (
                f"This is a re-voting turn, we will only focus on the votes {[p.name for p in focused_players]} got.\n"
            )
        most_voted_players = get_most_voted_players()
        if len(most_voted_players) > 1:  # tied
            msg = (
                f"{voting_detail}{[p.name for p in most_voted_players]} are having the same "
                f"votes, for those players, please re-describe the key you received again."
            )
            return ModeratorVoteSummary(
                sender=self.profile,
                receivers=[player for player in self.id2player.values()],
                content=Text(text=msg, display_text=msg),
                tied_players=most_voted_players
            )
        else:  # eliminate
            for player in most_voted_players:
                self.id2status[player.id] = PlayerStatus.ELIMINATED
            msg = f"{voting_detail}{most_voted_players[0].name} has the most votes and is eliminated."
            return ModeratorVoteSummary(
                sender=self.profile,
                receivers=[player for player in self.id2player.values()],
                content=Text(text=msg, display_text=msg)
            )

    def check_if_game_over(self) -> ModeratorSummary:
        def return_game_over(role: PlayerRoles):
            winners = [
                player.name for player in self.role2players[role]
                if self.id2status[player.id] == PlayerStatus.ALIVE
            ]
            msg = f"Game Over! {role.value} win, winners are: {winners}."
            return ModeratorSummary(
                sender=self.profile,
                receivers=[player for player in self.id2player.values()],
                content=Text(text=msg, display_text=msg),
                is_game_over=True
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

        msg = f"Not any side wins, game continues."
        return ModeratorSummary(
            sender=self.profile,
            receivers=[player for player in self.id2player.values()],
            content=Text(text=msg, display_text=msg),
        )

    def reset_inner_status(self):
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
