from enum import Enum
from typing import Dict, List, Literal, Optional, Set, Union
from typing_extensions import Annotated

from pydantic import Field

from leaf_playground.core.scene_definition import *
from leaf_playground.data.environment import ConstantEnvironmentVariable
from leaf_playground.data.message import TextMessage
from leaf_playground.data.media import Audio, Image, Text
from leaf_playground.data.profile import Profile

from .text_utils import get_most_similar_text


KEY_PLACEHOLDER = "<KEY>"


class PlayerRoles(Enum):
    CIVILIAN = "civilian"
    SPY = "spy"
    BLANK = "blank"


class PlayerStatus(Enum):
    ALIVE = "alive"
    ELIMINATED = "eliminated"


class KeyModalities(Enum):
    TEXT = "text"
    # IMAGE = "image"
    # AUDIO = "audio"


KeyTypes = Annotated[Union[Text, Image, Audio], Field(discriminator="type")]


class KeyModalityEnvVar(ConstantEnvironmentVariable):
    current_value: KeyModalities = Field(default=KeyModalities.TEXT)


class NumRoundsEnvVar(ConstantEnvironmentVariable):
    current_value: int = Field(default=1)


class HasBlankEnvVar(ConstantEnvironmentVariable):
    current_value: bool = Field(default=False)


class ModeratorSummary(TextMessage):
    msg_type: Literal["ModeratorSummary"] = Field(default="ModeratorSummary")
    is_game_over: bool = Field(default=False)


class ModeratorVoteSummary(ModeratorSummary):
    msg_type: Literal["ModeratorVoteSummary"] = Field(default="ModeratorVoteSummary")
    tied_players: Optional[List[Profile]] = Field(default=None)


class ModeratorKeyAssignment(TextMessage):
    msg_type: Literal["ModeratorKeyAssignment"] = Field(default="ModeratorKeyAssignment")
    key: Optional[KeyTypes] = Field(default=None)

    @classmethod
    def create_with_key(cls, key: Union[Audio, Image, Text], sender: Profile, receiver: Profile):
        return cls(
            sender=sender,
            receivers=[receiver],
            content=Text(text=f"{receiver.name}, your key is: {KEY_PLACEHOLDER}"),
            key=key
        )

    @classmethod
    def create_without_key(cls, sender: Profile, receiver: Profile):
        return cls(
            sender=sender,
            receivers=[receiver],
            content=Text(text=f"{receiver.name}, you got a blank clue."),
            key=None
        )


class ModeratorAskForDescription(TextMessage):
    msg_type: Literal["ModeratorAskForDescription"] = Field(default="ModeratorAskForDescription")

    @classmethod
    def create(cls, sender: Profile, receivers: List[Profile]) -> "ModeratorAskForDescription":
        return cls(
            sender=sender,
            receivers=receivers,
            content=Text(
                text=f"Now, please using ONE-sentence to describe your key."
            )
        )


class ModeratorAskForRolePrediction(TextMessage):
    msg_type: Literal["ModeratorAskForRolePrediction"] = Field(default="ModeratorAskForRolePrediction")

    @classmethod
    def create(
        cls,
        sender: Profile,
        receivers: List[Profile],
        player_names: List[str],
        has_blank_slate: bool
    ) -> "ModeratorAskForRolePrediction":
        if has_blank_slate:
            msg = (
                f"Now, think about who is the spy and who is the blank slate. "
                f"And tell me your predictions in the following format:\n"
                "spy: [<player_name>, ..., <player_name>]; blank: [<player_name>, ..., <player_name>]<EOS>\n"
                "Where <player_name> is the name of the player you think is the spy or the blank slate.\n"
            )
        else:
            msg = (
                f"Now, think about who is the spy. "
                f"And tell me your predictions in the following format:\n"
                f"spy: [<player_name>, ..., <player_name>]<EOS>\n"
                f"Where <player_name> is the name of the player you think is the spy.\n"
            )
        return cls(
            sender=sender,
            receivers=receivers,
            content=Text(
                text=msg + f"Player names are: {player_names}.\nYour response MUST starts with 'spy:'"
            )
        )


class ModeratorAskForVote(TextMessage):
    msg_type: Literal["ModeratorAskForVote"] = Field(default="ModeratorAskForVote")

    @classmethod
    def create(
        cls,
        sender: Profile,
        receivers: List[Profile],
        has_blank_slate: bool
    ) -> "ModeratorAskForVote":
        if has_blank_slate:
            msg = (
                "And now，let's vote for who should be eliminated. "
                "Civilians vote for who they suspect is the Spy or Blank Slate. "
                "Spies vote for a likely Civilian or Blank Slate. "
                "Blank Slates vote for their suspected Spy. "
                "Each person can only cast one vote and cannot vote for themselves, "
                "please send me your vote in the following format:\n"
                "vote: <player_name><EOS>\n"
                "Where <player_name> is the name of the player you want to vote for.\n"
            )
        else:
            msg = (
                "And now, let's vote for who should be eliminated. "
                "Civilians vote for who they suspect is the Spy. "
                "Spies vote for a likely Civilian. "
                "Each person can only cast one vote and cannot vote for themselves, "
                "please send me your vote in the following format:\n"
                "vote: <player_name><EOS>\n"
                "Where <player_name> is the name of the player you want to vote for.\n"
            )
        return cls(
            sender=sender,
            receivers=receivers,
            content=Text(
                text=msg + f"Player names are: {','.join([player.name for player in receivers])}.\n"
                           "Your response MUST starts with 'vote:'"
            )
        )


class ModeratorWarning(TextMessage):
    msg_type: Literal["ModeratorWarning"] = Field(default="ModeratorWarning")
    has_warn: bool = Field(default=...)


class PlayerDescription(TextMessage):
    msg_type: Literal["PlayerDescription"] = Field(default="PlayerDescription")


class PlayerPrediction(TextMessage):
    msg_type: Literal["PlayerPrediction"] = Field(default="PlayerPrediction")

    def get_prediction(self, player_names: List[str], has_blank_slate: bool) -> Dict[PlayerRoles, Set[str]]:
        def retrieve_names(symbol: str) -> Set[str]:
            names = set()
            content = self.content.text
            if symbol in names:
                content = content[content.index(symbol) + len(symbol):].strip()
                content = content.split(":")[0].strip()
                for pred in content.split(","):
                    pred = pred.strip()
                    names.add(get_most_similar_text(pred, [each.strip() for each in player_names]))
            return names

        preds = {PlayerRoles.SPY: retrieve_names("spy:")}
        if has_blank_slate:
            preds[PlayerRoles.BLANK] = retrieve_names("blank:")
        return preds


class PlayerVote(TextMessage):
    msg_type: Literal["PlayerVote"] = Field(default="PlayerVote")

    def get_vote(self, player_names: List[str]) -> str:
        vote = self.content.text
        get_vote = False
        if "vote:" in vote:
            vote = vote[vote.index("vote:") + len("vote:"):].strip()
            vote = get_most_similar_text(vote, [each.strip() for each in player_names])
            get_vote = True

        return vote if get_vote else ""


MessageTypes = Annotated[
    Union[
        ModeratorSummary, ModeratorVoteSummary, ModeratorKeyAssignment, ModeratorWarning,
        ModeratorAskForDescription, ModeratorAskForRolePrediction, ModeratorAskForVote,
        PlayerDescription, PlayerPrediction, PlayerVote
    ],
    Field(discriminator="msg_type")
]


SCENE_DEFINITION = SceneDefinition(
    name="WhoIsTheSpy",
    description="A scene that simulates the Who is the Spy game.",
    roles=[
        RoleDefinition(
            name="moderator",
            description="the one that moderate the game",
            num_agents_range=(1, 1),
            is_static=True,
            actions=[
                ActionDefinition(
                    name="registry_players",
                    description="registry players",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="players",
                                annotation=List[Profile]
                            )
                        ],
                        return_annotation=None,
                        is_static_method=False
                    )
                ),
                ActionDefinition(
                    name="init_game",
                    description="initialize game",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorSummary,
                        is_static_method=False
                    )
                ),
                ActionDefinition(
                    name="introduce_game_rule",
                    description="introduce game rules",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorSummary
                    )
                ),
                ActionDefinition(
                    name="announce_game_start",
                    description="announce game start",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorSummary
                    )
                ),
                ActionDefinition(
                    name="assign_keys",
                    description="assign keys to players",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="player",
                                annotation=Profile
                            )
                        ],
                        return_annotation=ModeratorKeyAssignment
                    )
                ),
                ActionDefinition(
                    name="ask_for_key_description",
                    description="ask a player to give a key description",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorAskForDescription
                    )
                ),
                ActionDefinition(
                    name="valid_player_description",
                    description="validate player's description",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="description",
                                annotation=PlayerDescription
                            )
                        ],
                        return_annotation=ModeratorWarning
                    )
                ),
                ActionDefinition(
                    name="ask_for_role_prediction",
                    description="ask a player to predict other player's role",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorAskForRolePrediction
                    )
                ),
                ActionDefinition(
                    name="summarize_players_prediction",
                    description="summarize all players' role predictions",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="predictions",
                                annotation=List[PlayerPrediction]
                            )
                        ],
                        return_annotation=ModeratorSummary
                    )
                ),
                ActionDefinition(
                    name="ask_for_vote",
                    description="ask a player to vote",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorAskForVote
                    )
                ),
                ActionDefinition(
                    name="summarize_player_votes",
                    description="summarize all players' votes",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="votes",
                                annotation=List[PlayerVote]
                            ),
                            ActionSignatureParameterDefinition(
                                name="focused_players",
                                annotation=Optional[List[Profile]]
                            )
                        ],
                        return_annotation=ModeratorVoteSummary
                    )
                ),
                ActionDefinition(
                    name="check_if_game_over",
                    description="check whether the game is over",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorSummary
                    )
                ),
                ActionDefinition(
                    name="reset_inner_status",
                    description="reset inner status",
                    signature=ActionSignatureDefinition()
                )
            ]
        ),
        RoleDefinition(
            name="player",
            description="the one that plays the game",
            num_agents_range=(4, 9),
            is_static=False,
            actions=[
                ActionDefinition(
                    name="receive_key",
                    description="receive key",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="key_assignment",
                                annotation=ModeratorKeyAssignment
                            )
                        ],
                        return_annotation=None
                    )
                ),
                ActionDefinition(
                    name="describe_key",
                    description="describe key",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="history",
                                annotation=List[MessageTypes]
                            ),
                            ActionSignatureParameterDefinition(
                                name="receivers",
                                annotation=List[Profile]
                            )
                        ],
                        return_annotation=PlayerDescription
                    )
                ),
                ActionDefinition(
                    name="predict_role",
                    description="predict role",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="history",
                                annotation=List[MessageTypes]
                            ),
                            ActionSignatureParameterDefinition(
                                name="moderator",
                                annotation=Profile
                            )
                        ],
                        return_annotation=PlayerPrediction
                    )
                ),
                ActionDefinition(
                    name="vote",
                    description="vote",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="history",
                                annotation=List[MessageTypes]
                            ),
                            ActionSignatureParameterDefinition(
                                name="moderator",
                                annotation=Profile
                            )
                        ],
                        return_annotation=PlayerVote
                    )
                ),
                ActionDefinition(
                    name="reset_inner_status",
                    description="reset inner status",
                    signature=ActionSignatureDefinition()
                )
            ]
        )
    ],
    env_vars=[
        EnvVarDefinition(
            name="key_modality",
            description="the modality of the key, i.e: text, image, audio, etc.",
            env_var_cls=KeyModalityEnvVar
        ),
        EnvVarDefinition(
            name="num_rounds",
            description="num rounds to play",
            env_var_cls=NumRoundsEnvVar
        ),
        EnvVarDefinition(
            name="has_blank",
            description="whether the game has blank",
            env_var_cls=HasBlankEnvVar
        )
    ]
)


__all__ = [
    "KEY_PLACEHOLDER",
    "KeyTypes",
    "PlayerRoles",
    "PlayerStatus",
    "KeyModalities",
    "KeyModalityEnvVar",
    "HasBlankEnvVar",
    "NumRoundsEnvVar",
    "PlayerDescription",
    "PlayerPrediction",
    "PlayerVote",
    "ModeratorSummary",
    "ModeratorVoteSummary",
    "ModeratorWarning",
    "ModeratorKeyAssignment",
    "ModeratorAskForDescription",
    "ModeratorAskForRolePrediction",
    "ModeratorAskForVote",
    "MessageTypes",
    "SCENE_DEFINITION"
]
