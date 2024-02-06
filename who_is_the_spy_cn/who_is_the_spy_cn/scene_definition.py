import json
from enum import Enum
from typing import Dict, List, Literal, Optional, Set, Union

from leaf_playground.core.scene_definition.definitions.metric import _RecordData
from typing_extensions import Annotated

from pydantic import Field, BaseModel, create_model

from leaf_playground.core.scene_definition import *
from leaf_playground.data.environment import ConstantEnvironmentVariable
from leaf_playground.data.message import TextMessage
from leaf_playground.data.media import Audio, Image, Text
from leaf_playground.data.profile import Profile

from .text_utils import get_most_similar_text

KEY_PLACEHOLDER = "<关键词>"


class PlayerRoles(Enum):
    CIVILIAN = "平民"
    SPY = "卧底"
    BLANK = "白板"


class PlayerStatus(Enum):
    ALIVE = "存活"
    ELIMINATED = "被淘汰"


class KeyModalities(Enum):
    TEXT = "文本"
    IMAGE = "图片"
    # AUDIO = "音频"


KeyTypes = Annotated[Union[Text, Image, Audio], Field(discriminator="type")]


class KeyModality(ConstantEnvironmentVariable):
    current_value: KeyModalities = Field(default=KeyModalities.TEXT)


class NumGames(ConstantEnvironmentVariable):
    current_value: int = Field(default=1)


class HasBlank(ConstantEnvironmentVariable):
    current_value: bool = Field(default=False)


class ModeratorSummary(TextMessage):
    msg_type: Literal["ModeratorSummary"] = Field(default="ModeratorSummary")


class ModeratorInitGameSummary(ModeratorSummary):
    msg_type: Literal["ModeratorInitGameSummary"] = Field(default="ModeratorInitGameSummary")
    role2players: Dict[str, List[str]] = Field(default=...)
    keys: Dict[str, str] = Field(default=...)


class ModeratorPredictionSummary(ModeratorSummary):
    msg_type: Literal["ModeratorPredictionSummary"] = Field(default="ModeratorPredictionSummary")
    predictions: Dict[str, Dict[str, List[str]]] = Field(default=...)
    ground_truth: Dict[str, List[str]] = Field(default=...)


class ModeratorVoteSummary(ModeratorSummary):
    msg_type: Literal["ModeratorVoteSummary"] = Field(default="ModeratorVoteSummary")
    tied_players: Optional[List[Profile]] = Field(default=None)
    player_received_votes: Dict[str, int] = Field(default=...)
    players_voted_to: Dict[str, str] = Field(default=...)


class ModeratorCheckGameOverSummary(ModeratorSummary):
    msg_type: Literal["ModeratorCheckGameOverSummary"] = Field(default="ModeratorCheckGameOverSummary")
    is_game_over: bool = Field(default=False)
    winners: Optional[List[str]] = Field(default=None)


class ModeratorKeyAssignment(TextMessage):
    msg_type: Literal["ModeratorKeyAssignment"] = Field(default="ModeratorKeyAssignment")
    key: Optional[KeyTypes] = Field(default=None)

    @classmethod
    def create_with_key(cls, key: Union[Audio, Image, Text], sender: Profile, receiver: Profile):
        return cls(
            sender=sender,
            receivers=[receiver],
            content=Text(
                text=f"{receiver.name}, 你的关键词是： {KEY_PLACEHOLDER}。这可能是正确或者错误的关键词",
                display_text=f"{receiver.name}, 你的关键词是： [{key.display_text}]。这可能是正确或者错误的关键词"
            ),
            key=key
        )

    @classmethod
    def create_without_key(cls, sender: Profile, receiver: Profile):
        return cls(
            sender=sender,
            receivers=[receiver],
            content=Text(text=f"{receiver.name}, 你的角色是白板."),
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
                text=f"描述阶段：现在，请所有玩家依次进行一句话描述，描述的对象可以是你收到的关键词也可以是你认为正确的关键词."
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
        has_blank: bool
    ) -> "ModeratorAskForRolePrediction":
        if has_blank:
            msg = (
                f"预测阶段：\n"
                f"现在，进行角色预测，仅自己可见。首先分析自己是否是卧底，然后找出可能存在的卧底和白板。\n"
                f"请保持回答的简短，并用以下形式作为你回答的结尾：\n"
                f"卧底：[<spy_name>, ..., <spy_name>]；白板：[<blank_name>]<EOS>\n"
                f"其中<spy_name>是你认为卧底的玩家名，<blank_name>是你认为白板的玩家名。\n"
                f"玩家的名字是: {player_names}.\n"
            )
        else:
            msg = (
                f"预测阶段：\n"
                f"现在，进行角色预测，仅自己可见。首先分析自己是否是卧底，然后找出可能存在的卧底。\n"
                f"请保持回答的简短，并用以下形式作为你回答的结尾：\n"
                f"卧底：[<spy_name>, ..., <spy_name>]<EOS>\n"
                f"其中<spy_name>是你认为卧底的玩家名。\n"
                f"玩家的名字是: {player_names}.\n"
            )
        return cls(
            sender=sender,
            receivers=receivers,
            content=Text(text=msg, display_text=msg)
        )


class ModeratorAskForVote(TextMessage):
    msg_type: Literal["ModeratorAskForVote"] = Field(default="ModeratorAskForVote")

    @classmethod
    def create(
        cls,
        sender: Profile,
        receivers: List[Profile],
        targets: List[Profile]
    ) -> "ModeratorAskForVote":
        msg = (
            "投票阶段：现在，请结合之前的角色预测，进行投票。"
            "为了达到自己的胜利条件，请做出选择，并以以下形式给我你的投票：\n"
            "投票：<player_name><EOS>\n"
            "<player_name>是你想投票的玩家名。\n"
            f"可以被投票的玩家是: {','.join([target.name for target in targets])}.\n"
            "你的回答应该以 '投票：' 作为开头"
        )
        return cls(
            sender=sender,
            receivers=receivers,
            content=Text(text=msg, display_text=msg)
        )


class ModeratorWarning(TextMessage):
    msg_type: Literal["ModeratorWarning"] = Field(default="ModeratorWarning")
    has_warn: bool = Field(default=...)


class PlayerDescription(TextMessage):
    msg_type: Literal["PlayerDescription"] = Field(default="PlayerDescription")


class PlayerPrediction(TextMessage):
    msg_type: Literal["PlayerPrediction"] = Field(default="PlayerPrediction")

    def get_prediction(self, player_names: List[str], has_blank: bool) -> Dict[PlayerRoles, Set[str]]:
        def retrieve_names(symbol: str) -> Set[str]:
            names = set()
            content = self.content.text
            if symbol in content:
                content = content[content.index(symbol) + len(symbol):].strip()
                content = content.split("：")[0].strip()
                for pred in content.split(","):
                    pred = pred.strip()
                    names.add(
                        get_most_similar_text(pred, [each.strip() for each in player_names])
                    ) if "[]" not in pred else names.add("")
            return names

        preds = {PlayerRoles.SPY: retrieve_names("卧底：")}
        if has_blank:
            preds[PlayerRoles.BLANK] = retrieve_names("白板：")
        return preds


class PlayerVote(TextMessage):
    msg_type: Literal["PlayerVote"] = Field(default="PlayerVote")

    @staticmethod
    def generate_data_schema(player_names: List[str]) -> dict:
        CurrentPlayers = Literal[tuple(player_names)]

        DynamicUserModel = create_model(
            "卧底投票",
            vote_target=(CurrentPlayers, Field(..., description="选择玩家进行投票"))
        )

        return DynamicUserModel.model_json_schema()

    def get_vote(self, player_names: List[str]) -> str:
        vote = self.content.text
        get_vote = False
        if "投票：" in vote:
            vote = vote[vote.index("投票：") + len("投票："):].strip()
            vote = get_most_similar_text(vote, [each.strip() for each in player_names])
            get_vote = True

        return vote if get_vote else ""


MessageTypes = Annotated[
    Union[
        ModeratorSummary, ModeratorInitGameSummary, ModeratorPredictionSummary, ModeratorVoteSummary,
        ModeratorCheckGameOverSummary, ModeratorKeyAssignment, ModeratorWarning,
        ModeratorAskForDescription, ModeratorAskForRolePrediction, ModeratorAskForVote,
        PlayerDescription, PlayerPrediction, PlayerVote
    ],
    Field(discriminator="msg_type")
]


def avg_fn(records: List[_RecordData]) -> AggregationMethodOutput:
    avg = round(sum(record.value for record in records) / len(records), 4)
    return AggregationMethodOutput(value=avg)


SCENE_DEFINITION = SceneDefinition(
    name="谁是卧底",
    description="谁是卧底是一款游戏。游戏中有三种角色：平民，卧底和白板。每位玩家在游戏开始时都会被告知一个关键词，"
                "其中平民与卧底的关键词不同但相似，白板是一个空白关键词。玩家需要对自己的关键词进行描述，每轮描述结束"
                "后进行投票，获票最多的玩家将被淘汰。平民需要抓出每一位卧底和白板，卧底需要假装平民活到最后，白板的胜利"
                "条件是淘汰所有的卧底后依旧留在场上。",
    roles=[
        RoleDefinition(
            name="moderator",
            description="主持游戏的角色",
            num_agents_range=(1, 1),
            is_static=True,
            actions=[
                ActionDefinition(
                    name="registry_players",
                    description="注册玩家",
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
                    description="初始化游戏",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorInitGameSummary,
                        is_static_method=False
                    )
                ),
                ActionDefinition(
                    name="introduce_game_rule",
                    description="介绍游戏规则",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorSummary
                    )
                ),
                ActionDefinition(
                    name="announce_game_start",
                    description="宣布游戏开始",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorSummary
                    )
                ),
                ActionDefinition(
                    name="assign_keys",
                    description="给玩家分配关键词",
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
                    description="询问描述",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorAskForDescription
                    )
                ),
                ActionDefinition(
                    name="valid_player_description",
                    description="验证描述格式",
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
                    description="开始身份预测",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorAskForRolePrediction
                    )
                ),
                ActionDefinition(
                    name="summarize_players_prediction",
                    description="总结身份预测",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="predictions",
                                annotation=List[PlayerPrediction]
                            )
                        ],
                        return_annotation=ModeratorPredictionSummary
                    )
                ),
                ActionDefinition(
                    name="ask_for_vote",
                    description="进行投票",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="targets",
                                annotation=List[Profile]
                            )
                        ],
                        return_annotation=ModeratorAskForVote
                    )
                ),
                ActionDefinition(
                    name="summarize_player_votes",
                    description="总结投票结果",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="votes",
                                annotation=List[PlayerVote]
                            ),
                            ActionSignatureParameterDefinition(
                                name="patience",
                                annotation=int
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
                    description="检查游戏是否结束",
                    signature=ActionSignatureDefinition(
                        parameters=None,
                        return_annotation=ModeratorCheckGameOverSummary
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
            description="参与游戏的角色",
            num_agents_range=(4, 9),
            is_static=False,
            actions=[
                ActionDefinition(
                    name="receive_key",
                    description="收到关键词",
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
                    description="描述关键词",
                    exec_timeout=120,
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
                    ),
                    metrics=[MetricDefinition(
                        name="伪装能力",
                        description=f"玩家对于词汇描述的逻辑性",
                        record_value_dtype=ValueDType.INT,
                        record_display_type=DisplayType.FIVE_STARTS_RATE,
                        expect_resp_msg_type=PlayerDescription,
                        agg_method=DynamicAggregationFn.create_dynamic_fn(fn=avg_fn),
                        is_comparison=False
                    )]
                ),
                ActionDefinition(
                    name="predict_role",
                    description="角色预测",
                    exec_timeout=120,
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
                    ),
                    metrics=[MetricDefinition(
                        name="推理能力",
                        description=f"玩家根据发言，推理分析其他玩家身份与关键词的能力",
                        record_value_dtype=ValueDType.INT,
                        record_display_type=DisplayType.FIVE_STARTS_RATE,
                        expect_resp_msg_type=PlayerPrediction,
                        agg_method=DynamicAggregationFn.create_dynamic_fn(fn=avg_fn),
                        is_comparison=False
                    )]
                ),
                ActionDefinition(
                    name="vote",
                    description="投票",
                    exec_timeout=120,
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="history",
                                annotation=List[MessageTypes]
                            ),
                            ActionSignatureParameterDefinition(
                                name="moderator",
                                annotation=Profile
                            ),
                            ActionSignatureParameterDefinition(
                                name="player_names",
                                annotation=List[str]
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
            description="关键词的模态类型, 可选：文本、图片",
            env_var_cls=KeyModality
        ),
        EnvVarDefinition(
            name="num_games",
            description="游戏进行多少场",
            env_var_cls=NumGames
        ),
        EnvVarDefinition(
            name="has_blank",
            description="游戏中是否包含白板玩家",
            env_var_cls=HasBlank
        )
    ]
)

__all__ = [
    "KEY_PLACEHOLDER",
    "KeyTypes",
    "PlayerRoles",
    "PlayerStatus",
    "KeyModalities",
    "KeyModality",
    "HasBlank",
    "NumGames",
    "PlayerDescription",
    "PlayerPrediction",
    "PlayerVote",
    "ModeratorSummary",
    "ModeratorInitGameSummary",
    "ModeratorPredictionSummary",
    "ModeratorVoteSummary",
    "ModeratorCheckGameOverSummary",
    "ModeratorWarning",
    "ModeratorKeyAssignment",
    "ModeratorAskForDescription",
    "ModeratorAskForRolePrediction",
    "ModeratorAskForVote",
    "MessageTypes",
    "SCENE_DEFINITION"
]
