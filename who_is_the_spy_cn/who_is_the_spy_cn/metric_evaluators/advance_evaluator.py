from typing import Any, Dict, List, Optional, Type

from leaf_playground.core.workers import MetricEvaluatorConfig, MetricEvaluator
from leaf_playground.core.workers.evaluator import _MetricName, CompareOutput, RecordOutput
from leaf_playground.data.media import Text
from leaf_playground.data.message import Message
from leaf_playground.eval_tools.general_open import GeneralOpenEvalToolConfig, GeneralOpenEvalTool
from pydantic import Field

from ..scene_definition import PlayerDescription, SCENE_DEFINITION, ModeratorInitGameSummary, PlayerPrediction

ROLE_NAME = "player"
ACTION_NAMES = ["describe_key", "predict_role"]

ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition(ROLE_NAME)

SUPPORT_METRIC_NAMES = ["伪装能力", "推理能力"]
SUPPORT_METRIC_FULL_NAMES = ["player.describe_key.伪装能力", "player.predict_role.推理能力"]
SUPPORT_METRICS = ROLE_DEFINITION.get_action_definition(
    ACTION_NAMES[0]
).metrics + ROLE_DEFINITION.get_action_definition(ACTION_NAMES[1]).metrics

SYSTEM_TEMPLATE_DICT = {
    "伪装能力": """你是一名「谁是卧底」的游戏评估专家。你需要评估每名玩家在游戏中的表现。

游戏规则：
游戏中有三个角色：平民，卧底和白板。在游戏的开始，每个人都会收到一条关键词，仅自己可见
- 平民们会收到相同的正确的关键词。
- 卧底们会收到相同的错误的关键词，但是跟正确的关键词较为相似。
- 白板会被告知自己是白板角色。

现在是游戏阶段，每个玩家会按照顺序描述自己的关键词。
描述的对象可以是你收到的关键词，也可以是任何你认为可能是正确的关键词。
平民总是希望找出白板和卧底，卧底总是希望隐藏自己的身份，白板总是希望找出卧底。

请用以下的评估标准，对当前玩家的伪装能力进行评分：

得分 1：该发言与关键词无关，不符合游戏规则。 
得分 2：该发言虽然与关键词相关，但是完全重复他人或自己的历史发言。
得分 3：该发言非常直白，让人一眼就能猜出描述的关键词。
得分 4：该发言较为模糊，仅能用于描述自己的关键词，不容易猜出所描述关键词是什么。
得分 5：该发言足够模糊不容易暴露关键词，但是用在不同角色的关键词下又都恰当合理，让人无法判断该发言的角色身份。

""",
"推理能力": """你是一名「谁是卧底」的游戏评估专家。你需要评估每名玩家在游戏中的表现。

游戏规则：
游戏中有三个角色：平民，卧底和白板。在游戏的开始，每个人都会收到一条关键词，仅自己可见
- 平民们会收到相同的正确的关键词。
- 卧底们会收到相同的错误的关键词，但是跟正确的关键词较为相似。
- 白板会被告知自己是白板角色。

现在是推理阶段，每个玩家在之前已经按照顺序描述了自己的关键词，现在，每个玩家都在分析其他玩家的描述，尝试推理出有效信息。
平民总是希望找出白板和卧底，卧底总是希望隐藏自己的身份，白板总是希望找出卧底。

请用以下的评估标准，对当前玩家的推理过程和结果进行评分：

得分 1：该推理毫无有效信息。 
得分 2：该推理逻辑性不佳，没有得出有价值的结论。
得分 3：该推理找到了部分有价值的信息，但是分析过程存在逻辑上的错误。
得分 4：该推理过程逻辑性强，成功从不同玩家的发言中提取了部分有价值的信息，但是结论不完全正确。
得分 5：该推理过程逻辑完备，从不同玩家的发言中找出线索，正确推理出了玩家的身份和所持有的关键词。

"""
}

PROMPT_TEMPLATE = {
    "伪装能力": """
身份信息如下：
{{role_info}}

游戏的历史信息如下：
{{history}}

当前发言玩家是：{{name}}:

他的发言为：{{answer}}

---
请逐步思考，并保持解释的简短。用 "Score: <score>" 作为你回答的结束。
""",
"推理能力": """
身份信息如下：
{{role_info}}

游戏的历史信息如下：
{{history}}

当前正在思考的玩家是：{{name}}:

他的推理为：{{answer}}

---
请逐步思考，并保持解释的简短。用 "Score: <score>" 作为你回答的结束。
"""
}


class AdvanceEvaluatorConfig(MetricEvaluatorConfig):
    non_ignored_message_type: Optional[List[Type[Message]]] = Field(default=[ModeratorInitGameSummary], exclude=True)
    openEvalConfig: GeneralOpenEvalToolConfig = Field(...)


class EvalTool(GeneralOpenEvalTool):
    _role_info: dict

    def __init__(self, config: GeneralOpenEvalToolConfig):
        super().__init__(config)
        self._role_info = {}

    @property
    def role_info(self):
        return self._role_info

    @role_info.setter
    def role_info(self, value):
        self._role_info = value


class AdvanceEvaluator(
    MetricEvaluator,
    metric_definitions=SUPPORT_METRICS,
    cls_description="谁是卧底专用评估器，分析发言的伪装能力和推理的逻辑能力",
):
    config_cls = AdvanceEvaluatorConfig
    config: config_cls

    @staticmethod
    def _init_evaluator(
        config: MetricEvaluatorConfig,
        record_metrics: List[_MetricName],
        compare_metrics: List[_MetricName]
    ) -> Any:
        if isinstance(config, AdvanceEvaluatorConfig):
            open_eval_tool: GeneralOpenEvalTool = GeneralOpenEvalTool.from_config(
                config.openEvalConfig
            )
            open_eval_tool.set_activated_metrics(record_metrics)
            open_eval_tool.set_max_tokens(512)
            open_eval_tool.set_temperature(0.1)

            return open_eval_tool
        else:
            raise ValueError(f"Invalid config type {type(config)}")

    @staticmethod
    def _role_info_to_str(role_info: dict):
        result = ""
        for name, info in role_info.items():
            result += f"玩家[{name}]的身份是[{info['role']}]，关键词[{info['key']}]\n"
        return result

    @staticmethod
    def _reference_to_history_str(reference: List[Message]):
        result = ""
        for msg in reference:
            if msg.sender_name == 'moderator':
                continue
            result += f"[{msg.sender_name} -> {msg.receiver_names}]: {msg.content.display_text}\n"
        return result

    @staticmethod
    async def _record(
        response: Message, references: Optional[List[Message]], ground_truth: Optional[Text], evaluator: Any, **kwargs
    ) -> Dict[_MetricName, RecordOutput]:
        result = {}
        if isinstance(response, ModeratorInitGameSummary):
            summary = response
            role_info = {}
            for role, player_names in summary.role2players.items():
                for player_name in player_names:
                    role_info[player_name] = {
                        "role": role,
                        "key": summary.keys[role]
                    }
            evaluator.role_info = role_info

        if isinstance(response, PlayerDescription) or isinstance(response, PlayerPrediction):
            try:
                answer = response
                value_dict = {
                    "role_info": AdvanceEvaluator._role_info_to_str(evaluator.role_info),
                    "history": AdvanceEvaluator._reference_to_history_str(references),
                    "name": answer.sender_name,
                    "answer": answer.content.text
                }

                for metric in SUPPORT_METRICS:
                    if metric.belonged_chain not in evaluator.activated_metrics or not metric.belonged_chain.startswith(
                        kwargs["action_belonged_chain"]
                    ):
                        continue

                    value = await evaluator.evaluate(
                        system_template=SYSTEM_TEMPLATE_DICT[metric.name],
                        prompt_template=PROMPT_TEMPLATE[metric.name],
                        value_dict=value_dict
                    )
                    score = value.split("Score: ")[-1]
                    reason = value.split("Score: ")[0]

                    try:
                        score = int(score)
                    except:
                        score = 0

                    result[metric.belonged_chain] = RecordOutput(
                        record_value=score,
                        reason=reason,
                        misc=value_dict
                    )
            except Exception as e:
                print(f'Error when evaluating: {e}')

        return result

    @staticmethod
    async def _compare(
        response: Message, references: Optional[List[Message]], ground_truth: Optional[Text], evaluator: Any, **kwargs
    ) -> Dict[_MetricName, CompareOutput]:
        return {}


__all__ = [
    "AdvanceEvaluatorConfig",
    "AdvanceEvaluator"
]
