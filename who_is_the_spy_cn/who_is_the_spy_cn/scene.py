import asyncio
import random
from typing import List, Optional, Type, Union

from pydantic import Field

from leaf_playground.core.workers import Logger
from leaf_playground.core.scene import Scene
from leaf_playground.core.scene_definition import SceneConfig
from leaf_playground.data.log_body import ActionLogBody
from leaf_playground.data.media import Text

from .agents.moderator import Moderator
from .agents.player import BaseAIPlayer
from .agents.human_player import HumanPlayer
from .scene_definition import *

Player = Union[BaseAIPlayer, HumanPlayer]


class WhoIsTheSpyLogBody(ActionLogBody):
    references: Optional[List[MessageTypes]] = Field(default=None)
    response: MessageTypes = Field(default=...)
    game_id: int = Field(default=...)
    round_id: int = Field(default=...)


WhoIsTheSpySceneConfig = SceneConfig.create_config_model(
    SCENE_DEFINITION,
    additional_config_fields={
        "debug_mode": (bool, Field(default=False, exclude=True))
    }
)


class WhoIsTheSpyScene(Scene, scene_definition=SCENE_DEFINITION, log_body_class=WhoIsTheSpyLogBody):
    config_cls = WhoIsTheSpySceneConfig
    config: config_cls

    log_body_class: Type[WhoIsTheSpyLogBody]

    def __init__(self, config: config_cls, logger: Logger):
        super().__init__(config=config, logger=logger)

        self.moderator: Moderator = self.static_agents["moderator"][0]
        self.players: List[Player] = self.agents["player"]

    async def _run(self):
        def put_message(message: MessageTypes, log_msg: str, action_belonged_chain: Optional[str] = None):
            references = None
            if not message.sender_id == self.moderator.id:
                references = self.message_pool.get_messages(message.sender)
            self.message_pool.put_message(message)
            log = self.log_body_class(
                references=references,
                response=message,
                log_msg=log_msg,
                action_belonged_chain=action_belonged_chain,
                game_id=game_id,
                round_id=round_id
            )
            self.logger.add_log(log)
            self.notify_evaluators_record(log)

        async def player_receive_key(player: Player) -> None:
            history = self.message_pool.get_messages(player.profile)
            key_assignment_msg: ModeratorKeyAssignment = history[-1]
            try:
                await player.receive_key(key_assignment_msg)
            except:
                if self.config.debug_mode:
                    raise

        async def player_describe_key(player_: Player):
            history = self.message_pool.get_messages(player_.profile)
            try:
                description = await player_.describe_key(
                    history, [self.moderator.profile] + [p.profile for p in self.players]
                )
            except:
                if self.config.debug_mode:
                    raise
                description = PlayerDescription(
                    sender=player_.profile,
                    receivers=[self.moderator.profile] + [p.profile for p in self.players],
                    content=Text(text="我无话可说。")
                )
            put_message(
                message=description,
                log_msg=f"{player_.name} 将对自己获得的信息的描述发送给所有游戏参与者",
                action_belonged_chain=player_.role_definition.get_action_definition("describe_key").belonged_chain
            )

        async def players_describe_key(players_: List[Player]):
            put_message(
                await self.moderator.ask_for_key_description(),
                log_msg=f"{self.moderator.name} 要求玩家依次对获得的信息进行描述",
                action_belonged_chain=self.moderator.role_definition.get_action_definition(
                    "ask_for_key_description"
                ).belonged_chain
            )
            for player_ in players_:
                await player_describe_key(player_)

        async def player_predict_role(player_: Player) -> PlayerPrediction:
            history = self.message_pool.get_messages(player_.profile)
            try:
                prediction = await player_.predict_role(history, self.moderator.profile)
            except:
                if self.config.debug_mode:
                    raise
                prediction = PlayerPrediction(
                    sender=player_.profile,
                    receivers=[self.moderator.profile],
                    content=Text(text="")
                )
            put_message(
                prediction,
                log_msg=f"{player_.name} 预测其他玩家的身份",
                action_belonged_chain=player_.role_definition.get_action_definition("predict_role").belonged_chain
            )
            return prediction

        async def players_predict_role(players_: List[Player]):
            # ask players to predict who is spy or blank
            put_message(
                await self.moderator.ask_for_role_prediction(),
                log_msg=f"{self.moderator.name} 要求各玩家预测其他玩家的身份",
                action_belonged_chain=self.moderator.role_definition.get_action_definition(
                    "ask_for_role_prediction"
                ).belonged_chain
            )
            predictions = await asyncio.gather(*[player_predict_role(player_) for player_ in players_])

            # summarize player predictions
            put_message(
                await self.moderator.summarize_players_prediction(predictions=list(predictions)),
                log_msg=f"{self.moderator.name} 总结各玩家的预测结果",
                action_belonged_chain=self.moderator.role_definition.get_action_definition(
                    "summarize_players_prediction"
                ).belonged_chain
            )

        async def player_vote(player_: Player) -> PlayerVote:
            history = self.message_pool.get_messages(player_.profile)
            try:
                vote = await player_.vote(history, self.moderator.profile)
            except:
                if self.config.debug_mode:
                    raise
                vote = PlayerVote(
                    sender=player_.profile,
                    receivers=[self.moderator.profile],
                    content=Text(text="")
                )
            put_message(
                vote,
                log_msg=f"{player_.name} 提交其投票。",
                action_belonged_chain=player_.role_definition.get_action_definition("vote").belonged_chain
            )
            return vote

        async def players_vote(players_: List[Player]):
            patience = 3
            most_voted_players = None
            while patience:
                patience -= 1
                # ask players to vote
                put_message(
                    await self.moderator.ask_for_vote(targets=most_voted_players or [p.profile for p in players_]),
                    log_msg=f"{self.moderator.name} 要求各玩家进行投票",
                    action_belonged_chain=self.moderator.role_definition.get_action_definition(
                        "ask_for_vote"
                    ).belonged_chain
                )
                votes = list(await asyncio.gather(*[player_vote(player) for player in players]))
                # summarize player votes, if there is a tie, ask most voted players to re-describe key
                vote_summarization = await self.moderator.summarize_player_votes(
                    votes=votes, patience=patience, focused_players=most_voted_players
                )
                put_message(
                    vote_summarization,
                    log_msg=f"{self.moderator.name} 总结各玩家的投票和宣布被淘汰者",
                    action_belonged_chain=self.moderator.role_definition.get_action_definition(
                        "summarize_player_votes"
                    ).belonged_chain
                )
                if not vote_summarization.tied_players:
                    break
                most_voted_players = vote_summarization.tied_players
                # most voted players re-describe
                await players_describe_key([player for player in players if player.profile in most_voted_players])

        num_games = self.env_vars["num_games"].current_value
        game_id = 0
        while num_games:
            round_id = 0

            players = self.players

            # clear information in the past round
            self.message_pool.clear()
            await self.moderator.reset_inner_status()
            await asyncio.gather(*[p.reset_inner_status() for p in players])

            # prepare the new game
            await self.moderator.registry_players(players=[player.profile for player in players])
            game_init_summary = await self.moderator.init_game()
            put_message(
                game_init_summary,
                log_msg=f"{self.moderator.name} 准备新一局的游戏资源。",
                action_belonged_chain=self.moderator.role_definition.get_action_definition("init_game").belonged_chain
            )
            # randomize players' speak order
            if game_init_summary.role2players.get(PlayerRoles.BLANK.value, None):
                blank_players_name = game_init_summary.role2players[PlayerRoles.BLANK.value]
                blank_players = [p for p in players if p.name in blank_players_name]
                non_blank_players = [p for p in players if p.name not in blank_players_name]
                if len(blank_players) >= len(players) // 2:
                    random.shuffle(non_blank_players)
                    players = non_blank_players + blank_players
                else:
                    blank_players_pos = random.choices(
                        list(range(len(players) // 2, len(players))), k=len(blank_players)
                    )
                    random.shuffle(blank_players_pos)
                    non_blank_players_pos = [i for i in range(len(players)) if i not in blank_players_pos]
                    random.shuffle(non_blank_players_pos)
                    players_pos = (
                        list(zip(non_blank_players, non_blank_players_pos)) +
                        list(zip(blank_players, blank_players_pos))
                    )
                    players = [each[0] for each in sorted(players_pos, key=lambda x: x[1])]
            else:
                random.shuffle(players)
            put_message(
                await self.moderator.introduce_game_rule(),
                log_msg=f"{self.moderator.name} 介绍游戏规则。",
                action_belonged_chain=self.moderator.role_definition.get_action_definition(
                    "introduce_game_rule"
                ).belonged_chain
            )
            put_message(
                await self.moderator.announce_game_start(),
                log_msg=f"{self.moderator.name} 宣布游戏开始。",
                action_belonged_chain=self.moderator.role_definition.get_action_definition(
                    "announce_game_start"
                ).belonged_chain
            )

            # assign keys
            async def assign_key(player_):
                key_assignment = await self.moderator.assign_keys(player=player_.profile)
                put_message(
                    key_assignment,
                    log_msg=f"{self.moderator.name} 发送给 {player_.name} 与其身份相对应的关键词。",
                    action_belonged_chain=self.moderator.role_definition.get_action_definition(
                        "assign_keys"
                    ).belonged_chain
                )

            await asyncio.gather(*[assign_key(player) for player in self.players])
            await asyncio.gather(
                *[player_receive_key(player) for player in players]
            )

            # run game
            while True:  # for each round
                round_id += 1
                # 1. ask players to give a description for the key they got sequentially,
                #    then validate player's prediction
                await players_describe_key(players)

                # 2. ask players to predict other players' roles
                await players_predict_role(players)

                # 3. ask players to vote
                await players_vote(players)

                # 4. check is game over and announce winners
                game_over_summary = await self.moderator.check_if_game_over()
                put_message(
                    game_over_summary,
                    log_msg=f"{self.moderator.name} 确认本局游戏是否结束。",
                    action_belonged_chain=self.moderator.role_definition.get_action_definition(
                        "check_if_game_over"
                    ).belonged_chain
                )
                if game_over_summary.is_game_over:
                    break
                # 5. if game not end, exclude eliminated players
                players = [
                    player for player in players if self.moderator.id2status[player.id] == PlayerStatus.ALIVE
                ]

            num_games -= 1
            game_id += 1


__all__ = [
    "WhoIsTheSpySceneConfig",
    "WhoIsTheSpyScene"
]
