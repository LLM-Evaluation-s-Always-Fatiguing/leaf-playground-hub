import asyncio
import random
from typing import List, Optional

from pydantic import Field

from leaf_playground.core.workers import Logger
from leaf_playground.core.scene import Scene
from leaf_playground.core.scene_definition import SceneConfig
from leaf_playground.data.log_body import ActionLogBody
from leaf_playground.data.media import Text

from .agents.moderator import Moderator
from .agents.player import BaseAIPlayer
from .scene_definition import *


class WhoIsTheSpyLogBody(ActionLogBody):
    references: Optional[List[MessageTypes]] = Field(default=None)
    response: MessageTypes = Field(default=...)


WhoIsTheSpySceneConfig = SceneConfig.create_config_model(
    SCENE_DEFINITION,
    additional_config_fields={
        "debug_mode": (bool, Field(default=False, exclude=True))
    }
)


class WhoIsTheSpyScene(Scene, scene_definition=SCENE_DEFINITION, log_body_class=WhoIsTheSpyLogBody):
    config_cls = WhoIsTheSpySceneConfig
    config: config_cls

    def __init__(self, config: config_cls, logger: Logger):
        super().__init__(config=config, logger=logger)

        self.moderator: Moderator = self.static_agents["moderator"][0]
        self.players: List[BaseAIPlayer] = self.agents["player"]

    async def _run(self):
        def put_message(message: MessageTypes, log_msg: str, action_belonged_chain: Optional[str] = None):
            self.message_pool.put_message(message)

            references = None
            if not message.sender_id == self.moderator.id:
                references = self.message_pool.get_messages(message.sender)[:-1]
            self.logger.add_log(
                self.log_body_class(
                    references=references,
                    response=message,
                    log_msg=log_msg,
                    action_belonged_chain=action_belonged_chain
                )
            )

        async def player_receive_key(player: BaseAIPlayer) -> None:
            history = self.message_pool.get_messages(player.profile)
            key_assignment_msg: ModeratorKeyAssignment = history[-1]
            try:
                await player.receive_key(key_assignment_msg)
            except:
                if self.config.debug_mode:
                    raise

        async def player_describe_key(player_: BaseAIPlayer) -> PlayerDescription:
            history = self.message_pool.get_messages(player_.profile)
            try:
                description = await player_.describe_key(
                    history, [self.moderator.profile]
                )
            except:
                if self.config.debug_mode:
                    raise
                description = PlayerDescription(
                    sender=player_.profile,
                    receivers=[self.moderator.profile],
                    content=Text(text="I have nothing to say.")
                )
            put_message(
                description,
                log_msg=f"{player_.name} describes key",
                action_belonged_chain=None
            )
            return description

        async def player_describe_with_validation(player_: BaseAIPlayer):
            description = await player_describe_key(player_)
            patience_ = 3
            while patience_:
                moderator_warning = self.moderator.valid_player_description(description=description)
                if not moderator_warning.has_warn:
                    break
                else:
                    put_message(
                        moderator_warning,
                        log_msg=f"{self.moderator.name} warns {player_.name} to not break key description rules.",
                        action_belonged_chain=None
                    )  # will be only seen by the player
                    description = await player_describe_key(player_)
                patience_ -= 1
            description.receivers += [p.profile for p in players if p.id != player_.id]
            put_message(
                description,
                log_msg=f"{player_.name} describes key.",
                action_belonged_chain=None
            )  # public to all other players

        async def players_describe_key(players_: List[BaseAIPlayer]):
            put_message(
                self.moderator.ask_for_key_description(),
                log_msg=f"{self.moderator.name} asks players to describe keys they get.",
                action_belonged_chain=None
            )
            await asyncio.gather(
                *[player_describe_with_validation(player_) for player_ in players_]
            )

        async def player_predict_role(player_: BaseAIPlayer) -> PlayerPrediction:
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
                log_msg=f"{player_.name} predicts other players' role.",
                action_belonged_chain=None
            )
            return prediction

        async def player_vote(player_: BaseAIPlayer) -> PlayerVote:
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
                log_msg=f"{player_.name} commits vote.",
                action_belonged_chain=None
            )
            return vote

        num_rounds = self.env_vars["num_rounds"].current_value
        while num_rounds:
            players = self.players
            random.shuffle(players)  # shuffle to randomize the speak order

            # clear information in the past round
            self.message_pool.clear()
            self.moderator.reset_inner_status()
            for player in players:
                player.reset_inner_status()

            # prepare the new game
            self.moderator.registry_players(players=[player.profile for player in players])
            put_message(self.moderator.init_game(), log_msg=f"{self.moderator.name} initialize the game.")
            put_message(self.moderator.introduce_game_rule(), log_msg=f"{self.moderator.name} introduces game rules.")
            put_message(self.moderator.announce_game_start(), log_msg=f"{self.moderator.name} announces game start.")
            # assign keys
            for player in self.players:
                key_assignment = self.moderator.assign_keys(player=player.profile)
                put_message(key_assignment, f"{self.moderator.name} assigns a key to {player.name}.")
            await asyncio.gather(
                *[player_receive_key(player) for player in players]
            )

            # run game
            while True:  # for each turn
                # 1. ask players to give a description for the key they got sequentially,
                #    then validate player's prediction
                await players_describe_key(players)

                # 3. ask players to predict who is spy or blank
                put_message(
                    self.moderator.ask_for_role_prediction(),
                    log_msg=f"{self.moderator.name} asks players to predict others' role."
                )
                predictions = await asyncio.gather(*[player_predict_role(player) for player in players])

                # 4. summarize player predictions
                put_message(
                    self.moderator.summarize_players_prediction(predictions=list(predictions)),
                    log_msg=f"{self.moderator.name} summarizes players predictions."
                )

                patience = 3
                most_voted_players = None
                while patience:
                    # 5. ask players to vote
                    put_message(
                        self.moderator.ask_for_vote(),
                        log_msg=f"{self.moderator.name} asks players to vote."
                    )
                    votes = list(await asyncio.gather(*[player_vote(player) for player in players]))
                    # 6. summarize player votes, if there is a tie, ask most voted players to re-describe key
                    vote_summarization = self.moderator.summarize_player_votes(
                        votes=votes, focused_players=most_voted_players
                    )
                    put_message(
                        vote_summarization,
                        log_msg=f"{self.moderator.name} summarizes players' votes."
                    )
                    if not vote_summarization.tied_players:
                        break
                    most_voted_players = vote_summarization.tied_players
                    # 7. most voted players re-describe
                    await players_describe_key([player for player in players if player.profile in most_voted_players])

                # 8. check is game over and announce winners
                game_over_summary = self.moderator.check_if_game_over()
                put_message(
                    game_over_summary,
                    log_msg=f"{self.moderator.name} checks if this round of game is finished."
                )
                if game_over_summary.is_game_over:
                    break
                # 9. exclude eliminated players
                players = [
                    player for player in players if self.moderator.id2status[player.id] == PlayerStatus.ALIVE
                ]

                # TODO: more things to do?

            num_rounds -= 1


__all__ = [
    "WhoIsTheSpySceneConfig",
    "WhoIsTheSpyScene"
]
