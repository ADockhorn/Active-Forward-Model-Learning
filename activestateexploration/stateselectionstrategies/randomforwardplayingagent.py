import random

from activestateexploration.stateselectionstrategies.abstractactivestateselectionagent import ActiveStateSelectionAgent
from sokoban.fast_sokoban import add_patterns_to_data_set, transform_data_set
import pickle


class RandomForwardPlayingAgent(ActiveStateSelectionAgent):
    def __init__(self, mask, span):
        super().__init__(mask, span)

    def train(self, env, levels, steps_per_level, checkpoint_ticks, target_file, reset_after_x_ticks=100):
        history = {"known_patterns_per_timestep": [0],
                   "model_checkpoints": []}
        total_ticks = 0
        final_tick = max(checkpoint_ticks)

        while total_ticks < final_tick:
            for level in levels:
                game = env(level)
                prev_game_state = game.level.copy()
                for i in range(1, 1+steps_per_level):
                    total_ticks += 1
                    next_action = random.choice([0, 1, 2, 3])
                    game.next(next_action)
                    add_patterns_to_data_set(prev_game_state, next_action, game.level,
                                             self.mask, self.span, self.data_set)

                    if i % reset_after_x_ticks == 0:
                        game = env(level)
                    prev_game_state = game.level.copy()
                    history["known_patterns_per_timestep"].append(len(self.data_set))

                    if total_ticks in checkpoint_ticks:
                        history["model_checkpoints"].append(self.get_forward_model())
                        with open(target_file, "wb") as file:
                            pickle.dump(history, file)
                    if total_ticks == final_tick:
                        break
        return history

    def get_data_set(self):
        return transform_data_set(self.data_set, "argmax")
