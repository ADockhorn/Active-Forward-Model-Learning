from abc import ABC, abstractmethod

from activestateexploration.stateselectionstrategies.abstractactivestateselectionagent import ActiveStateSelectionAgent
from sokoban.fast_sokoban import add_patterns_to_data_set
import random
import pickle


class QueuingStateActionSelection(ActiveStateSelectionAgent, ABC):
    def __init__(self, mask, span, consecutive_level_exploration):
        super().__init__(mask, span)
        self.state_action_queue = []
        self.known_states = set()
        self.consecutive_level_exploration = consecutive_level_exploration

    def train(self, env, levels, steps_per_level, checkpoint_ticks, target_file):
        if self.consecutive_level_exploration:
            return self.consecutive_level_exploration_training(env, levels, steps_per_level, checkpoint_ticks, target_file)
        else:
            return self.combined_level_training(env, levels, checkpoint_ticks, target_file)

    def consecutive_level_exploration_training(self, env, levels, steps_per_level, checkpoint_ticks, target_file):
        history = {"known_patterns": [0],
                   "model_checkpoints": []}
        total_ticks = 0

        for level in levels:
            game = env(level)
            self.known_states = set()
            self.state_action_queue = []
            self.add_state(game)

            for i in range(1, 1 + steps_per_level):
                total_ticks += 1
                if len(self.state_action_queue) == 0:
                    break
                else:
                    state, next_action = self.select_state_and_action()

                state = state.get_copy()
                prev_game_state = state.level.copy()
                state.next(next_action)
                add_patterns_to_data_set(prev_game_state, next_action, state.level,
                                         self.mask, self.span, self.data_set)
                self.add_state(state)
                history["known_patterns"].append(len(self.data_set))

                if total_ticks in checkpoint_ticks:
                    history["model_checkpoints"].append(self.get_forward_model())

        return history

    def combined_level_training(self, env, levels, checkpoint_ticks, target_file):
        history = {"known_patterns": [0],
                   "model_checkpoints": []}
        self.known_states = set()
        self.state_action_queue = []

        for level in levels:
            game = env(level)
            self.add_state(game)

        max_steps = max(checkpoint_ticks)
        for i in range(1, 1+max_steps):
            if len(self.state_action_queue) == 0:
                break
            else:
                state, next_action = self.select_state_and_action()

            state = state.get_copy()
            prev_game_state = state.level.copy()
            state.next(next_action)
            add_patterns_to_data_set(prev_game_state, next_action, state.level,
                                     self.mask, self.span, self.data_set)
            self.add_state(state)
            history["known_patterns"].append(len(self.data_set))

            if i in checkpoint_ticks:
                history["model_checkpoints"].append(self.get_forward_model())
                with open(target_file, "wb") as file:
                    pickle.dump(history, file)
            if i == max_steps:
                break

        return history

    @abstractmethod
    def select_state_and_action(self):
        pass

    def add_state(self, game_state, parent_state_tag=None, previous_action=None):
        # for each state track the game_state, observable patterns that have not been observed yet,
        # and a list of child_states

        level = game_state.level
        level_tag = tuple(level.flatten())

        # never process a state twice (this would be signal, that there is nothing new to be found)
        if level_tag in self.known_states:
            return level_tag

        self.known_states.add(level_tag)
        self.state_action_queue.extend([(game_state, action) for action in range(4)])


class BreadthFirstSearchStateActionSelection(QueuingStateActionSelection):
    def __init__(self, mask, span, consecutive_level_exploration):
        super().__init__(mask, span, consecutive_level_exploration)

    def select_state_and_action(self):
        return self.state_action_queue.pop(0)


class RandomStateActionSelection(QueuingStateActionSelection):
    def __init__(self, mask, span, consecutive_level_exploration):
        super().__init__(mask, span, consecutive_level_exploration)

    def select_state_and_action(self):
        return self.state_action_queue.pop(random.choice(range(len(self.state_action_queue))))
