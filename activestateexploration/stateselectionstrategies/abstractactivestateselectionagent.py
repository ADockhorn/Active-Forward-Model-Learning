from abc import ABC, abstractmethod
from sokoban.fast_sokoban import transform_data_set
from sklearn.tree import DecisionTreeClassifier
from sokoban.fast_sokoban import extract_all_patterns_without_action, add_patterns_to_data_set
import random
import tqdm
import pickle
from sklearn.ensemble import ExtraTreesClassifier


class ActiveStateSelectionAgent(ABC):
    def __init__(self, mask, span):
        super().__init__()
        self.mask = mask
        self.span = span
        self.data_set = dict()

    @abstractmethod
    def train(self, env, levels, steps_per_level, checkpoints_every_x_ticks, target_file):
        pass

    def get_data_set(self):
        return transform_data_set(self.data_set, "argmax")

    def get_forward_model(self):
        model = DecisionTreeClassifier()
        data = self.get_data_set()
        model.fit(data[:, :-1], data[:, -1])
        return model


class CertaintyBasedStateSelection(ActiveStateSelectionAgent):
    def __init__(self, mask, span, consecutive_level_exploration):
        super().__init__(mask, span)
        self.known_states = dict()      # stores which patterns occur per state
        self.known_patterns = dict()    # stores in which state each pattern patterns occurs
        self.consecutive_level_exploration = consecutive_level_exploration
        self.proba_model = ExtraTreesClassifier()
        self.model_ready = False

        pass

    def add_state(self, game_state, parent_state_tag=None, previous_action=None):
        # for each state track the game_state, observable patterns that have not been observed yet,
        # and a list of child_states

        level = game_state.level
        level_tag = tuple(level.flatten())

        if parent_state_tag is not None:
            self.known_states[parent_state_tag]["child_states"][previous_action] = level_tag

        # never process a state twice (this would be signal, that there is nothing new to be found)
        if level_tag in self.known_states:
            return level_tag

        self.known_states[level_tag] = {"state": game_state.get_copy(),
                                        "patterns": {x: set() for x in range(4)},
                                        "child_states": [None] * 4}

        # add patterns -> state links
        patterns = extract_all_patterns_without_action(level, self.mask, self.span)
        for action in [0, 1, 2, 3]:
            for pattern in patterns:
                pattern = (*pattern, action)
                if pattern in self.known_patterns:
                    # only add the link in case the pattern is still unknown to the learner
                    if not self.known_patterns[pattern]["observed"]:
                        self.known_patterns[pattern]["states"].add(level_tag)
                else:
                    self.known_patterns[pattern] = {"observed": False, "states": {level_tag}}

                # add state -> pattern links
                # only add the pattern in case the pattern was unobserved and we have not already looked at this state
                if self.known_states[level_tag]["child_states"][action] is None:
                    if not self.known_patterns[pattern]["observed"]:
                        self.known_states[level_tag]["patterns"][action].add(pattern)
                else:
                    raise ValueError("the level_tag's child states should be None since the state was just added")
        return level_tag

    def add_child_state(self, parent_tag, action, child_tag):
        self.known_states[parent_tag]["child_states"][action] = child_tag

    def remove_observed_patterns(self, parent_state_tag, next_action):
        patterns_to_be_removed = self.known_states[parent_state_tag]["patterns"][next_action].copy()
        for pattern in patterns_to_be_removed:
            if pattern in self.known_patterns:
                self.known_patterns[pattern]["observed"] = True
                for state in self.known_patterns[pattern]["states"]:
                    self.known_states[state]["patterns"][next_action].discard(pattern)
                self.known_patterns[pattern]["states"] = set()
            else:
                raise ValueError("I fucked up")
        return

    def reset_patterns(self):
        for pattern in self.known_patterns:
            self.known_patterns[pattern]["states"] = set()
        return

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
            self.known_states = dict()
            self.reset_patterns()
            self.add_state(game)

            for i in range(1, 1+steps_per_level):
                total_ticks += 1
                parent_state_tag, next_action, value = self.select_state_and_action()
                if next_action is None:
                    break
                state = self.known_states[parent_state_tag]["state"].get_copy()
                prev_game_state = state.level.copy()
                state.next(next_action)
                add_patterns_to_data_set(prev_game_state, next_action, state.level,
                                         self.mask, self.span, self.data_set)
                self.remove_observed_patterns(parent_state_tag, next_action)
                self.add_state(state, parent_state_tag, previous_action=next_action)
                history["known_patterns"].append(len(self.data_set))

                if total_ticks in checkpoint_ticks:
                    history["model_checkpoints"].append(self.get_forward_model())
                    with open(target_file, "wb") as file:
                        pickle.dump(history, file)

        return history

    def combined_level_training(self, env, levels, checkpoint_ticks, target_file):
        history = {"known_patterns": [0],
                   "model_checkpoints": []}

        self.known_states = dict()
        self.reset_patterns()
        for level in levels:
            game = env(level)
            self.add_state(game)

        max_ticks = max(checkpoint_ticks)

        for i in tqdm.trange(1, 1+max_ticks, desc="perform active selection"):
            parent_state_tag, next_action, value = self.select_state_and_action()
            if next_action is None:
                break
            state = self.known_states[parent_state_tag]["state"].get_copy()
            prev_game_state = state.level.copy()
            state.next(next_action)
            add_patterns_to_data_set(prev_game_state, next_action, state.level,
                                     self.mask, self.span, self.data_set)
            self.remove_observed_patterns(parent_state_tag, next_action)
            self.add_state(state, parent_state_tag, previous_action=next_action)
            history["known_patterns"].append(len(self.data_set))

            if i in checkpoint_ticks:
                data = self.get_data_set()
                self.proba_model.fit(data[:, :-1], data[:, -1])
                self.model_ready = True

                history["model_checkpoints"].append(self.get_forward_model())
                with open(target_file, "wb") as file:
                    pickle.dump(history, file)
        return history

    def select_state_and_action(self):
        best_state, best_action, best_value = None, None, -1
        for state in self.known_states:
            for action in range(4):
                value = self.rate_state_action_pair(state, action)
                if value > best_value:
                    best_state = state
                    best_action = action
                    best_value = value
        if best_value == 0:
            possible_extensions = [(parent, action) for parent in self.known_states for action in range(4)
                                   if self.known_states[parent]["child_states"][action] is None]
            if len(possible_extensions) == 0:
                return None, None, 0
            random_state, random_action = random.choice(possible_extensions)
            return random_state, random_action, 0
        return best_state, best_action, best_value

    @abstractmethod
    def rate_state_action_pair(self, state, action):
        pass
