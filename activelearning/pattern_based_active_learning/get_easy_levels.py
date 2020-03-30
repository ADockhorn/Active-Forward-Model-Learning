from math import inf


class TrueBFS:

    def __init__(self, max_expansions=inf, max_sequence_length=inf):
        self.max_expansions = max_expansions
        self.max_sequence_length = max_sequence_length
        pass

    def get_identifier(self, level):
        return tuple(level.flatten())

    def get_action_sequence(self, state):
        known_states = dict()
        expansion_list = []

        state_tag = self.get_identifier(state.level)
        if state_tag not in known_states:
            expansion_list.append((state, state_tag, ()))

        iteration = 0
        # add some expansions
        while True:
            if iteration >= self.max_expansions:
                break
            if len(expansion_list) == 0:
                # all predictable states are already known
                break

            parent_state, parent_tag, parent_actions = expansion_list.pop(0)
            if parent_tag in known_states:
                continue

            known_states[parent_tag] = [None, None, None, None]

            for action in range(4):
                child_state = parent_state.get_copy()
                child_state.next(action)
                child_tag = self.get_identifier(child_state.level)
                child_action_sequence = (*parent_actions, action)
                if terminal_model(child_state.level):
                    return child_action_sequence, iteration
                if len(child_action_sequence) > self.max_sequence_length:
                    break

                known_states[parent_tag][action] = (child_state, child_tag)
                if child_tag not in known_states:
                    expansion_list.append((child_state, child_tag, child_action_sequence))
        return None, iteration

    def get_agent_name(self) -> str:
        return "True BFS Agent"


import os
from sokoban.fast_sokoban import Sokoban
import numpy as np


if __name__ == "__main__":
    max_expansions = 2500
    max_sequence_length = 25

    simple_levels = set()

    terminal_model = lambda level: np.sum(level == 3) == 0 and np.sum(level == 4) == 0

    for file in os.listdir("sokoban\\random_levels\\additional_levels"):
        game = Sokoban(f"sokoban\\random_levels\\additional_levels\\{file}")
        agent = TrueBFS(max_expansions=max_expansions, max_sequence_length=max_sequence_length)
        action_sequence, _ = agent.get_action_sequence(game)
        if action_sequence:
            simple_levels.add(file)
            print(file)
        if len(simple_levels) > 100:
            break

    for file in simple_levels:
        os.rename(f"sokoban\\random_levels\\additional_levels\\{file}",
                  f"sokoban\\random_levels\\easy_levels\\{file}")


