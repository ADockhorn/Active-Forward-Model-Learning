import random

from abstractclasses.AbstractForwardModelAgent import AbstractForwardModelAgent


class SimpleBFS(AbstractForwardModelAgent):

    def __init__(self, expansions, discount_factor=1.0, forward_model=None, score_model=None, terminal_model=None):
        super().__init__(forward_model, score_model)

        self._terminal_model = terminal_model
        self._expansions = expansions
        self._discount_factor = discount_factor
        self.known_states = dict()
        self._exploration_penalty = 0.1
        self.expansion_list = []

    def re_initialize(self):
        self.known_states = dict()

    def get_identifier(self, level):
        return tuple(level.flatten())

    def get_next_action(self, state, actions):
        state_tag = self.get_identifier(state.level)
        if state_tag not in self.known_states:
            self.expansion_list.append((state.level, state_tag))

        # add some expansions
        for i in range(self._expansions):
            if len(self.expansion_list) == 0:
                # all predictable states are already known
                break

            parent_level, parent_tag = self.expansion_list.pop(0)
            self.known_states[parent_tag] = [None]*4
            for action in range(4):
                child_level = self._forward_model.predict(parent_level, action)
                child_tag = self.get_identifier(child_level)
                score = self._score_model(state)
                self.known_states[parent_tag][action] = (child_level, child_tag, score)
                if child_tag not in self.known_states:
                    self.expansion_list.append((child_level, child_tag))

        # find best action_sequence using BFS starting from the current state
        checked_states = set()
        best_state = [None, -1]
        open_states = [(state_tag, ())]
        while len(open_states) > 0:
            state_tag, action_sequence = open_states.pop(0)
            checked_states.add(state_tag)
            if state_tag not in self.known_states:
                continue

            for action in range(4):
                child_level, child_tag, score = self.known_states[state_tag][action]
                if child_tag not in checked_states:
                    child_action_sequence = (*action_sequence, action)
                    open_states.append((child_tag, child_action_sequence))
                    if score > best_state[1]:
                        best_state = [child_action_sequence, score]
        if best_state[1] > -1:
            return best_state[0][0]
        # The next best action is the first action from the solution space
        else:
            return random.choice(range(4))

    def get_solution(self, state, max_length=100):
        known_states = dict()
        expansion_list = []

        state_tag = self.get_identifier(state.level)
        if state_tag not in known_states:
            expansion_list.append((state.level, state_tag, ()))

        for i in range(self._expansions):
            if len(expansion_list) == 0:
                return None

            parent_state, parent_tag, parent_actions = expansion_list.pop(0)
            if parent_tag in known_states:
                continue

            known_states[parent_tag] = [None, None, None, None]

            for action in range(4):
                child_state = self._forward_model.predict(parent_state, action)
                child_tag = self.get_identifier(child_state)
                child_action_sequence = (*parent_actions, action)

                if self._terminal_model(child_state):
                    yield child_action_sequence
                if len(child_action_sequence) > max_length:
                    break

                known_states[parent_tag][action] = (child_state, child_tag)
                if child_tag not in known_states:
                    expansion_list.append((child_state, child_tag, child_action_sequence))
        return None

    def get_agent_name(self) -> str:
        return "BFS Agent"
