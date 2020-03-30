from activestateexploration.stateselectionstrategies.abstractactivestateselectionagent import CertaintyBasedStateSelection
import random
import numpy as np


class UncertaintyBasedStateSelectionAgent(CertaintyBasedStateSelection):

    def __init__(self, mask, span, consecutive_level_exploration, uncertainty_measure, aggregation_function=np.max):
        super().__init__(mask, span, consecutive_level_exploration)
        self.uncertainty_measure = uncertainty_measure
        self.aggregation_function = aggregation_function

    def select_state_and_action(self):
        if len(self.data_set) == 0:
            possible_extensions = [(parent, action) for parent in self.known_states for action in range(4)
                                   if self.known_states[parent]["child_states"][action] is None]
            if len(possible_extensions) == 0:
                return None, None, 0
            random_state, random_action = random.choice(possible_extensions)
            return random_state, random_action, 0

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

    def rate_state_action_pair(self, state, action):
        if len(self.data_set) == 0 or not self.model_ready:
            return 0
        try:
            pattern_data = np.array([x for x in self.known_states[state]["patterns"][action]])
            if len(pattern_data) == 0:
                return 0

            class_prob = self.proba_model.predict_proba(pattern_data)
            return self.aggregation_function(self.uncertainty_measure(class_prob))
        except Exception:
            print("crashed")
            pass

        return 0
