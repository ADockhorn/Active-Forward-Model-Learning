from activestateexploration.stateselectionstrategies.abstractactivestateselectionagent import CertaintyBasedStateSelection


class MaxUnknownPatternsStateActionSelection(CertaintyBasedStateSelection):

    def __init__(self, mask, span, consecutive_level_exploration):
        super().__init__(mask, span, consecutive_level_exploration)

    def rate_state_action_pair(self, state, action):
        return len(self.known_states[state]["patterns"][action])
