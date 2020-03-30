import numpy as np
import logging
import math
import random

from abstractclasses.AbstractForwardModelAgent import AbstractForwardModelAgent


class SimpleRHEA(AbstractForwardModelAgent):

    def __init__(self, rollout_actions_length, mutation_probability, num_evaluations, use_shift_buffer=True,
                 flip_at_least_one=True, discount_factor=1.0, ignore_frames=0, forward_model=None, score_model=None):
        super().__init__(forward_model, score_model)

        self._rollout_actions_length = rollout_actions_length
        self._flip_at_least_one = flip_at_least_one
        self._mutation_probability = mutation_probability
        self._num_evaluations = num_evaluations
        self._ignore_frames = ignore_frames
        self._solution = self._random_solution(range(4))
        self._use_shift_buffer = use_shift_buffer
        self.known_predictions = dict()

        self._discount_factors = []
        for i in range(self._rollout_actions_length):
            self._discount_factors .append(math.pow(discount_factor, i))

    def re_initialize(self):
        self.known_predictions = dict()
        pass

    def _shift_and_append(self, solution):
        """
        Remove the first element and add a random action on the end
        """
        new_solution = np.copy(solution[1:])
        new_solution = np.hstack([new_solution, random.choice(range(4))])
        return new_solution

    def get_next_action(self, state, actions):
        """
        Get the next best action by evaluating a bunch of mutated solutions
        """
        if self._use_shift_buffer:
            solution = self._shift_and_append(self._solution)
        else:
            solution = self._random_solution(actions)

        candidate_solutions = self._mutate(solution, actions, self._mutation_probability)

        mutated_scores = self.evaluate_rollouts(state, candidate_solutions)
        best_idx = int(np.argmax(mutated_scores, axis=0))

        best_score_in_evaluations = mutated_scores[best_idx]

        self._solution = candidate_solutions[best_idx]

        logging.info('Best score in evaluations: %.2f' % best_score_in_evaluations)

        # The next best action is the first action from the solution space
        return self._solution[0]

    def get_agent_name(self) -> str:
        return "RHEA Agent"

    def _random_solution(self, actions):
        """
        Create a random set fo actions
        """
        return np.array([random.choice(actions) for _ in range(self._rollout_actions_length)])

    def _mutate(self, solution, actions, mutation_probability):
        """
        Mutate the solution
        """

        candidate_solutions = []
        # Solution here is 2D of rollout_actions x batch_size
        for b in range(self._num_evaluations):
            # Create a set of indexes in the solution that we are going to mutate
            mutation_indexes = set()
            solution_length = len(solution)
            if self._flip_at_least_one:
                mutation_indexes.add(np.random.randint(solution_length))

            mutation_indexes = mutation_indexes.union(
                set(np.where(np.random.random([solution_length]) < mutation_probability)[0]))

            # Create the number of mutations that is the same as the number of mutation indexes
            num_mutations = len(mutation_indexes)
            mutations = [random.choice(actions) for _ in range(num_mutations)]

            # Replace values in the solutions with mutated values
            new_solution = np.copy(solution)
            new_solution[list(mutation_indexes)] = mutations
            candidate_solutions.append(new_solution)

        return np.stack(candidate_solutions)

    def evaluate_rollouts(self, state, candidate_solutions):
        scores = []
        for solution in candidate_solutions:
            scores.append(self.evaluate_rollout(state.level, solution))

        return scores

    def evaluate_rollout(self, state, action_sequence):
        discounted_return = 0
        for idx, action in enumerate(action_sequence):
            state_tag = tuple(state.flatten())
            if state_tag in self.known_predictions and self.known_predictions[state_tag][action] is not None:
                state, score = self.known_predictions[state_tag][action]
            else:
                state = self._forward_model.predict(state, action)
                score = self._score_model(state)
                if state_tag not in self.known_predictions:
                    self.known_predictions[state_tag] = [None] * 4
                self.known_predictions[state_tag][action] = (state, score)

            discounted_return += score * self._discount_factors[idx]

        return discounted_return
