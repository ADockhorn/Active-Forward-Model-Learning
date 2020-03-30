from activestateexploration.stateselectionstrategies.queuingstateactionselection import RandomStateActionSelection
from activestateexploration.stateselectionstrategies.randomforwardplayingagent import RandomForwardPlayingAgent
from activestateexploration.stateselectionstrategies.activelearningstateactionselection import UncertaintyBasedStateSelectionAgent
from activestateexploration.stateselectionstrategies.maxunknownpatternagent import *
import pickle
import os

import numpy as np
from scipy.stats import entropy


def margin(pred_proba):
    part = np.partition(-pred_proba, 1, axis=1)
    # report 1-margin to select the state with the smallest margin as the most interesting one
    return - part[:, 0] + part[:, 1]


def entropy_sampling(pred_proba):
    return entropy(np.transpose(pred_proba))


if __name__ == "__main__":
    from sokoban.fast_sokoban import Sokoban
    from abstractclasses.AbstractNeighborhoodPattern import CrossNeighborhoodPattern

    span = 2
    mask = CrossNeighborhoodPattern(span).get_mask()

    uncertainty = lambda pred_proba: 1 - np.max(pred_proba, axis=1)

    agents = {
        "Random": RandomForwardPlayingAgent(mask, span),
        "Max_Unknown_Patterns": MaxUnknownPatternsStateActionSelection(mask, span, consecutive_level_exploration=False),
        "Random_State_Selection": RandomStateActionSelection(mask, span, consecutive_level_exploration=False),
        "Uncertainty_Sampling": UncertaintyBasedStateSelectionAgent(mask, span, consecutive_level_exploration=False,
                                                                    uncertainty_measure=uncertainty,
                                                                    aggregation_function=np.sum),
        "Margin_Sampling": UncertaintyBasedStateSelectionAgent(mask, span, consecutive_level_exploration=False,
                                                               uncertainty_measure=margin,
                                                               aggregation_function=np.sum),
        "Entropy_Sampling": UncertaintyBasedStateSelectionAgent(mask, span, consecutive_level_exploration=False,
                                                                uncertainty_measure=entropy_sampling,
                                                                aggregation_function=np.sum),
        "Uncertainty_Sampling_min": UncertaintyBasedStateSelectionAgent(mask, span, consecutive_level_exploration=False,
                                                                    uncertainty_measure=uncertainty,
                                                                    aggregation_function=np.min),
        "Margin_Sampling_min": UncertaintyBasedStateSelectionAgent(mask, span, consecutive_level_exploration=False,
                                                               uncertainty_measure=margin,
                                                               aggregation_function=np.min),
        "Entropy_Sampling_min": UncertaintyBasedStateSelectionAgent(mask, span, consecutive_level_exploration=False,
                                                                uncertainty_measure=entropy_sampling,
                                                                aggregation_function=np.min),
        "Uncertainty_Sampling_max": UncertaintyBasedStateSelectionAgent(mask, span, consecutive_level_exploration=False,
                                                                    uncertainty_measure=uncertainty,
                                                                    aggregation_function=np.max),
        "Margin_Sampling_max": UncertaintyBasedStateSelectionAgent(mask, span, consecutive_level_exploration=False,
                                                               uncertainty_measure=margin,
                                                               aggregation_function=np.max),
        "Entropy_Sampling_max": UncertaintyBasedStateSelectionAgent(mask, span, consecutive_level_exploration=False,
                                                                uncertainty_measure=entropy_sampling,
                                                                aggregation_function=np.max)
    }

    agent_results = dict()
    reevaluate_accuracy = False

    training_levels = [f"sokoban\\random_levels\\training_levels\\{file}" for file in os.listdir("sokoban\\random_levels\\training_levels")]

    import os
    for agent_name, agent in agents.items():

        if os.path.exists(
                f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}_lock.txt"):
            continue
        else:
            with open(f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}_lock.txt",
                      "wb") as file:
                pickle.dump(" ", file)
            print(agent_name)
            evaluation_steps = [x * 5 for x in range(1, 10)] + [x * 10 for x in range(5, 10)] + \
                               [x * 100 for x in range(1, 26)]# + \
                               #[x * 1000 for x in range(1, 10)] + [x * 10000 for x in range(1, 11)]
            agent_results = agent.train(Sokoban, training_levels, steps_per_level=100, checkpoint_ticks=set(evaluation_steps),
                                        target_file=f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}.txt")
            agent_results["evaluation_steps"] = evaluation_steps
            print(agent_results)
            with open(f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}_final_result.txt",
                      "wb") as file:
                pickle.dump(agent_results, file)