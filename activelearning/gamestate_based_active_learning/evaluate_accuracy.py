import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from tqdm import trange, tqdm
from sklearn.metrics import accuracy_score
from activestateexploration.simple_lfm import MinimalLocalForwardModel
from abstractclasses.AbstractNeighborhoodPattern import CrossNeighborhoodPattern


disable_tqdm = False

def test_pattern_accuracy(results, test_data):
    available_models = results["model_checkpoints"]
    if "pattern-based-accuracy" in results:
        accuracy = results["pattern-based-accuracy"]
    else:
        accuracy = []

    pbar = trange(len(available_models), desc=f"pattern_accuracy_{agent_name}")
    pbar.update(len(accuracy))

    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    for model_idx in range(len(accuracy), len(results["model_checkpoints"])):
        model = results["model_checkpoints"][model_idx]
        accuracy.append(accuracy_score(y_test, model.predict(x_test)))
        pbar.update(1)
    pbar.close()
    results["pattern-based-accuracy"] = accuracy


def test_state_accuracy(results, test_data, mask, span):
    if "model_checkpoints" not in results:
        return

    x_state, x_action, y_test = test_data[:, :100], test_data[:, 100], test_data[:, 101:]

    available_models = results["model_checkpoints"]
    if "state-based-accuracy" in results:
        accuracy = results["state-based-accuracy"]
    else:
        accuracy = []

    pbar = trange(len(available_models), desc=f"state_accuracy {agent_name}")
    pbar.update(len(accuracy))

    for model_idx in range(len(accuracy), len(results["model_checkpoints"])):
        model = MinimalLocalForwardModel(results["model_checkpoints"][model_idx], mask, span, remember_predictions=True)

        correct = 0
        #pbar = trange(x_state.shape[0], desc="measure game-state-accuracy")
        for prev_gamestate, action, result_state in zip(x_state, x_action, y_test):
            #pbar.update(1)
            pred = model.predict(prev_gamestate.reshape((10, 10)), action)
            if np.all(pred == result_state.reshape((10, 10))):
                correct += 1
        #pbar.close()
        accuracy.append(correct/len(x_state))
        pbar.update(1)
    results["state-based-accuracy"] = accuracy



if __name__ == "__main__":
    agents = [
        "Random",
        "Max_Unknown_Patterns",
        "Random_State_Selection",
        "Uncertainty_Sampling",
        "Margin_Sampling",
        "Entropy_Sampling",
        "Uncertainty_Sampling_min",
        "Margin_Sampling_min",
        "Entropy_Sampling_min",
        "Uncertainty_Sampling_max",
        "Margin_Sampling_max",
        "Entropy_Sampling_max",
    ]

    span = 2
    mask = CrossNeighborhoodPattern(span).get_mask()

    evaluation_steps = [x * 5 for x in range(1, 10)] + [x * 10 for x in range(5, 10)] + [x * 100 for x in range(1, 26)]

    all_results = dict()

    evaluation_patterns = np.load("activelearning\\pattern_based_active_learning\\data\\evaluation_patterns.npy")
    evaluation_state_transitions = np.load("activelearning\\pattern_based_active_learning\\data\\evaluation-state-transitions.npy")


    for agent_name in agents:
        results = None

        if os.path.exists(f"activelearning\\gamestate_based_active_learning\\accuracy_results\\{agent_name}_pattern__accuracy_lock.txt"):
            continue
        else:
            with open(f"activelearning\\gamestate_based_active_learning\\accuracy_results\\{agent_name}_pattern__accuracy_lock.txt", "wb") as file:
                pickle.dump(" ", file)

        if os.path.exists(f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}_final_result.txt"):
            with open(f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}_final_result.txt",
                      "rb") as file:
                results = pickle.load(file)

        elif os.path.exists(f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}.txt"):
            with open(f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}.txt", "rb") as file:
                results = pickle.load(file)

        if results is not None:
            test_pattern_accuracy(results, evaluation_patterns)

            with open(f"activelearning\\gamestate_based_active_learning\\accuracy_results\\{agent_name}_pattern__accuracy.txt", "wb") as file:
                pickle.dump(results, file)


    for agent_name in agents:
        results = None

        if os.path.exists(f"activelearning\\gamestate_based_active_learning\\accuracy_results\\{agent_name}_state_accuracy_lock.txt"):
            continue
        else:
            with open(f"activelearning\\gamestate_based_active_learning\\accuracy_results\\{agent_name}_state_accuracy_lock.txt", "wb") as file:
                pickle.dump(" ", file)

        if os.path.exists(f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}_final_result.txt"):
            with open(f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}_final_result.txt",
                      "rb") as file:
                results = pickle.load(file)

        elif os.path.exists(f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}.txt"):
            with open(f"activelearning\\gamestate_based_active_learning\\results\\{agent_name}.txt", "rb") as file:
                results = pickle.load(file)

        if results is not None:
            test_state_accuracy(results, evaluation_state_transitions, mask, span)

            with open(f"activelearning\\gamestate_based_active_learning\\accuracy_results\\{agent_name}_state_accuracy.txt", "wb") as file:
                pickle.dump(results, file)

