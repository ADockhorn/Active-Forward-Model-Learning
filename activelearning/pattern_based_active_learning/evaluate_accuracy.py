from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from abstractclasses.AbstractNeighborhoodPattern import CrossNeighborhoodPattern
import numpy as np
from activestateexploration.simple_lfm import MinimalLocalForwardModel

from tqdm import trange
import pickle

disable_tqdm = False


def test_pattern_accuracy(results, test_data):
    if "models" not in results:
        return

    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    accuracy = np.zeros((len(results["models"])))
    for model_idx in trange(len(results["models"]), disable=disable_tqdm, desc=f"{sampler}-{data_set_name}"):
        model = results["models"][model_idx]
        accuracy[model_idx] = accuracy_score(y_test, model.predict(x_test))
    results["pattern-based-accuracy"] = accuracy


def test_state_accuracy(results, test_data, mask, span):
    if "models" not in results:
        return

    x_state, x_action, y_test = test_data[:, :100], test_data[:, 100], test_data[:, 101:]

    accuracy_values = np.zeros(len(results["models"]))
    for idx in trange(len(results["models"]), disable=disable_tqdm, desc=f"{sampler}-{data_set_name}"):
        model = MinimalLocalForwardModel(results["models"][idx], mask, span, remember_predictions=True)

        correct = 0
        #pbar = trange(x_state.shape[0], desc="measure game-state-accuracy")
        for prev_gamestate, action, result_state in zip(x_state, x_action, y_test):
            #pbar.update(1)
            pred = model.predict(prev_gamestate.reshape((10, 10)), action)
            if np.all(pred == result_state.reshape((10, 10))):
                correct += 1
        #pbar.close()
        accuracy_values[idx] = correct/len(x_state)
    results["state-based-accuracy"] = accuracy_values


if __name__ == "__main__":
    import os

    span = 2
    mask = CrossNeighborhoodPattern(span).get_mask()

    samplers = [
        "Random",
        "Uncertainty",
        "Margin",
        "Entropy"
    ]

    evaluation_patterns = np.load("activelearning\\pattern_based_active_learning\\data\\evaluation_patterns.npy")
    evaluation_state_transitions = np.load("activelearning\\pattern_based_active_learning\\data\\evaluation-state-transitions.npy")

    data_sets = [
        "game_state_patterns",
        "random_patterns",
        "changing_random_patterns"
    ]

    # Test pattern accuracy
    for sampler in samplers:
        for data_set_name in data_sets:
            if os.path.exists(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}_lock.txt"):
                continue
            else:
                with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}_lock.txt", "wb") as file :
                    pickle.dump(" ", file)

                with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}.txt", "rb") as file :
                    results = pickle.load(file)

            if "pattern-based-accuracy" not in results:
                results["pattern-based-accuracy"] = None
                with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}.txt", "wb") as file:
                    pickle.dump(results, file)

                test_pattern_accuracy(results, evaluation_patterns)

                with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}.txt", "wb") as file:
                    pickle.dump(results, file)

            # Test state accuracy
            if "state-based-accuracy" not in results or results["state-based-accuracy"] == [-1]:
                results["state-based-accuracy"] = None
                with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}.txt", "wb") as file:
                    pickle.dump(results, file)

                test_state_accuracy(results, evaluation_state_transitions, mask, span)

                with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}.txt", "wb") as file:
                    pickle.dump(results, file)
