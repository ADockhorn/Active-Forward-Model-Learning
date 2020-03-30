from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from abstractclasses.AbstractNeighborhoodPattern import CrossNeighborhoodPattern
import numpy as np
from modAL.uncertainty import uncertainty_sampling, entropy_sampling, margin_sampling
from utils import random_sampling
from tqdm import trange
import pickle

disable_tqdm = False


def train_model(sampler, data_set, evaluation_steps):
    if isinstance(data_set, list):
        return _train_using_dynamic_data_set(sampler, data_set, evaluation_steps)
    else:
        return _train_using_static_data_set(sampler, data_set, evaluation_steps)


def _train_using_dynamic_data_set(sampler, data_set, evaluation_steps):
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=samplers[sampler],
    )

    queried_points = 0
    training_results = {"models": []}

    tmp_x = []
    tmp_y = []

    for data_index in trange(len(data_set), disable=disable_tqdm, desc=f"{sampler}-{data_set_name}"):
        x_train = data_set[data_index][:, :-1]
        y_train = data_set[data_index][:, -1]

        query_idx, query_inst = learner.query(x_train, n_instances=1)
        tmp_x.append(query_inst)
        tmp_y.append(y_train[query_idx])
        queried_points += 1

        if data_index+1 in evaluation_steps:
            learner.teach(np.array(tmp_x).reshape((len(tmp_x),-1)), np.array(tmp_y).flatten())
            tmp_x = []
            tmp_y = []

            lfm = DecisionTreeClassifier().fit(learner.X_training, learner.y_training)
            training_results["models"].append(lfm)

    return training_results


def _train_using_static_data_set(sampler, data_set, evaluation_steps):
    x_train = data_set[:, :-1]
    y_train = data_set[:, -1]

    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=samplers[sampler],
    )

    tmp_x_train, tmp_y_train = x_train.copy(), y_train.copy()

    queried_points = 0
    training_results = {"models": []}

    for step in trange(len(evaluation_steps), disable=disable_tqdm, desc=f"{sampler}-{data_set_name}"):
        query_idx, query_inst = learner.query(tmp_x_train, n_instances=evaluation_steps[step]-queried_points)

        # ...obtaining new labels from the pool...
        learner.teach(query_inst, tmp_y_train[query_idx])
        queried_points += evaluation_steps[step] - queried_points

        tmp_x_train = np.delete(tmp_x_train, query_idx, axis=0)
        tmp_y_train = np.delete(tmp_y_train, query_idx, axis=0)

        lfm = DecisionTreeClassifier().fit(learner.X_training, learner.y_training)
        training_results["models"].append(lfm)

    return training_results


if __name__ == "__main__":
    import os

    span = 2
    mask = CrossNeighborhoodPattern(span).get_mask()

    samplers = {
        "Random": random_sampling,
        "Uncertainty": uncertainty_sampling,
        "Margin": margin_sampling,
        "Entropy": entropy_sampling
    }

    game_state_patterns = np.load("activelearning\\pattern_based_active_learning\\data\\game_state_patterns.npy")
    random_patterns = np.load("activelearning\\pattern_based_active_learning\\data\\random_patterns.npy")

    changing_random_patterns = np.load("activelearning\\pattern_based_active_learning\\data\\changing_random_patterns.npy")
    changing_random_patterns = [changing_random_patterns[(batch * 20):((batch + 1) * 20), :] for batch in range(100000)]

    data_sets = {
        "game_state_patterns": game_state_patterns,
        "random_patterns": random_patterns,
        "changing_random_patterns": changing_random_patterns
    }

    evaluation_steps = [x*5 for x in range(1, 10)] + [x*10 for x in range(5, 10)] + [x*100 for x in range(1, 10)] + \
                       [x*1000 for x in range(1, 10)] + [x*10000 for x in range(1, 11)]

    for sampler in samplers:
        for data_set_name, data_set in data_sets.items():
            if os.path.exists(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}.txt"):
                continue
            else:
                with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}.txt", "wb") as file :
                    pickle.dump(" ", file)

            results = train_model(sampler, data_set, evaluation_steps)
            results["checkpoints"] = evaluation_steps

            with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}.txt", "wb") as file:
                pickle.dump(results, file)

