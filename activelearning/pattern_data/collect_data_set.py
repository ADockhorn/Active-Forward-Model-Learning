import pickle
from sokoban.fast_sokoban import Sokoban, add_patterns_to_data_set, oracle, transform_data_set
from abstractclasses.AbstractNeighborhoodPattern import CrossNeighborhoodPattern
import numpy as np
import os
import tqdm


def generate_pattern_data_set_from_gameplay(max_size, mask, span, levels):
    pbar = tqdm.trange(max_size, desc="collect game state patterns")

    game_state_patterns = dict()
    for i in range(10):
        for file in levels:
            game = Sokoban(f"sokoban\\random_levels\\training_levels\\{file}")
            prev_game_state = game.level.copy()
            for tick in range(100):
                action = game.get_random_action
                game.next(action)
                add_patterns_to_data_set(prev_game_state, action, game.level, mask, span, game_state_patterns)
                prev_game_state = game.level.copy()
                pbar.update(len(game_state_patterns)-pbar.n)
                if len(game_state_patterns) >= max_size:
                    break
            if len(game_state_patterns) >= max_size:
                break
        if len(game_state_patterns) >= max_size:
            break
    pbar.close()

    keys = list(game_state_patterns.keys())[:max_size]
    reduced_data_set = dict()
    for key in keys:
        reduced_data_set[key] = game_state_patterns[key]

    return reduced_data_set


def generate_state_data_set_from_gameplay(nr_of_transitions, levels, data_set=None):
    if data_set is None:
        data_set = dict()

    for i in range(10):
        for file in levels:
            game = Sokoban(f"sokoban\\random_levels\\test_levels\\{file}")
            prev_game_state = game.level.copy()
            for tick in range(100):
                action = game.get_random_action
                game.next(action)
                el = (*prev_game_state.flatten(), action)
                if el not in data_set:
                    data_set[el] = (*game.level.flatten(),)
                if len(data_set) >= nr_of_transitions:
                    break
                prev_game_state = game.level.copy()
            else:
                print(len(data_set))
            if len(data_set) >= nr_of_transitions:
                break
        if len(data_set) >= nr_of_transitions:
            break

    return data_set


def generate_random_patterns(nr_of_patterns, pattern_length, disjunct_set=None):
    if disjunct_set is None:
        disjunct_set = dict()

    data_set = dict()

    nr_of_missing_patterns = nr_of_patterns
    pbar = tqdm.trange(nr_of_missing_patterns, desc="collect random patterns")
    while nr_of_missing_patterns > 0:
        new_instance = np.random.randint(7, size=pattern_length)
        if np.sum(np.logical_or(new_instance == 2, new_instance == 5)) > 1:
            continue
        new_instance = np.hstack([new_instance, np.random.randint(4)])
        new_instance = tuple(new_instance.tolist())
        if new_instance not in data_set and new_instance not in disjunct_set:
            data_set[new_instance] = {oracle(new_instance[:-1], mask, new_instance[-1]): 1}
            pbar.update(1)
            nr_of_missing_patterns -= 1
        else:
            continue
    pbar.close()

    return data_set


if __name__ == "__main__":
    span = 2
    mask = CrossNeighborhoodPattern(span).get_mask()

    training_levels = os.listdir("sokoban\\random_levels\\training_levels")
    game_state_patterns = generate_pattern_data_set_from_gameplay(100000, mask, span, training_levels)
    random_patterns = generate_random_patterns(100000, np.sum(mask), game_state_patterns)
    previous_data_sets = set(game_state_patterns.keys()).union(set(random_patterns.keys()))
    changing_random_patterns = generate_random_patterns(100000*20, np.sum(mask), previous_data_sets)

    all_data_sets = set(game_state_patterns.keys()).union(set(random_patterns.keys())).union(set(changing_random_patterns.keys()))
    evaluation_data = generate_random_patterns(1000000, np.sum(mask), all_data_sets)

    print("validation of sets being disjoint")
    print("are game_state_patterns disjoint from evaluation data?",
          set(game_state_patterns.keys()).isdisjoint(evaluation_data.keys()))
    print("are random_patterns disjoint from evaluation data?",
          set(random_patterns.keys()).isdisjoint(evaluation_data.keys()))
    print("are changing_random_patterns disjoint from evaluation data?",
          set(changing_random_patterns.keys()).isdisjoint(evaluation_data.keys()))

    print("saving collected patterns")
    game_state_patterns = transform_data_set(game_state_patterns, "argmax")
    game_state_patterns = np.array(game_state_patterns, dtype=np.int)
    np.save("activelearning\\pattern_based_active_learning\\data\\game_state_patterns", game_state_patterns)
    print("game_state_patterns", game_state_patterns.shape)

    random_patterns = transform_data_set(random_patterns, "argmax")
    random_patterns = np.array(random_patterns, dtype=np.int)
    np.save("activelearning\\pattern_based_active_learning\\data\\random_patterns", random_patterns)
    print("random_patterns", random_patterns.shape)

    changing_random_patterns = transform_data_set(changing_random_patterns, "argmax")
    changing_random_patterns = np.array(changing_random_patterns, dtype=np.int)
    np.save("activelearning\\pattern_based_active_learning\\data\\changing_random_patterns", changing_random_patterns)
    print("changing_random_patternss", changing_random_patterns.shape)

    evaluation_patterns = transform_data_set(evaluation_data, "argmax")
    evaluation_patterns = np.array(evaluation_patterns, dtype=np.int)
    np.save("activelearning\\pattern_based_active_learning\\data\\evaluation_patterns", evaluation_patterns)
    print("evaluation_patterns", evaluation_patterns.shape)

    test_levels = os.listdir("sokoban\\random_levels\\test_levels")[:100]
    state_transition_data_set = dict()
    while len(state_transition_data_set) < 100000:
        state_transition_data_set = generate_state_data_set_from_gameplay(100000, test_levels, state_transition_data_set)

    data = np.zeros((100000, 201))
    for i, (state, resulting_state) in enumerate(state_transition_data_set.items()):
        data[i, :101] = state
        data[i, 101:] = resulting_state
    np.save("activelearning\\pattern_based_active_learning\\data\\evaluation-state-transitions", data)
