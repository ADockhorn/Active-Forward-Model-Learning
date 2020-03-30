from sklearn.metrics import accuracy_score
from abstractclasses.AbstractNeighborhoodPattern import CrossNeighborhoodPattern
import numpy as np
from activestateexploration.simple_lfm import MinimalLocalForwardModel
from sokoban.fast_sokoban import Sokoban
from tqdm import trange, tqdm
import pickle
from activestateexploration.simple_bfs import SimpleBFS

disable_tqdm = False


def test_level_solving_ability(results, test_levels, mask, span):
    if "models" not in results:
        return

    results["level-solving-ability"] = [0]*len(results["models"])

    for model_idx in trange(len(results["models"]), disable=disable_tqdm, desc=f"{sampler}-{data_set_name}"):
        solved = 0
        checked = 0
        for level in test_levels:
            checked += 1
            lfm = MinimalLocalForwardModel(results["models"][model_idx], mask, span)
            game = Sokoban(level)
            agent = SimpleBFS(expansions=2500,
                              terminal_model=lambda level: np.sum(level == 3) == 0 and np.sum(level == 4) == 0)
            agent.set_forward_model(lfm)

            for candidate_solution in agent.get_solution(game, max_length=25):
                if check_valid_solution(game.get_copy(), candidate_solution):
                    solved += 1
                    break
        print("model", model_idx, "solved", solved)

        results["level-solving-ability"][model_idx] = solved/len(test_levels)
        #print(model_idx, solved/len(test_levels))


def check_valid_solution(game, candidate_solution):
    if candidate_solution is None:
        return False

    for action in candidate_solution:
        game.next(action)
        if game.is_terminal():
            return True
    return False


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

    level_folder = "sokoban\\random_levels\\easy_levels"
    test_levels = [f"{level_folder}\\{file}" for file in os.listdir(level_folder)]

    data_sets = [
        "game_state_patterns",
        "random_patterns",
        "changing_random_patterns"
    ]

    # Test pattern accuracy
    all_results = dict()
    for sampler in samplers:
        all_results[sampler] = dict()
        for data_set_name in data_sets:

            if os.path.exists(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}_solvability_lock.txt"):
                continue
            else:
                with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}_solvability_lock.txt", "wb") as file :
                    pickle.dump(" ", file)

            with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}.txt", "rb") as file :
                results = pickle.load(file)

            test_level_solving_ability(results, test_levels, mask, span)
            all_results[sampler][data_set_name] = results

            with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}_solvability.txt",
                      "wb") as file:
                pickle.dump(results, file)
