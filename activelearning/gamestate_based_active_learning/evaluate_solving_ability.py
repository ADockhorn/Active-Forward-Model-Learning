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
    if "model_checkpoints" not in results:
        return

    results["level-solving-ability"] = [0]*len(results["model_checkpoints"][:24])

    for model_idx in trange(len(results["model_checkpoints"][:24]), disable=disable_tqdm, desc=f"{sampler}"):
        solved = 0
        checked = 0
        for level in test_levels:
            checked += 1
            lfm = MinimalLocalForwardModel(results["model_checkpoints"][model_idx], mask, span)
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

    level_folder = "sokoban\\random_levels\\easy_levels"
    test_levels = [f"{level_folder}\\{file}" for file in os.listdir(level_folder)]

    # Test pattern accuracy
    all_results = dict()
    for sampler in agents:

        if os.path.exists(f"activelearning\\gamestate_based_active_learning\\solving_results\\{sampler}-Sampling_solvability_lock.txt"):
            continue
        else:
            with open(f"activelearning\\gamestate_based_active_learning\\solving_results\\{sampler}-Sampling_solvability_lock.txt", "wb") as file :
                pickle.dump(" ", file)

        if os.path.exists(f"activelearning\\gamestate_based_active_learning\\results\\{sampler}_final_result.txt"):
            with open(f"activelearning\\gamestate_based_active_learning\\results\\{sampler}_final_result.txt",
                      "rb") as file:
                results = pickle.load(file)

        elif os.path.exists(f"activelearning\\gamestate_based_active_learning\\results\\{sampler}.txt"):
            with open(f"activelearning\\gamestate_based_active_learning\\results\\{sampler}.txt", "rb") as file:
                results = pickle.load(file)

        if results is not None:
            test_level_solving_ability(results, test_levels, mask, span)
            all_results[sampler] = results

            with open(f"activelearning\\gamestate_based_active_learning\\solving_results\\{sampler}_solvability.txt", "wb") as file:
                pickle.dump(results, file)
