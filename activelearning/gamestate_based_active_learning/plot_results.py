import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


if __name__ == "__main__":
    agents = {
        "Random": "Random Agent",
        "Random_State_Selection": "Random State Action Selection",
        "Max_Unknown_Patterns": "Max Unknown Patterns",
        "Uncertainty_Sampling": "Uncertainty-Sampling",
        "Margin_Sampling": "Margin-Sampling",
        "Entropy_Sampling": "Entropy-Sampling"
        #"Uncertainty_Sampling_min",
        #"Margin_Sampling_min",
        #"Entropy_Sampling_min",
        #"Uncertainty_Sampling_max",
        #"Margin_Sampling_max",
        #"Entropy_Sampling_max",
    }

    evaluation_steps = [x * 5 for x in range(1, 10)] + [x * 10 for x in range(5, 10)] + [x * 100 for x in range(1, 26)]

    all_results = dict()

    for agent, agent_plotname in agents.items():
        results = None
        if os.path.exists(f"activelearning\\gamestate_based_active_learning\\results\\{agent}_final_result.txt"):
            with open(f"activelearning\\gamestate_based_active_learning\\results\\{agent}_final_result.txt",
                      "rb") as file:
                results = pickle.load(file)

        elif os.path.exists(f"activelearning\\gamestate_based_active_learning\\results\\{agent}.txt"):
            with open(f"activelearning\\gamestate_based_active_learning\\results\\{agent}.txt", "rb") as file:
                results = pickle.load(file)

        if results is not None:
            if os.path.exists(f"activelearning\\gamestate_based_active_learning\\accuracy_results\\{agent}_pattern__accuracy.txt"):
                with open(f"activelearning\\gamestate_based_active_learning\\accuracy_results\\{agent}_pattern__accuracy.txt",
                          "rb") as file:
                    pattern_accuracy = pickle.load(file)
                    results["pattern-based-accuracy"] = pattern_accuracy["pattern-based-accuracy"]

            if os.path.exists(
                    f"activelearning\\gamestate_based_active_learning\\accuracy_results\\{agent}_state_accuracy.txt"):
                with open(
                        f"activelearning\\gamestate_based_active_learning\\accuracy_results\\{agent}_state_accuracy.txt",
                        "rb") as file:
                    state_accuracy = pickle.load(file)
                    results["state-based-accuracy"] = state_accuracy["state-based-accuracy"]

            if os.path.exists(
                    f"activelearning\\gamestate_based_active_learning\\solving_results\\{agent}_solvability.txt"):
                with open(f"activelearning\\gamestate_based_active_learning\\solving_results\\{agent}_solvability.txt",
                          "rb") as file:
                    solvability = pickle.load(file)
                    if solvability["level-solving-ability"] is None:
                        print()
                    results["level-solving-ability"] = solvability["level-solving-ability"]

            all_results[agent_plotname] = results

    plt.figure(figsize=(8.5, 1.45))
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")
    ax = plt.subplot(111)
    # Number of Queries vs. Known Patterns
    for agent_name in all_results:
        if "known_patterns" in all_results[agent_name]:
            plt.plot(0, label=agent_name)
        elif "known_patterns_per_timestep" in all_results[agent_name]:
            plt.plot(0, label=agent_name)
    box = ax.get_position()
    ax.set_axis_off()
    plt.figlegend(ncol=2, )
    plt.tight_layout()
    plt.xlabel("number of queries")
    plt.ylabel("number of known patterns")
    plt.savefig(f"activelearning\\gamestate_based_active_learning\\figures\\legend.pdf")
    plt.show()


    plt.figure(figsize=(7,5))
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")
    ax = plt.subplot(111)
    # Number of Queries vs. Known Patterns
    for agent_name in all_results:
        if "known_patterns" in all_results[agent_name]:
            plt.plot(all_results[agent_name]["known_patterns"][:1000], label=agent_name)
        elif "known_patterns_per_timestep" in all_results[agent_name]:
            plt.plot(all_results[agent_name]["known_patterns_per_timestep"][:1000], label=agent_name)

    #plt.figlegend(ncol=2, )
    plt.xlabel("number of queries")
    plt.ylabel("number of known patterns")
    plt.tight_layout()
    plt.savefig(
        f"activelearning\\gamestate_based_active_learning\\figures\\queries-vs-known_patterns.pdf")
    plt.savefig(
        f"activelearning\\gamestate_based_active_learning\\figures\\queries-vs-known_patterns.png")
    plt.show()


    # Number of Queries vs. Pattern Accuracy
    plt.figure(figsize=(7, 5))
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")
    for agent_name in all_results:
        plt.plot(evaluation_steps[:24], all_results[agent_name]["pattern-based-accuracy"][:24], label=agent_name)
    plt.gca().set_ylim(-0.05, 1.05)
    #plt.xscale("log")
    plt.xlabel("number of queries")
    plt.ylabel("pattern-prediction accuracy")
    #plt.legend(ncol=2)
    plt.tight_layout()

    plt.savefig(
        f"activelearning\\gamestate_based_active_learning\\figures\\queries-vs-pattern-accuracy.pdf")
    plt.savefig(
        f"activelearning\\gamestate_based_active_learning\\figures\\queries-vs-pattern-accuracy.png")
    plt.show()


    # Number of Queries vs. State Accuracy
    plt.figure(figsize=(7, 5))
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")
    for agent_name in all_results:
        plt.plot(evaluation_steps[:24], all_results[agent_name]["state-based-accuracy"][:24], label=agent_name)
    plt.gca().set_ylim(-0.05, 1.05)
    #plt.xscale("log")
    plt.xlabel("number of queries")
    plt.ylabel("state-prediction accuracy")
    plt.tight_layout()
    plt.savefig(
        f"activelearning\\gamestate_based_active_learning\\figures\\queries-vs-state-accuracy.pdf")
    plt.savefig(
        f"activelearning\\gamestate_based_active_learning\\figures\\queries-vs-state-accuracy.png")
    plt.show()



    # Number of Queries vs. Level Solving Ability
    plt.figure(figsize=(7, 5))
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")
    for agent_name in all_results:
        plt.plot(evaluation_steps[:24], [x*100 for x in all_results[agent_name]["level-solving-ability"][:24]], label=agent_name)
    #plt.gca().set_ylim(-0.05, 1.05)
    #plt.xscale("log")
    plt.xlabel("number of queries")
    plt.ylabel("number of levels solved")
    plt.ylim((-0.5, 8.5))
    #plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"activelearning\\gamestate_based_active_learning\\figures\\queries-vs-solvability.pdf")
    plt.savefig(
        f"activelearning\\gamestate_based_active_learning\\figures\\queries-vs-solvability.png")
    plt.show()


    data = {"pattern-prediction accuracy": np.array([all_results[agent_name]["pattern-based-accuracy"][:24] for agent_name in all_results]).flatten(),
            "state-prediction accuracy": np.array([all_results[agent_name]["state-based-accuracy"][:24] for agent_name in all_results]).flatten(),
            "number of levels solved": np.array([all_results[agent_name]["level-solving-ability"][:24] for agent_name in all_results]).flatten(),
            }
    scatter_data = pd.DataFrame(data)


    # Pattern Accuracy vs. State Accuracy
    plt.figure(figsize=(7, 7))
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")
    #cmap = sns.color_palette("muted", len(data_sets))
    ax = sns.scatterplot(x="pattern-prediction accuracy", y="state-prediction accuracy", #palette=cmap,
                         #hue="training data set", style="training data set",
                         alpha=0.9, data=scatter_data,
                         s= 100)
    plt.tight_layout()
    plt.savefig("activelearning\\gamestate_based_active_learning\\figures\\pattern-vs-state-accuracy.pdf")
    plt.savefig("activelearning\\gamestate_based_active_learning\\figures\\pattern-vs-state-accuracy.png")
    plt.show()

    # Pattern Accuracy vs. Level Solvability
    plt.figure(figsize=(7, 7))
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")
    #cmap = sns.color_palette("muted", len(data_sets))
    ax = sns.scatterplot(x="pattern-prediction accuracy", y="number of levels solved", #palette=cmap,
                         #hue="training data set", style="training data set",
                         alpha=0.9, data=scatter_data,
                         s=100)
    plt.tight_layout()
    plt.savefig("activelearning\\gamestate_based_active_learning\\figures\\pattern-accuracy-vs-solvability.pdf")
    plt.savefig("activelearning\\gamestate_based_active_learning\\figures\\pattern-accuracy-vs-solvability.png")
    plt.show()

    # Pattern Accuracy vs. State Accuracy
    plt.figure(figsize=(7, 7))
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")
    #cmap = sns.color_palette("muted", len(data_sets))
    ax = sns.scatterplot(x="state-prediction accuracy", y="number of levels solved", #palette=cmap,
                         #hue="training data set", style="training data set",
                         alpha=0.9, data=scatter_data,
                         s=100)
    plt.tight_layout()
    plt.savefig("activelearning\\gamestate_based_active_learning\\figures\\state-accuracy-vs-solvability.pdf")
    plt.savefig("activelearning\\gamestate_based_active_learning\\figures\\state-accuracy-vs-solvability.png")
    plt.show()


