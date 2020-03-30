import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


if __name__ == "__main__":
    samplers = [
        "Random",
        "Uncertainty",
        "Margin",
        "Entropy"
    ]

    data_sets = {
        "game_state_patterns": "game-state patterns",
        "random_patterns": "random patterns",
        "changing_random_patterns": "changing random patterns"
    }

    evaluation_steps = [x*5 for x in range(1, 10)] + [x*10 for x in range(5, 10)] + [x*100 for x in range(1, 10)] + \
                       [x*1000 for x in range(1, 10)] + [x*10000 for x in range(1, 11)]

    all_scatter_data = pd.DataFrame()
    for i, (data_set_name, data_set_plot_name) in enumerate(data_sets.items()):
        pattern_based_accuracy = dict()
        state_based_accuracy = dict()
        solvability = dict()

        for sampler in samplers:

            if os.path.exists(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}.txt"):
                results = None
                with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}.txt", "rb") as file:
                    results = pickle.load(file)
                    if "pattern-based-accuracy" in results and results["pattern-based-accuracy"] is not None:
                        pattern_based_accuracy[sampler] = results["pattern-based-accuracy"]
                    if "state-based-accuracy" in results and results["state-based-accuracy"] is not None:
                        state_based_accuracy[sampler] = results["state-based-accuracy"]
            if os.path.exists(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}_solvability.txt"):

                with open(f"activelearning\\pattern_based_active_learning\\results\\{sampler}-Sampling_{data_set_name}_solvability.txt", "rb") as file:
                    results = pickle.load(file)
                    if "level-solving-ability" in results and results["level-solving-ability"] is not None:
                        solvability[sampler] = results["level-solving-ability"]

        data = {"pattern-prediction accuracy": np.array([pattern_based_accuracy[x] for x in pattern_based_accuracy]).flatten(),
                "state-prediction accuracy": np.array([state_based_accuracy[x] for x in state_based_accuracy]).flatten(),
                "number of levels solved": np.array([[y*100]  for x in solvability for y in solvability[x]]).flatten(),
                "training data set": [data_set_plot_name]*len(np.array([state_based_accuracy[x] for x in state_based_accuracy]).flatten()),
                "size": [10]*len(np.array([state_based_accuracy[x] for x in state_based_accuracy]).flatten())}
        scatter_data = pd.DataFrame(data)
        all_scatter_data = all_scatter_data.append(scatter_data)

        # Number of Queries vs. Pattern Accuracy
        plt.figure(figsize=(7, 5))
        sns.set(font_scale=1.7)

        sns.set_style("whitegrid")
        for sampler in samplers:
            plt.plot(evaluation_steps, pattern_based_accuracy[sampler], label=sampler)
        plt.gca().set_ylim(0, 1)

        # Put a legend below current axis
        plt.gca().legend(loc='lower right',
                  fancybox=True, shadow=False, ncol=2)
        plt.xscale("log")
        plt.xlabel("number of queries")
        plt.ylabel("pattern-prediction accuracy")
        plt.gca().set_ylim(-0.05, 1.05)
        #plt.title(data_set_name)
        plt.tight_layout()
        plt.savefig(f"activelearning\\pattern_based_active_learning\\figures\\queries-vs-pattern-accuracy-{data_set_name}.pdf")
        plt.savefig(f"activelearning\\pattern_based_active_learning\\figures\\queries-vs-pattern-accuracy-{data_set_name}.png")
        plt.show()

        # Number of Queries vs. State Accuracy
        plt.figure(figsize=(7, 5))
        sns.set(font_scale=1.7)
        sns.set_style("whitegrid")

        for sampler in samplers:
            plt.plot(evaluation_steps, state_based_accuracy[sampler], label=sampler)
        plt.gca().set_ylim(-0.05, 1.05)

        plt.xscale("log")
        plt.xlabel("number of queries")
        plt.ylabel("state-prediction accuracy")
        plt.tight_layout()
        plt.savefig(f"activelearning\\pattern_based_active_learning\\figures\\queries-vs-state-accuracy-{data_set_name}.pdf")
        plt.savefig(f"activelearning\\pattern_based_active_learning\\figures\\queries-vs-state-accuracy-{data_set_name}.png")
        plt.show()


        # Number of Queries vs. State Accuracy
        plt.figure(figsize=(7, 5))
        sns.set(font_scale=1.7)
        sns.set_style("whitegrid")

        for sampler in samplers:
            plt.plot(evaluation_steps, [x*100 for x in solvability[sampler]], label=sampler)
        #plt.gca().set_ylim(-0.05, 1.05)

        plt.xscale("log")
        plt.ylim((-2, 52))
        plt.xlabel("number of queries")
        plt.ylabel("number of levels solved")
        plt.tight_layout()
        plt.savefig(f"activelearning\\pattern_based_active_learning\\figures\\queries-vs-solvability-{data_set_name}.pdf")
        plt.savefig(f"activelearning\\pattern_based_active_learning\\figures\\queries-vs-solvability-{data_set_name}.png")
        plt.show()

    
    # Pattern Accuracy vs. State Accuracy
    plt.figure(figsize=(7, 7))
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")

    cmap = sns.color_palette("muted", len(data_sets))
    ax = sns.scatterplot(x="pattern-prediction accuracy", y="state-prediction accuracy", palette=cmap,
                         hue="training data set", style="training data set", alpha=0.9, data=all_scatter_data,
                         s= 100)
    plt.tight_layout()
    plt.savefig("activelearning\\pattern_based_active_learning\\figures\\pattern-vs-state-accuracy.pdf")
    plt.savefig("activelearning\\pattern_based_active_learning\\figures\\pattern-vs-state-accuracy.png")
    plt.show()

    # Pattern Accuracy vs. Level Solvability
    plt.figure(figsize=(7, 7))
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")

    cmap = sns.color_palette("muted", len(data_sets))
    ax = sns.scatterplot(x="pattern-prediction accuracy", y="number of levels solved", palette=cmap,
                         hue="training data set", style="training data set", alpha=0.7, data=all_scatter_data,
                         s=100)
    plt.tight_layout()
    plt.savefig("activelearning\\pattern_based_active_learning\\figures\\pattern-accuracy-vs-solvability.pdf")
    plt.savefig("activelearning\\pattern_based_active_learning\\figures\\pattern-accuracy-vs-solvability.png")
    plt.show()

    # Pattern Accuracy vs. State Accuracy
    plt.figure(figsize=(7, 7))
    sns.set(font_scale=1.7)
    sns.set_style("whitegrid")

    cmap = sns.color_palette("muted", len(data_sets))
    ax = sns.scatterplot(x="state-prediction accuracy", y="number of levels solved", palette=cmap,
                         hue="training data set", style="training data set", alpha=0.9, data=all_scatter_data,
                         s=100)
    plt.tight_layout()
    plt.savefig("activelearning\\pattern_based_active_learning\\figures\\state-accuracy-vs-solvability.pdf")
    plt.savefig("activelearning\\pattern_based_active_learning\\figures\\state-accuracy-vs-solvability.png")
    plt.show()
