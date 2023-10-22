import os

import numpy as np
from matplotlib import pyplot as plt

from toolkit.helpers import common

METRICS_FOLDER = 'toolkit/plots/metrics/'


def plot_disparity_impact_by_ethnicity_algorithm(target):
    # Example to create a image from a dataframe
    metadata = common.get_model_metadata()
    metadata = {
        key: value for key, value in metadata.items() if target in key
    }
    for key, value in metadata.items():
        models = value.get('models', {})
        algorithms = {
            name: {k: v.get('disparate_impact') for k, v in values.get('prediction_metrics_by_group').items()}
            for name, values
            in models.items()
        }
        main_groups = list(algorithms.keys())
        subgroups = ['All' if ethnicity == 'Privileged' else ethnicity for ethnicity in
                     list(list(algorithms.values())[0].keys())]

        values = list([di for di in algo.values()] for algo in algorithms.values())

        # Define bar width and positions
        bar_width = 0.1
        positions = np.arange(len(main_groups))

        # Plotting
        plt.figure(figsize=(10, 7))

        for i, subgroup in enumerate(subgroups):
            bars = plt.barh(positions + i * bar_width, [v[i] for v in values], height=bar_width, label=subgroup)
            for bar, value in zip(bars, [v[i] for v in values]):
                plt.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                         va='center', ha='left', fontsize=8)  # Display value to the right of each bar

        plt.axvline(x=0.25, color='black', linestyle='--')
        plt.axvline(x=0.5, color='black', linestyle='--')
        plt.axvline(x=0.75, color='black', linestyle='--')
        plt.axvline(x=1, color='black', linestyle='--')
        plt.yticks(positions + bar_width, main_groups)
        plt.xlim(0, 2.2)
        plt.xlabel('Disparity Impact from 0 to 1')
        plt.ylabel('Algorithms')
        plt.title(f"Disparity Impact for {' '.join(key.split('_'))}")
        plt.legend(loc="upper right", title="Ethnicities")
        plt.gca().invert_yaxis()  # To have Maori at the top as per the order given
        plt.tight_layout()
        if not os.path.exists(METRICS_FOLDER):
            os.makedirs(METRICS_FOLDER)
        plt.savefig(f"{METRICS_FOLDER}di_eth_alg_{key}_{target.lower()}.png", dpi=80)
        plt.show()
        plt.close()


def plot_accuracy_by_ethnicity_algorithm(target):
    # Example to create a image from a dataframe
    metadata = common.get_model_metadata()
    metadata = {
        key: value for key, value in metadata.items() if target in key
    }
    for key, value in metadata.items():
        models = value.get('models', {})
        algorithms = {
            name: {k: v.get('accuracy') * 100 for k, v in values.get('prediction_metrics_by_group').items()}
            for name, values
            in models.items()
        }
        main_groups = list(algorithms.keys())
        subgroups = ['All' if ethnicity == 'Privileged' else ethnicity for ethnicity in
                     list(list(algorithms.values())[0].keys())]

        values = list([di for di in algo.values()] for algo in algorithms.values())

        # Define bar width and positions
        bar_width = 0.1
        positions = np.arange(len(main_groups))

        # Plotting
        plt.figure(figsize=(10, 7))

        for i, subgroup in enumerate(subgroups):
            bars = plt.barh(positions + i * bar_width, [v[i] for v in values], height=bar_width, label=subgroup)
            for bar, value in zip(bars, [v[i] for v in values]):
                plt.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                         va='center', ha='left', fontsize=8)  # Display value to the right of each bar

        plt.yticks(positions + bar_width, main_groups)
        plt.xlim(0, 120)
        plt.xlabel('Accuracy')
        plt.ylabel('Algorithms')
        plt.title(f"Accuracy for {' '.join(key.split('_'))}")
        plt.legend(loc="upper right", title="Ethnicities")
        plt.gca().invert_yaxis()  # To have Maori at the top as per the order given
        plt.tight_layout()
        if not os.path.exists(METRICS_FOLDER):
            os.makedirs(METRICS_FOLDER)
        plt.savefig(f"{METRICS_FOLDER}acc_eth_alg_{key}_{target.lower()}.png", dpi=80)
        plt.show()
        plt.close()


def plot():
    plot_disparity_impact_by_ethnicity_algorithm('Justice')
    plot_accuracy_by_ethnicity_algorithm('Justice')
    plot_disparity_impact_by_ethnicity_algorithm('Police')
    plot_accuracy_by_ethnicity_algorithm('Police')
