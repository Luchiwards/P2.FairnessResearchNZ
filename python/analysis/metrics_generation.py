import numpy as np
from matplotlib import pyplot as plt

from python.helpers import common


def plot_disparity_impact_by_ethnicity_algorithm():
    # Example to create a image from a dataframe
    metadata = common.get_model_metadata()
    metadata = {
        key: value for key, value in metadata.items() if 'Police' in key
    }
    for key, value in metadata.items():
        models = value.get('models', {})
        algorithms = {
            name: {k: v.get('disparate_impact') for k, v in values.get('prediction_metrics_by_group').items()}
            for name, values
            in models.items()
        }
        main_groups = list(algorithms.keys())
        subgroups = list(list(algorithms.values())[0].keys())

        values = list([di for di in algo.values()] for algo in algorithms.values())

        # Define bar width and positions
        bar_width = 0.1
        positions = np.arange(len(main_groups))

        # Plotting
        plt.figure(figsize=(10, 7))

        for i, subgroup in enumerate(subgroups):
            plt.barh(positions + i * bar_width, [v[i] for v in values], height=bar_width, label=subgroup)

        plt.yticks(positions + bar_width, main_groups)
        plt.xlim(0, 2.2)
        plt.xlabel('Disparity Impact from 0 to 1')
        plt.ylabel('Algorithms')
        plt.title(f"Disparity Impact for {' '.join(key.split('_'))}")
        plt.legend(loc="upper right", title="Subgroups")
        plt.gca().invert_yaxis()  # To have Maori at the top as per the order given
        plt.tight_layout()
        plt.savefig(f"../plots/metrics/di_eth_alg_{key}.png", dpi=80)
        plt.show()
        plt.close()


if __name__ == "__main__":
    plot_disparity_impact_by_ethnicity_algorithm()
