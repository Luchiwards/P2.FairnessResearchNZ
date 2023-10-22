import random
from typing import Optional, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(760)

from python.helpers import common, project_helper
from python.datasets_objects import police_2

random.seed(42)

PATH_IMAGES = '../plots/'


def capture_table_dataframe_image(
    table_name: str,
    name_file: str,
    col_widths: List[float],
    dataframe: pd.DataFrame,
    figure_size_height=10.0,
    figure_size_width=5.0,
    font_size=4,
    head=None,
):
    fig, ax = plt.subplots(figsize=(figure_size_width, figure_size_height))
    ax.axis('off')
    ax.axis('tight')
    data_table = ax.table(
        cellText=dataframe.head(head).values if head is not None else dataframe.values,
        colLabels=[' '.join(col.split('_')).capitalize() for col in dataframe.columns],
        colWidths=col_widths,
        loc='center'
    )
    plt.title(f'{table_name}')
    data_table.auto_set_font_size(False)
    data_table.set_fontsize(font_size)
    fig.tight_layout()

    plt.show()
    # plt.savefig(f"{PATH_IMAGES}{name_file}.png", dpi=140, bbox_inches='tight')
    # plt.close()


def run_process(
    preprocessing_strategies: Optional[List[common.PostProcessingStrategy]] = None,
    postprocessing_strategy: Optional[common.PostProcessingStrategy] = None,
):
    sds = police_2.PoliceDataset()
    database_name = 'Police'

    dataset_orig_train, dataset_orig_test = sds.split([0.7], shuffle=True, seed=42)

    fairness_process = project_helper.FairnessProcessMitigation(
        dataset_origin_train=dataset_orig_train,
        dataset_origin_test=dataset_orig_test,
        privileged_groups=sds.default_privileged_groups,
        unprivileged_groups=sds.default_unprivileged_groups,
        privileged_groups_encoding_map=police_2.ETHNICITY_ENCODING,
        name_dataset=database_name,
        label_verb='taken to court',
        tuning_scoring='precision',
        n_jobs=-1,
        save_images=True,
        preprocessing_strategies=preprocessing_strategies,
        postprocessing_strategy=postprocessing_strategy,
        models_to_fit=[
            common.AvailableModels.DECISION_TREE,
            common.AvailableModels.LOGISTIC_REGRESSION,
            common.AvailableModels.RANDOM_FOREST,
            common.AvailableModels.GRADIENT_BOOSTING,
            # common.AvailableModels.K_NEIGHBORS,
        ]
    )
    fairness_process.fit_models()
    fairness_process.equalized_odds_models()


if __name__ == "__main__":
    run_process()
    run_process(
        preprocessing_strategies=[common.PreProcessingStrategies.REWEIGHING]
    )
    run_process(
        preprocessing_strategies=[common.PreProcessingStrategies.DISPARATE_IMPACT_REMOVER]
    )
    run_process(
        preprocessing_strategies=[
            common.PreProcessingStrategies.DISPARATE_IMPACT_REMOVER,
            common.PreProcessingStrategies.REWEIGHING
        ]
    )
    run_process(
        preprocessing_strategies=[
            common.PreProcessingStrategies.DISPARATE_IMPACT_REMOVER,
        ],
        postprocessing_strategy=common.PostProcessingStrategy.CALIBRATE_ODDS,
    )
    run_process(
        preprocessing_strategies=[
            common.PreProcessingStrategies.REWEIGHING
        ],
        postprocessing_strategy=common.PostProcessingStrategy.CALIBRATE_ODDS,
    )
    run_process(
        preprocessing_strategies=[
            common.PreProcessingStrategies.DISPARATE_IMPACT_REMOVER,
            common.PreProcessingStrategies.REWEIGHING
        ],
        postprocessing_strategy=common.PostProcessingStrategy.CALIBRATE_ODDS,
    )
    run_process(
        postprocessing_strategy=common.PostProcessingStrategy.CALIBRATE_ODDS,
    )
