from typing import List, Optional

import pandas as pd
from matplotlib import pyplot as plt

from python.helpers import common, project_helper

from python.datasets_objects import justice

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

    plt.savefig(f"{PATH_IMAGES}{name_file}.png", dpi=140, bbox_inches='tight')
    plt.close()


def run_process(
    preprocessing_strategies: Optional[List[common.PostProcessingStrategy]] = None,
    postprocessing_strategy: Optional[common.PostProcessingStrategy] = None,
):
    sds = justice.JusticeDataset()
    database_name = 'Justice'

    dataset_orig_train, dataset_orig_test = sds.split([0.7], shuffle=True, seed=42)

    fairness_process = project_helper.FairnessProcessMitigation(
        dataset_origin_train=dataset_orig_train,
        dataset_origin_test=dataset_orig_test,
        privileged_groups=sds.default_privileged_groups,
        unprivileged_groups=sds.default_unprivileged_groups,
        privileged_groups_encoding_map=justice.ETHNICITY_ENCODING,
        name_dataset=database_name,
        tuning_scoring='precision',
        label_verb='incarcerated',
        n_jobs=-1,
        save_images=True,
        preprocessing_strategies=preprocessing_strategies,
        postprocessing_strategy=postprocessing_strategy,
        models_to_fit=[
            common.AvailableModels.DECISION_TREE,
            common.AvailableModels.LOGISTIC_REGRESSION,
            common.AvailableModels.RANDOM_FOREST,
            common.AvailableModels.GRADIENT_BOOSTING,
            common.AvailableModels.K_NEIGHBORS,
        ]
    )
    fairness_process.fit_models()
    fairness_process.equalized_odds_models()


def plot_metrics():
    # Example to create a image from a dataframe
    metadata = common.get_model_metadata()
    names_db = []
    disparate_impacts_db = []
    statistical_parity_difference_db = []
    for db in metadata.keys():
        names_db.append(db.split('_')[1])
        db_selected = metadata.get(db)
        db_metrics = db_selected.get('db_metrics')

        disparate_impacts_db.append(
            round(db_metrics.get('disparate_impact'), 4)
        )
        statistical_parity_difference_db.append(
            round(db_metrics.get('statistical_parity_difference'), 4)
        )

        data = dict(
            db=names_db,
            DI=disparate_impacts_db,
            statistical_parity_diff=statistical_parity_difference_db,
        )
        df_table = pd.DataFrame(data)
        df_table.sort_values(by='db', inplace=True)
        capture_table_dataframe_image(
            table_name='testing',
            name_file='testing',
            col_widths=[0.1, 0.3, 0.3],
            figure_size_height=1,
            figure_size_width=3,
            dataframe=df_table,
            font_size=5
        )


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
