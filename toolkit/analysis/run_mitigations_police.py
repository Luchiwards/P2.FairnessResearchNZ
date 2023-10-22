import random
from typing import Optional, List

import numpy as np

np.random.seed(760)

from toolkit.helpers import common, project_helper
from toolkit.datasets_objects import police

random.seed(42)

PATH_IMAGES = 'toolkit/plots/'


def run_process(
        preprocessing_strategies: Optional[List[common.PostProcessingStrategy]] = None,
        postprocessing_strategy: Optional[common.PostProcessingStrategy] = None,
):
    sds = police.PoliceDataset()
    database_name = 'Police'

    dataset_orig_train, dataset_orig_test = sds.split([0.7], shuffle=True, seed=42)

    fairness_process = project_helper.FairnessProcessMitigation(
        dataset_origin_train=dataset_orig_train,
        dataset_origin_test=dataset_orig_test,
        privileged_groups=sds.default_privileged_groups,
        unprivileged_groups=sds.default_unprivileged_groups,
        privileged_groups_encoding_map=police.ETHNICITY_ENCODING,
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
            common.AvailableModels.K_NEIGHBORS,
        ]
    )
    fairness_process.fit_models()
    fairness_process.equalized_odds_models()


def run_mitigations():
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
