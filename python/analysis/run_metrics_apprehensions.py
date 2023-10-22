import random

from python.helpers import common, project_helper
from python.datasets_objects import annual_apprehensions

random.seed(42)

sds = annual_apprehensions.AnnualApprehensionsDataset()
annual_apprehensions.plots_annual_apprehension()

dataset_orig_train, dataset_orig_test = sds.split([0.9], shuffle=True)

# ###### 1. DATA SET METRICS ###########
fairness_metrics_dataset = project_helper.FairnessMetricsDataset(
    origin_dataset=dataset_orig_train,
    privileged_groups=sds.default_privileged_groups,
    unprivileged_groups=sds.default_unprivileged_groups,
)

print(fairness_metrics_dataset.summary())

###### 2. MODEL METRICS ###########

print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
print("::::::::::::::::::::::: Model Metrics :::::::::::::::::::::::::")
print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

fairness_metrics_dt = project_helper.FairnessProcessMitigation(
    dataset_origin_train=dataset_orig_train,
    dataset_origin_test=dataset_orig_test,
    privileged_groups=sds.default_privileged_groups,
    unprivileged_groups=sds.default_unprivileged_groups,
    privileged_groups_encoding_map=annual_apprehensions.ETHNICITY_ENCODING,
    name_dataset='Annual Apprehensions',
    tuning_scoring='precision',
    tuning=False,
    save_images=False,
    models_to_fit=[
        common.AvailableModels.DECISION_TREE,
        # common.AvailableModels.LOGISTIC_REGRESSION,
        # common.AvailableModels.RANDOM_FOREST,
        # AvailableModels.LINEAR_SVM,
    ]
)
fairness_metrics_dt.fit_models()
fairness_metrics_dt.equalized_odds_models()
