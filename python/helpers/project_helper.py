import json
import os
from typing import List, Dict, Optional
import joblib

import pandas as pd

from dataclasses import dataclass
from aif360 import metrics
import matplotlib.pyplot as plt
from sklearn import tree, linear_model, ensemble, svm, neighbors, naive_bayes, model_selection
from sklearn.metrics import accuracy_score

from python.helpers import common

from aif360 import datasets
from aif360.algorithms import preprocessing, postprocessing
from aif360.sklearn.metrics import statistical_parity_difference


def post_processing_calibrated_eq_odds(
    privileged_groups,
    unprivileged_groups
):
    # fpr, fnr or weighted
    return postprocessing.CalibratedEqOddsPostprocessing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        cost_constraint='fpr',
        seed=42
    )


def preprocessing_disparity_impact_remover(
    train_sds: datasets.StandardDataset
):
    # https://medium.com/ibm-data-ai/fairness-in-machine-learning-pre-processing-algorithms-a670c031fba8
    IR = preprocessing.DisparateImpactRemover(repair_level=1.0)
    return IR.fit_transform(train_sds)


def preprocessing_reweighing(
    train_sds: datasets.StandardDataset,
    unprivileged_groups,
    privileged_groups,
):
    # https://medium.com/ibm-data-ai/fairness-in-machine-learning-pre-processing-algorithms-a670c031fba8
    RW = preprocessing.Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    return RW.fit_transform(train_sds)


def demographic_parity(y_test, y_pred, hyperparameters):
    spd = statistical_parity_difference(y_test, y_pred, hyperparameters)
    # spd value indicates a disparity in favor of the privileged group,
    # and a negative value indicates a disparity in favor of the
    # unprivileged group. A value of 0 indicates that there is no disparity.
    return spd


@dataclass
class FairnessMetricsDataset:
    name_dataset: str
    privileged_groups_encoding_map: Dict[str, int]
    origin_dataset: datasets.StandardDataset
    privileged_groups: List[Dict[str, int]]
    unprivileged_groups: List[Dict[str, int]]
    metrics_dataset: metrics.BinaryLabelDatasetMetric = None

    def __post_init__(self):
        current_metadata = common.get_model_metadata()

        metrics_per_group = {}
        mappings = {v: k for k, v in self.privileged_groups_encoding_map.items()}

        for group in self.unprivileged_groups:
            mm = metrics.BinaryLabelDatasetMetric(
                dataset=self.origin_dataset,
                privileged_groups=self.privileged_groups,
                unprivileged_groups=[group],
            )

            metrics_per_group[mappings[group['ethnicity']]] = {
                "disparate_impact": mm.disparate_impact(),
                "statistical_parity_difference": mm.statistical_parity_difference(),
            }

        mm = metrics.BinaryLabelDatasetMetric(
            dataset=self.origin_dataset,
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
        )

        metrics_per_group['Privileged'] = {
            "disparate_impact": mm.disparate_impact(),
            "statistical_parity_difference": mm.statistical_parity_difference(),
        }

        with open(common.METADATA_FILE_NAME, 'w') as file:
            database_metadata = current_metadata.get(self.name_dataset, {})
            database_metadata.update({"db_metrics": metrics_per_group})
            current_metadata.update({self.name_dataset: database_metadata})
            json.dump(current_metadata, file)


@dataclass
class FairnessProcessMitigation:
    name_dataset: str
    label_verb: str
    dataset_origin_train: datasets.StandardDataset
    dataset_origin_test: datasets.StandardDataset
    privileged_groups: List[Dict[str, int]]
    privileged_groups_encoding_map: Dict[str, int]
    unprivileged_groups: List[Dict[str, int]]
    models_to_fit: List[common.AvailableModels]
    preprocessing_strategies: Optional[List[common.PreProcessingStrategies]]
    postprocessing_strategy: Optional[common.PostProcessingStrategy]
    tuning_scoring: str = 'accuracy'
    save_images: bool = False
    metrics_classifier: metrics.ClassificationMetric = None
    models_prediction_labels: Dict[common.AvailableModels, pd.Series] = None
    models_prediction_scores: Dict[common.AvailableModels, pd.Series] = None
    models_dataset_trans: Dict[common.AvailableModels, datasets.StandardDataset] = None
    _x_train: datasets.StandardDataset = None
    _y_train: datasets.StandardDataset = None
    _y_test: pd.Series = None
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.preprocessing_strategies is not None:
            for strategy in self.preprocessing_strategies:
                if strategy == common.PreProcessingStrategies.DISPARATE_IMPACT_REMOVER:
                    self.dataset_origin_train = preprocessing_disparity_impact_remover(self.dataset_origin_train)
                if strategy == common.PreProcessingStrategies.REWEIGHING:
                    self.dataset_origin_train = preprocessing_reweighing(
                        train_sds=self.dataset_origin_train,
                        privileged_groups=self.privileged_groups,
                        unprivileged_groups=self.unprivileged_groups,
                    )

        FairnessMetricsDataset(
            privileged_groups_encoding_map=self.privileged_groups_encoding_map,
            name_dataset=self.get_model_database_name(),
            origin_dataset=self.dataset_origin_train,
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
        )

        self._x_train = self.dataset_origin_train.features
        self._y_train = self.dataset_origin_train.labels.ravel()
        self._x_test = self.dataset_origin_test.features
        self._y_test = self.dataset_origin_test.labels.ravel()
        self.models_prediction_labels = {}
        self.models_prediction_scores = {}
        self.models_dataset_trans = {}

    def get_model_database_name(self):
        name = f"{'_'.join(self.name_dataset.split(' '))}_NM"
        if self.preprocessing_strategies is not None or self.postprocessing_strategy is not None:
            name = self.name_dataset
            if self.preprocessing_strategies is not None:
                for strategy in self.preprocessing_strategies:
                    name = f'{name}_{strategy.value}'
            if self.postprocessing_strategy is not None:
                name = f'{name}_{self.postprocessing_strategy.value}'
        return '_'.join(name.split(' '))

    def get_strategies_used(self) -> List[str]:
        nom = ["NM"]
        if self.preprocessing_strategies is not None or self.postprocessing_strategy is not None:
            nom = []
            if self.preprocessing_strategies is not None:
                for strategy in self.preprocessing_strategies:
                    nom.append(common.PreProcessingStrategies.humanize(enum_strategy=strategy))
            if self.postprocessing_strategy is not None:
                nom.append(common.PostProcessingStrategy.humanize(enum_strategy=self.postprocessing_strategy))
        return nom

    def _best_model_params(self, params, classifier, enum_model: common.AvailableModels):
        print(f"Tuning Model for {self.tuning_scoring}")

        database_name = self.get_model_database_name()
        folder = f"../models/{database_name}/"

        if not os.path.exists(folder):
            os.makedirs(folder)

        model_name = '_'.join(enum_model.value.split(' '))
        model_filename = f'{folder}{model_name}_best_classifier.joblib.bz2'

        try:
            best_clf = joblib.load(model_filename)

        except FileNotFoundError:
            grid = model_selection.GridSearchCV(
                estimator=classifier,
                param_grid=params,
                scoring=self.tuning_scoring,
                n_jobs=self.n_jobs,
                cv=5,
                verbose=1,
            )
            grid.fit(X=self._x_train, y=self._y_train)
            best_clf = grid.best_estimator_

            joblib.dump(
                best_clf,
                model_filename,
                compress=('bz2', 3)
            )

            new_metadata = {
                "training_info": {
                    "best_params": grid.best_params_,
                    "best_score": grid.best_score_,
                },
            }
            feature_importance = common.get_feature_importance(
                model=best_clf,
                feature_names=self.dataset_origin_train.feature_names
            )
            if feature_importance:
                new_metadata.update({"feature_importance": feature_importance})

            current_metadata = common.get_model_metadata()

            with open(common.METADATA_FILE_NAME, 'w') as file:
                database_metadata = current_metadata.get(database_name, {})
                models_metadata = database_metadata.get("models", {})
                model_metadata = models_metadata.get(model_name, {})

                # Update metadata
                model_metadata.update(new_metadata)
                models_metadata.update({model_name: model_metadata})
                database_metadata.update({"models": models_metadata})
                current_metadata.update({database_name: database_metadata})

                json.dump(current_metadata, file)

            print("Best parameters :::::::::::")
            print(grid.best_params_)
        return best_clf

    # def plot_threshold_

    def apply_post_processing(self, enum_model: common.AvailableModels):
        predictions = self.models_prediction_labels[enum_model]
        scores = self.models_prediction_scores[enum_model]
        if self.postprocessing_strategy == common.PostProcessingStrategy.CALIBRATE_ODDS:
            print("Applying Calibrate Eq Odds :::::::::::")
            cpp = post_processing_calibrated_eq_odds(
                privileged_groups=self.privileged_groups,
                unprivileged_groups=self.unprivileged_groups,
            )
            origin_dataset_test_pred = self.dataset_origin_test.copy(deepcopy=True)
            origin_dataset_test_pred.labels = predictions
            origin_dataset_test_pred.scores = scores
            cpp.fit(self.dataset_origin_test, origin_dataset_test_pred)
            self.models_dataset_trans[enum_model] = cpp.predict(origin_dataset_test_pred)

    def fit_models(self) -> None:
        fit_function_mapper = {
            common.AvailableModels.LOGISTIC_REGRESSION: self._fit_logistic_regression,
            common.AvailableModels.DECISION_TREE: self._fit_decision_tree,
            common.AvailableModels.RANDOM_FOREST: self._fit_random_forest,
            common.AvailableModels.LINEAR_SVM: self._fit_svm,
            common.AvailableModels.BERNOULLI_NB: self._fit_bernoulli_nb,
            common.AvailableModels.GRADIENT_BOOSTING: self._fit_gradient_boosting,
            common.AvailableModels.K_NEIGHBORS: self._fit_k_neighbors,
            common.AvailableModels.TESTING: self._fit_testing,
        }
        for model_enum in self.models_to_fit:
            print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
            print(f"::::::::::::::: Fitting : {model_enum.value} :::::::::::::::::")
            print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
            fit_function_mapper.get(model_enum)()

            accuracy = accuracy_score(y_true=self._y_test, y_pred=self.models_prediction_labels[model_enum])
            print(f"Accuracy {model_enum.value}: {accuracy}")

    def _fit_predict_classifier(self, classifier, enum_model: common.AvailableModels):
        predictions = classifier.predict(self._x_test)
        scores = classifier.predict_proba(self._x_test)[:, 1].reshape(-1, 1)
        print("Predicting...")
        self.models_prediction_labels[enum_model] = predictions
        self.models_prediction_scores[enum_model] = scores
        self.apply_post_processing(enum_model=enum_model)

    def _fit_decision_tree(self) -> None:
        params = {
            'criterion': [
                'entropy',
                # 'gini',
            ],
            # 'max_depth': [10, 20, 30, 40, None],
            # 'min_samples_split': [1, 10, 20],
            # 'min_samples_leaf': [1, 4, 6, 8],
        }
        classifier = tree.DecisionTreeClassifier(random_state=42)
        classifier = self._best_model_params(
            params=params,
            classifier=classifier,
            enum_model=common.AvailableModels.DECISION_TREE
        )
        self._fit_predict_classifier(classifier=classifier, enum_model=common.AvailableModels.DECISION_TREE)

    def _fit_logistic_regression(self) -> None:
        params = {
            # 'C': np.logspace(-4, 4, 10),
            # 'penalty': ['l1', 'l2'],
            'solver': ['lbfgs'],
            # 'l1_ratio': np.linspace(0, 1, 10),
        }
        classifier = linear_model.LogisticRegression(n_jobs=self.n_jobs, max_iter=10000, random_state=42)
        classifier = self._best_model_params(
            params=params,
            classifier=classifier,
            enum_model=common.AvailableModels.LOGISTIC_REGRESSION
        )
        self._fit_predict_classifier(classifier=classifier, enum_model=common.AvailableModels.LOGISTIC_REGRESSION)

    def _fit_random_forest(self) -> None:
        params = {
            # 'n_estimators': [10, 50, 100, 200, 500],
            # 'max_features': ['auto', 'sqrt', 'log2'],
            # 'max_depth': [None, 10, 20, 30, 40, 50],
            # 'min_samples_split': [2, 5, 10],
            # 'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
        }
        classifier = ensemble.RandomForestClassifier(n_jobs=self.n_jobs, random_state=42)
        classifier = self._best_model_params(
            params=params,
            classifier=classifier,
            enum_model=common.AvailableModels.RANDOM_FOREST
        )
        self._fit_predict_classifier(classifier=classifier, enum_model=common.AvailableModels.RANDOM_FOREST)

    def _fit_svm(self) -> None:
        classifier = svm.SVC()
        self._fit_predict_classifier(classifier=classifier, enum_model=common.AvailableModels.LINEAR_SVM)

    def _fit_bernoulli_nb(self) -> None:
        classifier = naive_bayes.BernoulliNB()
        self._fit_predict_classifier(classifier=classifier, enum_model=common.AvailableModels.BERNOULLI_NB)

    def _fit_k_neighbors(self) -> None:
        params = {
            'n_neighbors': [1, 2, 4],
            #     'n_neighbors': np.arange(1, 21),
            #     'weights': ['uniform', 'distance'],
            'metric': ['euclidean'],
            #     'metric': ['euclidean', 'manhattan', 'minkowski'],
            #     'p': [1, 2]
        }
        classifier = neighbors.KNeighborsClassifier(n_jobs=self.n_jobs)
        classifier = self._best_model_params(
            params=params,
            classifier=classifier,
            enum_model=common.AvailableModels.K_NEIGHBORS
        )
        self._fit_predict_classifier(classifier=classifier, enum_model=common.AvailableModels.K_NEIGHBORS)

    def _fit_gradient_boosting(self) -> None:
        classifier = ensemble.GradientBoostingClassifier(random_state=42)
        params = {
            'n_estimators': [50],
            # 'n_estimators': [50, 100, 200],
            # 'learning_rate': [0.001, 0.01],
            # 'max_depth': [3, 4, 5, 6],
            # 'min_samples_split': [2, 3, 4],
            # 'min_samples_leaf': [1, 2, 3],
            # 'subsample': [0.8, 0.9, 1]
        }
        classifier = self._best_model_params(
            params=params,
            classifier=classifier,
            enum_model=common.AvailableModels.GRADIENT_BOOSTING
        )
        self._fit_predict_classifier(classifier=classifier, enum_model=common.AvailableModels.GRADIENT_BOOSTING)

    def _fit_testing(self) -> None:
        classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
        self._fit_predict_classifier(classifier=classifier, enum_model=common.AvailableModels.TESTING)

    def _plot_equalized_odds(self, metrics_per_group, name: str, strategies: str):
        """
        P = incarcerated
        N = not_incarcerated

        FPR: Of all the people who are truly not incarcerated, what fraction did the
        model inaccurately predict as "incarcerated"?

        TPR: Of all the people who are truly incarcerated, what fraction did the model
        correctly predict as "incarcerated"?

        """

        plt.figure(figsize=(10, 7))
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'black', 'cyan']

        for k, group_metrics in metrics_per_group.items():
            fpr = group_metrics.get('fpr')
            tpr = group_metrics.get('tpr')
            index = group_metrics.get('index')
            if k != 'Privileged':
                plt.scatter(fpr, tpr, color=colors[index], marker='o', label=f'{k}', s=100)
            else:
                plt.scatter(fpr, tpr, color=colors[index], marker='x', label='European', s=100)

        recall = round(metrics_per_group.get('Privileged').get('recall') * 100, 2)
        precision = round(metrics_per_group.get('Privileged').get('precision') * 100, 2)
        accuracy = round(metrics_per_group.get('Privileged').get('accuracy') * 100, 2)

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f'{self.name_dataset}- {name}: Equalized Odds Plot')
        plt.xlabel(
            f"False Positive Rate (FPR) \n People who are truly NOT {self.label_verb} and wrongly predict as {self.label_verb}")
        plt.ylabel(
            f"True Positive Rate (TPR) \n People who are truly {self.label_verb} and correctly predict as {self.label_verb}")
        plt.legend(loc='lower right')
        plt.grid(True)

        metrics_text = f"Accuracy: {accuracy}% | Precision: {precision}% | Recall: {recall}%"

        # params_text = "Param1: 10 | Param2: 15"
        # Adjust the main axis limits to make space for the text boxes
        ax = plt.gca()  # Get the current axis
        ax_pos = ax.get_position()
        ax.set_position([ax_pos.x0, ax_pos.y0 + 0.15, ax_pos.width, ax_pos.height * 0.85])  # Adjust the main plot size

        # Add text boxes above the plot
        ax.text(0, - 0.22, metrics_text, size=12, ha="left", va="top")
        ax.text(0, - 0.29, f"Using: {strategies}", size=12, ha="left", va="top")
        # ax.text(0, - 0.35, params_text, size=12, ha="left", va="top")

        if self.save_images:
            name_model_file = '_'.join([word[0:4] for word in name.split(' ')])
            if not os.path.exists("../plots/"):
                os.makedirs("../plots/")
            plt.savefig(f"../plots/{self.get_model_database_name()}_{name_model_file}_eq_odds.png", dpi=80)

        plt.show()

    def _compute_metrics(
        self, model_enum: common.AvailableModels,
        compute_postprocessing: Optional[bool] = None
    ):
        if compute_postprocessing:
            model_predictions = self.models_dataset_trans[model_enum].labels
        else:
            model_predictions = self.models_prediction_labels[model_enum]

        metrics_per_group = {}
        classified_dataset = self.dataset_origin_test.copy(deepcopy=True)
        classified_dataset.labels = model_predictions

        mappings = {v: k for k, v in self.privileged_groups_encoding_map.items()}

        for group in self.unprivileged_groups:
            metric = metrics.ClassificationMetric(
                dataset=self.dataset_origin_test,
                classified_dataset=classified_dataset,
                unprivileged_groups=[group],
                privileged_groups=self.privileged_groups
            )
            metrics_per_group[mappings[group['ethnicity']]] = dict(
                tpr=metric.true_positive_rate(privileged=False),
                fpr=metric.false_positive_rate(privileged=False),
                accuracy=metric.accuracy(privileged=False),
                recall=metric.recall(privileged=False),
                precision=metric.precision(privileged=False),
                disparate_impact=metric.disparate_impact(),
                equal_opportunity_difference=metric.equal_opportunity_difference(),
                index=group['ethnicity']
            )

        metric_priv = metrics.ClassificationMetric(
            dataset=self.dataset_origin_test,
            classified_dataset=classified_dataset,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        metrics_per_group['Privileged'] = dict(
            tpr=metric_priv.true_positive_rate(privileged=True),
            fpr=metric_priv.false_positive_rate(privileged=True),
            accuracy=metric_priv.accuracy(privileged=True),
            recall=metric_priv.recall(privileged=True),
            precision=metric_priv.precision(privileged=True),
            disparate_impact=metric_priv.disparate_impact(),
            equal_opportunity_difference=metric_priv.equal_opportunity_difference(),
            index=0
        )

        database_name = self.get_model_database_name()
        model_name = '_'.join(model_enum.value.split(' '))

        current_metadata = common.get_model_metadata()

        new_metadata = {
            "prediction_metrics": {
                "accuracy": metric_priv.accuracy(),
                "recall": metric_priv.recall(),
                "precision": metric_priv.precision(),
                "disparate_impact": metric_priv.disparate_impact(),
            },
            "prediction_metrics_by_group": metrics_per_group,
            "has_post_processing": compute_postprocessing,
        }

        with open(common.METADATA_FILE_NAME, 'w') as file:
            database_metadata = current_metadata.get(database_name, {})
            models_metadata = database_metadata.get('models', {})
            model_metadata = models_metadata.get(model_name, {})

            # Update metadata
            model_metadata.update(new_metadata)
            models_metadata[model_name] = model_metadata
            database_metadata['models'] = models_metadata
            current_metadata[database_name].update(database_metadata)

            json.dump(current_metadata, file)
        return metrics_per_group

    def equalized_odds_models(self):
        nom_used = self.get_strategies_used()
        strategies_text = ', '.join([nom for nom in nom_used])

        for model_enum in self.models_to_fit:
            name = model_enum.value
            print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
            print(f":::::::: Generating {name}::::::::::::::::::::::::")
            print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

            if self.postprocessing_strategy is None:
                metrics_per_group = self._compute_metrics(model_enum)
                self._plot_equalized_odds(
                    metrics_per_group=metrics_per_group,
                    name=name,
                    strategies=strategies_text
                )
            else:
                if self.postprocessing_strategy == common.PostProcessingStrategy.CALIBRATE_ODDS:
                    metrics_per_group = self._compute_metrics(model_enum, compute_postprocessing=True)
                self._plot_equalized_odds(
                    metrics_per_group=metrics_per_group,
                    name=name,
                    strategies=strategies_text
                )
