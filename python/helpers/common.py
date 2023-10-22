import bz2
import enum
import json
from collections import OrderedDict
from io import BytesIO
from typing import Optional

from sklearn import ensemble, tree, linear_model

DATA_PATH = '../../data/'

METADATA_FILE_NAME = f'../metrics.json'


def get_model_metadata():
    try:
        with open(METADATA_FILE_NAME, 'r') as file:
            models_metadata = json.load(file)

    except FileNotFoundError:
        models_metadata = {}

    return models_metadata


class AvailableModels(enum.Enum):
    RANDOM_FOREST = 'Random Forest'
    DECISION_TREE = 'Decision Tree'
    GRADIENT_BOOSTING = 'Gradient Boosting'
    LOGISTIC_REGRESSION = 'Logistic Regression'
    LINEAR_SVM = 'Linear SVM'
    BERNOULLI_NB = 'Bernoulli NB'
    K_NEIGHBORS = 'K Neighbors'
    TESTING = 'Testing'


class DatasetsFiles(enum.Enum):
    DS_POLICE = 'dataset_police.csv.bz2'
    DS_JUSTICE_2001_2009 = 'justice-coded-2001-2009.csv.bz2'
    DS_JUSTICE_2010_2014 = 'justice-coded-2010-2014.csv.bz2'
    DS_JUSTICE_2015_2022 = 'justice-coded-2015-2022.csv.bz2'


class PostProcessingStrategy(enum.Enum):
    CALIBRATE_ODDS = 'PTCO'

    @staticmethod
    def humanize(enum_strategy: "PostProcessingStrategy") -> str:
        strategy_map = {
            PostProcessingStrategy.CALIBRATE_ODDS: 'Calibrate Equalized Odds',
        }
        return strategy_map.get(enum_strategy, '')


class PreProcessingStrategies(enum.Enum):
    DISPARATE_IMPACT_REMOVER = 'PDIR'
    REWEIGHING = 'PRW'

    @staticmethod
    def humanize(enum_strategy: "PreProcessingStrategies") -> str:
        strategy_map = {
            PreProcessingStrategies.REWEIGHING: 'Reweighing',
            PreProcessingStrategies.DISPARATE_IMPACT_REMOVER: 'Disparate Impact Remover',
        }
        return strategy_map.get(enum_strategy, '')


def decompress_bz2_file(path):
    try:
        with open(f'{DATA_PATH}{path}', 'rb') as infile:
            with bz2.BZ2File(infile, 'rb') as bz2file:
                decompressed_data = bz2file.read()
        return BytesIO(decompressed_data)
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def get_feature_importance(model, feature_names: list) -> Optional[dict]:
    if isinstance(
        model,
        (ensemble.GradientBoostingClassifier, ensemble.RandomForestClassifier, tree.DecisionTreeClassifier)
    ):
        importances = model.feature_importances_
    elif isinstance(model, linear_model.LogisticRegression):
        importances = model.coef_[0]
    else:
        return None

    paired_importance = dict(zip(feature_names, importances))
    sorted_importance = OrderedDict(sorted(paired_importance.items(), key=lambda x: x[1], reverse=True))

    return sorted_importance
