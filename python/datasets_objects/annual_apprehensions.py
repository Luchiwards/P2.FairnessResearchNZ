import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
from matplotlib import pyplot as plt

from python.helpers import common

ETHNICITY_ENCODING = {
    'Caucasian': 0,
    'Asiatic': 1,
    'Maori': 2,
    'Other Ethnicities': 3,
    'Pacific Isle': 4,
    'Indian': 5
}


def default_preprocessing(df: pd.DataFrame, encode_ethnicity=True):
    """Perform the same preprocessing as the original analysis:
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    """
    df.columns = ["_".join(header.split(' ')).lower() for header in df.columns]
    df.rename(
        columns={
            "offence": "main_offence",
            "value": "number_of_observations",
            "age": "age_group",
        }, inplace=True
    )
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df['taken_to_court'] = np.where(df["resolution"] == 'Prosecution', 1, 0)
    "Other"
    df["ethnicity"] = np.where(df["ethnicity"] == 'Unknown', 'Other Ethnicities',
                               df["ethnicity"])
    df["ethnicity"] = np.where(df["ethnicity"] == 'Other', 'Other Ethnicities',
                               df["ethnicity"])
    df = df[df['gender'] != 'Other/Unknown']
    if encode_ethnicity:
        df['ethnicity'] = df.ethnicity.map(ETHNICITY_ENCODING)
    df = df.reindex(df.index.repeat(df['number_of_observations']))
    print(df['taken_to_court'].unique())
    return df.sample(frac=1)


class AnnualApprehensionsDataset(StandardDataset):
    default_privileged_groups = [{'ethnicity': 0}]

    default_unprivileged_groups = [{'ethnicity': 1}, {'ethnicity': 2}, {'ethnicity': 3}, {'ethnicity': 4},
                                   {'ethnicity': 5}]

    def __init__(
        self,
        label_name='taken_to_court',
        favorable_classes=[0],
        protected_attribute_names=['ethnicity'],
        privileged_classes=[[0]],  # From ethnicity_cat_map, European
        instance_weights_name=None,
        categorical_features=['main_offence', 'age_group', 'gender'],
        features_to_keep=[],
        features_to_drop=['location', 'year', 'resolution', 'number_of_observations', 'flags'],
        na_values=[],
        custom_preprocessing=default_preprocessing,
        metadata={
            'label_maps': [{1: 'Taken to Court', 0: 'Not Taken to Court'}],
            'protected_attribute_maps': [
                {
                    'Caucasian': 0,
                    'Asiatic': 1,
                    'Maori': 2,
                    'Other Ethnicities': 3,
                    'Pacific Isle': 4,
                    'Indian': 5
                }
            ]
        }
    ):
        file = common.decompress_bz2_file(common.DatasetsFiles.ANNUAL_APPREHENSIONS.value)
        df = pd.read_csv(file, sep='|')

        super(AnnualApprehensionsDataset, self).__init__(df=df, label_name=label_name,
                                                         favorable_classes=favorable_classes,
                                                         protected_attribute_names=protected_attribute_names,
                                                         privileged_classes=privileged_classes,
                                                         instance_weights_name=instance_weights_name,
                                                         categorical_features=categorical_features,
                                                         features_to_keep=features_to_keep,
                                                         features_to_drop=features_to_drop,
                                                         na_values=na_values,
                                                         custom_preprocessing=custom_preprocessing,
                                                         metadata=metadata
                                                         )


def plots_annual_apprehension():
    file = common.decompress_bz2_file(common.DatasetsFiles.ANNUAL_APPREHENSIONS.value)
    df = pd.read_csv(file, sep='|')
    df = default_preprocessing(df, encode_ethnicity=False)
    df = df.sort_values(by='ethnicity')
    plt.figure(figsize=(10, 6))
    plt.hist(df['ethnicity'], bins=10, edgecolor='black')  # Adjust the number of bins as needed
    plt.title('Annual Apprehensions: Ethnic Group Distribution')
    plt.xlabel('Ethnic Group')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

    count_is_free = df['taken_to_court'].value_counts()
    count_is_free.plot(kind='bar', figsize=(8, 6))
    plt.title("Annual Apprehensions: Number of Records for Taken to Court (True vs. False)")
    plt.xlabel("Taken to Court")
    plt.ylabel("Number of Records")
    plt.show()
