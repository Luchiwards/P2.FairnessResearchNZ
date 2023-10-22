import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
from matplotlib import pyplot as plt

from python.helpers import common

ETHNICITIES = ['European', 'Asian', 'Maori', 'Other Ethnicities', 'Pacific Island', 'Indian']

ETHNICITY_ENCODING = {ethnicity: index for index, ethnicity in enumerate(ETHNICITIES)}


def default_preprocessing(df: pd.DataFrame):
    """Perform the same preprocessing as the original analysis:
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    """
    df.columns = ["_".join(header.split(' ')).lower() for header in df.columns]
    df.rename(
        columns={
            'ethnic_group': 'ethnicity_cat',
            'sex': 'gender',
            'anzsoc_division': 'main_offence',
            'person/organisation': 'person_organisation',
        }, inplace=True
    )
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df["year_month"] = df["year_month"].str[-4:]
    df["year"] = pd.to_numeric(df['year_month'])
    df['no_court'] = np.where(df["method_of_proceeding"] == 'Court Action', 0, 1)
    df["ethnicity_cat"] = np.where(
        df["ethnicity_cat"] == 'Not Stated',
        'Other Ethnicities',
        df["ethnicity_cat"]
    )
    df = df[df['main_offence'] != 'Traffic and Vehicle Regulatory Offences']
    df['ethnicity'] = df.ethnicity_cat.map(ETHNICITY_ENCODING)
    # Protected attributes must be converted to ordinal in a new column
    # https://aif360.readthedocs.io/en/stable/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html
    df = df.reindex(df.index.repeat(df['proceedings']))
    return df


class PoliceDataset(StandardDataset):
    default_privileged_groups = [{'ethnicity': ETHNICITY_ENCODING.get('European')}]

    default_unprivileged_groups = [{'ethnicity': v} for k, v in ETHNICITY_ENCODING.items() if k != 'European']

    def __init__(
        self,
        label_name='no_court',
        favorable_classes=[1],
        protected_attribute_names=['ethnicity'],
        privileged_classes=[[ETHNICITY_ENCODING.get('European')]],  # From ethnicity_cat_map, European
        instance_weights_name=None,
        categorical_features=['main_offence', 'age_group', 'gender', 'ethnicity_cat'],
        features_to_keep=[],
        features_to_drop=['method_of_proceeding', 'person_organisation', 'year_month', 'proceedings', 'year'],
        na_values=[],
        custom_preprocessing=default_preprocessing,
        metadata={
            'label_maps': [{1: 'No Court', 0: 'Court'}],
            'protected_attribute_maps': ETHNICITY_ENCODING
        }
    ):
        file = common.decompress_bz2_file(common.DatasetsFiles.DS_POLICE.value)
        df = pd.read_csv(file)

        super(PoliceDataset, self).__init__(df=df, label_name=label_name,
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


def plots_police():
    file = common.decompress_bz2_file(common.DatasetsFiles.DS_POLICE.value)
    df = pd.read_csv(file)
    df = default_preprocessing(df)
    df = df.sort_values(by='ethnicity_cat')
    plt.figure(figsize=(10, 6))
    plt.hist(df['ethnicity_cat'], bins=10, edgecolor='black')  # Adjust the number of bins as needed
    plt.title('Police: Ethnic Group Distribution')
    plt.xlabel('Ethnic Group')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

    count_is_free = df['no_court'].value_counts()
    count_is_free.plot(kind='bar', figsize=(8, 6))
    plt.xlabel("Policy: Court")
    plt.ylabel("Number of Records")
    plt.title("Number of Records for Incarcerated (True vs. False)")
    plt.show()
