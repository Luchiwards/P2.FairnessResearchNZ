"""
To get the data for this database
1. go Url: https://www.police.govt.nz/about-us/publications-statistics/data-and-statistics/policedatanz/proceedings-offender-demographics
2. in download, download the version for tableau.
3. Open it in tableau
4. In the tab "Demographics", doing right-click over any of its graphs, select "View Data..."
5. In the left side, in Tabs, click in "Full Data"
6. Click "show fields" at the right-top and select the fields

Where to get tableau?
https://www.tableau.com/academic/students
"""
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset

from python.helpers import common

ETHNICITIES = [
    'Pacific Island',
    'Maori',
    'European',
    'African',
    'Latin American/Hispanic',
    'Middle Eastern',
    'Indian',
    'Asian',
]

ETHNICITY_ENCODING = {ethnicity: index for index, ethnicity in enumerate(ETHNICITIES)}


def default_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Perform the same preprocessing as the original analysis:
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    """
    df.columns = ["_".join(header.split('.')).lower() for header in df.columns]
    df.columns = ["_".join(header.split(' ')).lower() for header in df.columns]
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.rename(
        columns={
            'ethnicity': 'ethnicity_cat',
            'sex': 'gender',
            'anzsoc_group': 'main_offence',
            'person/organisation': 'person_organisation',
        }, inplace=True
    )
    df = df[df['ethnicity_cat'] != 'Organisation']
    df = df[df['ethnicity_cat'] != 'Not Stated']
    df = df[df['ethnicity_cat'] != 'Not Elsewhere Classified']
    df = df[df['ethnicity_cat'] != 'Other Ethnicities']

    df['year_month'] = pd.to_datetime(df['year_month'], format='%b-%y')
    # Extract only the year
    df['year'] = df['year_month'].dt.year
    df['court'] = np.where(df["method_of_proceeding"] == 'Court Action', 1, 0)
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
            label_name='court',
            favorable_classes=[0],
            protected_attribute_names=['ethnicity'],
            privileged_classes=[[ETHNICITY_ENCODING.get('European')]],  # From ethnicity_cat_map, European
            instance_weights_name=None,
            categorical_features=['main_offence', 'age_group', 'gender', 'ethnicity_cat'],
            features_to_keep=[],
            features_to_drop=[
                'anzsoc_division',
                'anzsoc_subdivision',
                'main_offence_group',
                'method_of_proceeding',
                'person_organisation',
                'year_month',
                'proceedings',
                'year',
            ],
            na_values=[],
            custom_preprocessing=default_preprocessing,
            metadata={
                'label_maps': [{1: 'Court', 0: 'Not Court'}],
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
