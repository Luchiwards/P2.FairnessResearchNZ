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
        file = common.decompress_bz2_file(common.DatasetsFiles.DS_POLICE_TWO.value)
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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Sample dataframe
    np.random.seed(42)  # For reproducibility
    data = {
        'value': np.random.uniform(0, 2, 30),  # Generating random values between 0 and 2
        'ethnicity': np.random.choice(['EthnicityA', 'EthnicityB', 'EthnicityC'], 30),
        'offense': np.random.choice(['Offense1', 'Offense2', 'Offense3', 'Offense4'], 30)
    }

    df = pd.DataFrame(data)

    # Symbol dictionary for each ethnicity
    symbol_dict = {
        'EthnicityA': 'o',  # Circle
        'EthnicityB': 's',  # Square
        'EthnicityC': '^'  # Triangle up
    }

    sizes = df.groupby(['offense', 'ethnicity']).size().reset_index(name='count')
    df = pd.merge(df, sizes, on=['offense', 'ethnicity'])
    df['size'] = df['count'] * 50

    # Plotting
    plt.figure(figsize=(10, 6))
    for ethnicity, symbol in symbol_dict.items():
        subset = df[df['ethnicity'] == ethnicity]
        plt.scatter(subset['value'], subset['offense'], marker=symbol, label=ethnicity, s=subset['size'],
                    edgecolors='k', alpha=0.7)
        # plt.scatter(subset['value'], subset['offense'], marker=symbol, label=ethnicity, s=100, edgecolors='k')
    plt.axvline(x=1, color='red', linestyle='--', label='x=1')
    plt.xlabel('Value')
    plt.ylabel('Offense')
    plt.title('Value by Offense and Ethnicity')
    plt.legend()
    plt.tight_layout()
    plt.show()


def rows_analysis():
    # pd.set_option('display.max_columns', None)
    file = common.decompress_bz2_file(common.DatasetsFiles.DS_POLICE_TWO.value)
    df = pd.read_csv(file)
    df = default_preprocessing(df)
    df.info()
    # plots_police()


if __name__ == "__main__":
    rows_analysis()
