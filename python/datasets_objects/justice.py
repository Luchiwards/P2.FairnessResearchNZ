import json
import os
import random
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from aif360 import datasets
from matplotlib import pyplot as plt
from sklearn.utils import resample

from python.helpers import common

random.seed(42)
ETHNICITIES = ['European', 'Asian', 'Maori', 'Other', 'Pacific Peoples']

ETHNICITY_ENCODING = {ethnicity: index for index, ethnicity in enumerate(ETHNICITIES)}

FOLDER = f"../datasets/justice/"


def default_preprocessing(df: pd.DataFrame, balanced=True):
    """Perform the same preprocessing as the original analysis:
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    """

    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    filename = f"{FOLDER}raw_balanced.parquet.gzip"

    if os.path.exists(filename):
        df = pd.read_parquet(filename)
        print(f"Data loaded from {filename}")
    else:
        df.sample(frac=1).reset_index(drop=True)
        df.rename(
            columns={
                'Main offence': 'main_offence',
                'Ethnicity': 'ethnicity_cat',
                'Calendar year': 'years',
                'Sentence': 'sentence',
                'Value': 'number_of_observations',
                'Gender': 'gender',
                'Age group': 'age_group',
            },
            inplace=True
        )
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        offences_groups = df[df['MAIN_OFFENCE'].str.len() == 2][['MAIN_OFFENCE', 'main_offence']].drop_duplicates(
            subset=['main_offence', 'MAIN_OFFENCE']
        )
        offences_groups.rename(
            columns={
                'MAIN_OFFENCE': 'offence_division_code',
                'main_offence': 'main_offence_group',
            },
            inplace=True
        )

        # groups of offences 11, 13 has just one level
        # then, we are not filtering those out
        keep_offences_two_levels = ['11', '13']
        drop_offences_third_levels = {code[0:3] for code in set(df[df['MAIN_OFFENCE'].str.len() == 4]['MAIN_OFFENCE'])}

        df = df[
            (df['MAIN_OFFENCE'].str.len() != 2) | (df['MAIN_OFFENCE'].isin(keep_offences_two_levels))
            ]

        # The offences that has four level must be cleaned in its upper level to avoid duplicity of data.
        df = df[~df['MAIN_OFFENCE'].isin(drop_offences_third_levels)]

        df['offence_division_code'] = df['MAIN_OFFENCE'].str[0:2]
        df = pd.merge(df, offences_groups, on='offence_division_code')

        remain_offences = df[
            ['main_offence_group', 'offence_division_code', 'main_offence', 'MAIN_OFFENCE']
        ].drop_duplicates(
            subset=['main_offence_group', 'offence_division_code', 'main_offence', 'MAIN_OFFENCE']).sort_values(
            by='MAIN_OFFENCE')[['main_offence', 'MAIN_OFFENCE']]

        json_structure = remain_offences.set_index('MAIN_OFFENCE')['main_offence'].to_dict()
        with open(f'{FOLDER}/offences_before_balancing_dataset.json', 'w') as outfile:
            json.dump(json_structure, outfile)

        df = df[df['ethnicity_cat'] != 'Unknown']
        df['incarcerated'] = np.where(df['sentence'].str.contains('Imprisonment', case=False), 1, 0)
        df = df[df['gender'] != 'Unknown/Organisation']
        df = df[df['age_group'] != 'Unknown/Organisation']
        df = df.reindex(df.index.repeat(df['number_of_observations']))
        df['ethnicity'] = df.ethnicity_cat.map(ETHNICITY_ENCODING)
        if balanced:
            balance_strategy = get_unbalance_data_strategy(df)
            df = balance_process(df, balance_strategy=balance_strategy)
        # Protected attributes must be converted to ordinal in a new column
        # https://aif360.readthedocs.io/en/stable/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html
        df.to_parquet(filename, compression='gzip')
        print(f"Data saved to {filename}")
        remain_offences = df[
            ['main_offence_group', 'offence_division_code', 'main_offence', 'MAIN_OFFENCE']
        ].drop_duplicates(
            subset=['main_offence_group', 'offence_division_code', 'main_offence', 'MAIN_OFFENCE']).sort_values(
            by='MAIN_OFFENCE')[['main_offence', 'MAIN_OFFENCE']]
        json_structure = remain_offences.set_index('MAIN_OFFENCE')['main_offence'].to_dict()
        with open(f'{FOLDER}/offences_after_balancing_dataset.json', 'w') as outfile:
            json.dump(json_structure, outfile)

    return df


# def default_postprocessing(train: StandardDataset, test: StandardDataset, privileged_groups: List[str],
#                            unprivileged_groups: List[str]):
#     cpp = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
#                                          unprivileged_groups=unprivileged_groups,
#                                          cost_constraint='weighted',
#                                          seed=np.random.seed())
#     cpp = cpp.fit(train, test)
#
#     # Transform the predictions to enhance fairness
#     fair_test_preds = cpp.predict(test)
#
#     # Evaluate performance and fairness
#     fair_test_metric = ClassificationMetric(train,
#                                             fair_test_preds,
#                                             unprivileged_groups=unprivileged_groups,
#                                             privileged_groups=privileged_groups)
#
#     accuracy = accuracy_score(train.labels, fair_test_preds.labels)
#     disp_impact = fair_test_metric.disparate_impact()
#     avg_odd_diff = fair_test_metric.average_odds_difference()


class JusticeDataset(datasets.StandardDataset):
    default_privileged_groups = [{'ethnicity': ETHNICITY_ENCODING.get('European')}]

    default_unprivileged_groups = [{'ethnicity': v} for k, v in ETHNICITY_ENCODING.items() if k != 'European']

    def __init__(
        self,
        label_name='incarcerated',
        favorable_classes=[0],
        protected_attribute_names=['ethnicity'],
        privileged_classes=[[ETHNICITY_ENCODING.get('European')]],  # From ethnicity_cat_map, European
        instance_weights_name=None,
        categorical_features=['main_offence', 'age_group', 'gender', 'ethnicity_cat'],
        features_to_keep=[],
        features_to_drop=[
            'sentence',
            'flags',
            'court',
            'years',
            'number_of_observations',
            "Court",
            "Flags",
            "MAIN_OFFENCE",
            "COURT_CLUSTER",
            "AGE_GROUP",
            "GENDER",
            "ETHNICITY",
            "YEAR",
            "SENTENCE",
            "offence_division_code",
            "main_offence_group",
        ],
        na_values=[],
        custom_preprocessing=default_preprocessing,
        metadata={
            'label_maps': [{1: 'Incarcerated', 0: 'Not Incarcerated'}],
            'protected_attribute_maps': ETHNICITY_ENCODING
        }
    ):
        filtered_paths = [
            file_path.value for file_path in common.DatasetsFiles
            if 'justice-coded-' in file_path.value
        ]
        dfs = [pd.read_csv(common.decompress_bz2_file(path), dtype={'MAIN_OFFENCE': str}) for path in filtered_paths]
        df = pd.concat(dfs, ignore_index=True)

        super(JusticeDataset, self).__init__(df=df, label_name=label_name,
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


def get_unbalance_data_strategy(df) -> List[Dict[str, Any]]:
    offences = df['main_offence'].unique()
    results = {offence: {
        'offence': offence,
        True: 0,
        False: 0
    } for offence in offences}
    for offence in offences:
        subset = df[df['main_offence'] == offence]
        count_incarcerated = subset['incarcerated'].value_counts()

        unique_values = count_incarcerated.index.tolist()
        counts = count_incarcerated.values.tolist()
        paired_values_and_counts = list(zip(unique_values, counts))

        if len(paired_values_and_counts) == 2:
            pair_one, pair_two = paired_values_and_counts
        else:
            pair_one = paired_values_and_counts[0]
            pair_two = (not pair_one[0], 0)

        results[offence][pair_one[0]] = pair_one[1]
        results[offence][pair_two[0]] = pair_two[1]

    results = [
        {
            'offence': value['offence'],
            'not_incarcerated': value[False],
            'incarcerated': value[True],
            'balance_ratio': round(
                (max(min(value[True], value[False]), 1) / (value[True] + value[False])) * 100, 4),
            'total': value[True] + value[False],
            'remove': False,
            'resampling': False,
        }
        for value
        in results.values()
    ]

    results = [
        {
            **values,
            "resampling": False,
            "remove": ((values['balance_ratio'] < 25) or (values['total'] < 100)),
        } for values in results
    ]
    return results


def balance_process(dataset: pd.DataFrame, balance_strategy: List[Dict[str, Any]]) -> pd.DataFrame:
    print("BALANCING DATASET BY OFFENCE")
    amount_undersampled = 0

    offences_to_keep = [
        set_of_data.get('offence') for set_of_data in balance_strategy if
        not set_of_data.get('remove')
    ]

    balance_strategy = [
        information for information in balance_strategy if
        not information.get('remove')
    ]

    dataset = dataset[dataset['main_offence'].isin(offences_to_keep)]

    plots_balance_justice(df=dataset, balance=False)
    plots_ethnicity_justice(df=dataset, balance=False)
    plot_balance_justice_ethnicity(df=dataset, balance=False)
    for set_to_balance in balance_strategy:
        resampling = set_to_balance.get('resampling', False)
        offence = set_to_balance.get('offence')

        if resampling:
            dataset_others = dataset[dataset['main_offence'] != offence]
            dataset_target = dataset[dataset['main_offence'] == offence]

            majority_class = 1 if set_to_balance['incarcerated'] > set_to_balance['not_incarcerated'] else 0
            minority_class = abs(1 - majority_class)

            df_majority = dataset_target[dataset_target.incarcerated == majority_class]
            df_minority = dataset_target[dataset_target.incarcerated == minority_class]

            df_majority_downsampled = resample(
                df_majority,
                replace=True,  # sample without replacement
                n_samples=(round(len(df_minority) * 1.4)),  # to match minority class
                random_state=123
            )  # reproducible results
            dataset = pd.concat([dataset_others, df_majority_downsampled, df_minority])

            amount_undersampled = amount_undersampled + len(df_majority) - len(df_majority_downsampled)

            print(
                f'Offence: {offence}, Majority Class: {majority_class}, Minority Count: {len(df_minority)}, Majority Count: {len(df_majority)}, Majority Downsampled: {len(df_majority_downsampled)}'
            )
    plots_balance_justice(df=dataset, balance=True)
    plots_ethnicity_justice(df=dataset, balance=True)
    plot_balance_justice_ethnicity(df=dataset, balance=True)
    return dataset


def plots_ethnicity_justice(df, balance: bool):
    name = 'Balanced' if balance else 'Unbalanced'
    name = f'Incarceration Counts {name} Ethnicities Dataset'
    df = df.sort_values(by='ethnicity_cat')
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.hist(df['ethnicity_cat'], bins=10, edgecolor='black')  # Adjust the number of bins as needed
    plt.title(name)
    plt.xlabel('Ethnic Group')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(f"{FOLDER}{'_'.join(name.split(' ')).lower()}.png", dpi=80)
    plt.show()


def plots_balance_justice(df, balance: bool):
    name = 'Balanced' if balance else 'Unbalanced'
    name = f'Incarceration Counts {name} Dataset'
    count_encarcerated = df['incarcerated'].value_counts()
    count_encarcerated.plot(kind='bar', figsize=(8, 6))
    plt.xlabel("Justice: Incarcerated")
    plt.ylabel("Number of Records")
    plt.title(name)
    plt.savefig(f"{FOLDER}{'_'.join(name.split(' ')).lower()}.png", dpi=80)
    plt.show()


def plot_balance_justice_ethnicity(df, balance: bool):
    name = 'Balanced' if balance else 'Unbalanced'
    name = f'Incarceration Counts {name} Dataset by Ethnicity'
    cross_tab = pd.crosstab(df['ethnicity_cat'], df['incarcerated'])
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Twin the axes
    ax2 = ax1.twinx()

    # Plotting count (bar chart)
    cross_tab.plot(kind='bar', ax=ax1, position=1, width=0.4)

    # Adjusting labels and title
    ax1.set_xlabel('Ethnicity')
    ax1.set_ylabel('Counts')
    ax2.set_ylabel('Percentage Incarcerated (%)')
    ax1.set_title(name)

    # Displaying the legend
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f"{FOLDER}{'_'.join(name.split(' ')).lower()}.png", dpi=80)
    plt.show()


def main():
    filtered_paths = [
        file_path.value for file_path in common.DatasetsFiles
        if 'justice-coded-' in file_path.value
    ]
    dfs = [pd.read_csv(common.decompress_bz2_file(path), dtype={'MAIN_OFFENCE': str}) for path in filtered_paths]
    df = pd.concat(dfs, ignore_index=True)

    df = default_preprocessing(df, balanced=True)
    df.info()


if __name__ == "__main__":
    main()
