import json
import os
import random
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from aif360 import datasets

from toolkit.helpers import common

random.seed(42)
ETHNICITIES = ['European', 'Asian', 'Maori', 'Other', 'Pacific Peoples']

ETHNICITY_ENCODING = {ethnicity: index for index, ethnicity in enumerate(ETHNICITIES)}

FOLDER = f"toolkit/datasets/justice/"


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
        remain_offences = df[
            ['main_offence_group', 'offence_division_code', 'main_offence', 'MAIN_OFFENCE']
        ].drop_duplicates(
            subset=['main_offence_group', 'offence_division_code', 'main_offence', 'MAIN_OFFENCE']).sort_values(
            by='MAIN_OFFENCE')[['main_offence', 'MAIN_OFFENCE']]
        json_structure = remain_offences.set_index('MAIN_OFFENCE')['main_offence'].to_dict()
        with open(f'{FOLDER}/offences_after_balancing_dataset.json', 'w') as outfile:
            json.dump(json_structure, outfile)

    return df


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
        }
        for value
        in results.values()
    ]

    results = [
        {
            **values,
            "remove": ((values['balance_ratio'] < 25) or (values['total'] < 100)),
        } for values in results
    ]
    return results


def balance_process(dataset: pd.DataFrame, balance_strategy: List[Dict[str, Any]]) -> pd.DataFrame:
    print("BALANCING DATASET BY OFFENCE")
    offences_to_keep = [
        set_of_data.get('offence') for set_of_data in balance_strategy if
        not set_of_data.get('remove')
    ]
    return dataset[dataset['main_offence'].isin(offences_to_keep)]
