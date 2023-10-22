"""
    Print out all the column headers in the file that's in data/justice-1980-1999.csv.bz2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Using bz2 library to read compressed file
import bz2

# Read in the file
justice_file = bz2.open('data/Ethnicity AES_Full Data_data.csv.bz2', 'rt')

# Read in the CSV file, it is separated by a "|" value instead of a ","
justice = pd.read_csv(justice_file)

# Print out the column headers
# print(justice.columns)
for col in justice.columns:
    print(col)

# The columns are:
"""
Court
Main offence
Age group
Gender
Ethnicity
Calendar year
Sentence
Value
Flags
"""

# I want to see how many people were convicted of each offence, and also a grap