import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Using bz2 library to read compressed file
import bz2

# Read in the file
justice_file = bz2.open('data/Ethnicity AES_Full Data_data.csv.bz2', 'rt')
justice = pd.read_csv(justice_file)

column_headers = [
    "Ethnic Group",
    "Age Group",
    "SEX",
    "ANZSOC Division",
    "Method of Proceeding",
    "Year Month",
    "Proceedings"
]


anzsoc_division = "Robbery, Extortion and Related Offences"
age_group = "35-39"

# Delete Person/Organisation column
del justice["Person/Organisation"]

# Filter the data by the anzsoc_division
robbery = justice[justice["ANZSOC Division"] == anzsoc_division]
age_group = robbery[robbery["Age Group"] == age_group]
# Filter Year Month so that it only shows the year
age_group['Year Month'] = pd.to_datetime(age_group['Year Month'], dayfirst=True)
age_group['Year Month'] = age_group['Year Month'].dt.year

columns_to_group_by = [    "Ethnic Group",
    "Age Group",
    "SEX",
    "ANZSOC Division",
    "Method of Proceeding",
    "Year Month"]

grouped_data = age_group.groupby(columns_to_group_by)['Proceedings'].sum().reset_index()

# Compare the number of Maori people vs Non-Maori convicted of robbery in the 35-39 age group
maori = grouped_data[grouped_data["Ethnic Group"] == "Maori"]
non_maori = grouped_data[grouped_data["Ethnic Group"] != "Maori"]

# Generate a bar chart for all the different types of methods of proceeding