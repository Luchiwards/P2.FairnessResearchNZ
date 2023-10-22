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
    "Person/Organisation",
    "Year Month",
    "Proceedings"
]

justice['Year Month'] = pd.to_datetime(justice['Year Month'], dayfirst=True)

# Filter the data for Maori and Non-Maori individuals
maori_data = justice[justice['ï»¿Ethnic Group'] == 'Maori']
non_maori_data = justice[justice['ï»¿Ethnic Group'] != 'Maori']

# Group and sum the proceedings for both groups by month
maori_monthly_proceedings = maori_data.groupby('Year Month')['Proceedings'].sum()
non_maori_monthly_proceedings = non_maori_data.groupby('Year Month')['Proceedings'].sum()
"""# Sort the data by date
maori_monthly_proceedings.sort_index(inplace=True)
non_maori_monthly_proceedings.sort_index(inplace=True)"""

# Plot the time-series distributions for both groups
plt.figure(figsize=(10, 6))

# plt.plot(monthly_proceedings.index, monthly_proceedings.values, label='Total Proceedings', linewidth=2)
plt.plot(maori_monthly_proceedings.index, maori_monthly_proceedings.values, label='Maori', linestyle='dashed')
plt.plot(non_maori_monthly_proceedings.index, non_maori_monthly_proceedings.values, label='Non-Maori', linestyle='dashed')

plt.xlabel('Date')
plt.ylabel('Total Proceedings')
plt.title('Monthly Proceedings Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
