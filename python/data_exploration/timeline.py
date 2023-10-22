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

# Convert "Year Month" column to datetime format
justice['Year Month'] = pd.to_datetime(justice['Year Month'], dayfirst=True)

# Aggregate data by month and sum the "Proceedings" column
monthly_proceedings = justice.groupby('Year Month')['Proceedings'].sum()

# Plot the time-series distribution
plt.figure(figsize=(10, 6))
plt.plot(monthly_proceedings.index, monthly_proceedings.values, marker='o')
plt.title('Monthly Proceedings Over Time')
plt.xlabel('Date')
plt.ylabel('Total Proceedings')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()