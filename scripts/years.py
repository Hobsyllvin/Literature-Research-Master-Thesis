import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from io import StringIO

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ['Times', 'Times New Roman', 'Liberation Serif'],  # Fallback options
})

# Simulating reading the CSV file 'years.csv'
# Assuming the file content is in the same format as the provided data
csv_data = """Year,Amount
2000,1
2001,0
2002,1
2003,0
2004,0
2005,0
2006,1
2007,2
2008,2
2009,0
2010,1
2011,0
2012,2
2013,0
2014,3
2015,1
2016,0
2017,1
2018,0
2019,5
2020,2
2021,2
2022,3
2023,7
2024,1"""

# Read the data into a pandas DataFrame
df = pd.read_csv(StringIO(csv_data))

# Convert 'Year' to datetime to ensure proper grouping
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Set 'Year' as the index
df.set_index('Year', inplace=True)

# Reset index to have 'Year' as a column again for plotting
df.reset_index(inplace=True)

# Extract the year from datetime for x-axis labels
df['Year'] = df['Year'].dt.year

# Plotting every year
plt.figure(figsize=(2.3*1.2, 1.85*1.2))
plt.bar(df['Year'].astype(str), df['Amount'], width=0.8)

# Setting the x-axis labels to show only every 5 years
xticks = range(2000, 2025, 5)  # 2000, 2005, 2010, etc.
plt.xticks([str(year) for year in xticks])  # Convert to string for plotting

# Setting labels and title
plt.xlabel('Year of publication')
plt.ylabel('Number of studies')
plt.title('')

plt.tight_layout()  # Adjust layout
plt.savefig("C:/Users/chris/Documents/Thesis/LiteratureResearch/figures/years.pdf", format="pdf", bbox_inches="tight")
plt.show()


