import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a pandas DataFrame
data = {
    'Paper': range(1, 37),
    'Palm': [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
    'Fingers': [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    'Feet': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'Forearm': [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    'Entire Arm': [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    'Entire Body': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ['Times', 'Times New Roman', 'Liberation Serif'],  # Fallback options
})

# Sum the occurrences of each body part
body_part_counts = df.drop(columns='Paper').sum()

# Define colors for the pie chart
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6']

def autopct_generator(pct, allvalues):
    absolute = int(round(pct/100.*sum(allvalues)))
    return '{:.1f}%\n({:d})'.format(pct, absolute)

# Create a pie chart with improved spacing and no overlapping texts
fig, ax = plt.subplots()

wedges, texts, autotexts = ax.pie(
    body_part_counts,
    labels=body_part_counts.index,
    autopct=lambda pct: autopct_generator(pct, body_part_counts),
    startangle=90,
    counterclock=False,
    colors=colors,
    pctdistance=0.7,  # Adjusting this to avoid overlap
    labeldistance=1.1  # Adjusting this to position labels further from the pie
)

# Adjust font sizes and layout to avoid overlaps
for text in texts:
    text.set_fontsize(22)
for autotext in autotexts:
    autotext.set_fontsize(20)

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Use tight_layout to optimize space and ensure nothing overlaps
plt.tight_layout()

# Display the pie chart
plt.savefig('C:/Users/chris/Documents/Thesis/LiteratureResearch/figures/body_pie.pdf', format='pdf')
plt.show()
