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
    "font.serif": ["Times"],  # or 'Times', 'Palatino', etc.
})

# Sum the occurrences of each body part
body_part_counts = df.drop(columns='Paper').sum()

# Define colors for the pie chart
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6']


def autopct_generator(pct, allvalues):
    absolute = int(round(pct/100.*sum(allvalues)))
    return '{:.1f}%\n({:d})'.format(pct, absolute)# if pct > 5 else ''  # Display values only if larger than 5%
# Create a pie chart with multiple levels
fig, ax = plt.subplots()

wedges, texts, autotexts = ax.pie(body_part_counts, labels=body_part_counts.index, autopct=lambda pct: autopct_generator(pct, body_part_counts), startangle=90, counterclock=False, colors=colors)

for text in texts:
    text.set_fontsize(14)

for autotext in autotexts:
    autotext.set_fontsize(12)

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart
plt.savefig('/Users/christian/Documents/Literature-Research-Master-Thesis/figures/body_pie.pdf', format='pdf')
plt.show()
