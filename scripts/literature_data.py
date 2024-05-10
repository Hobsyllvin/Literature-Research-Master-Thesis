import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times"],  # or 'Times', 'Palatino', etc.
})

def plot_voronoi(kmeans, ax, data):
    # Create Voronoi diagram
    vor = Voronoi(kmeans.cluster_centers_)
    
    # Plot Voronoi diagram
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black')
    
    # Scatter plot of the data points
    ax.scatter(data[:, 0], data[:, 1], s=30, c='#000000')
    
    # Roman numerals for labeling clusters from I to VIII
    roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']
    # Annotating each cluster center with a Roman numeral
    for i, center in enumerate(final_kmeans.cluster_centers_):
        plt.text(center[0], center[1]+0.2, roman_numerals[i], fontsize=14, ha='center', va='center', color='black')

    # Set plot limits
    ax.set_xlim(data[:, 0].min()-1, data[:, 0].max()+1)
    ax.set_ylim(data[:, 1].min()-1, data[:, 1].max()+1)
    ax.set_xlabel('Haptic Fidelity Scores', fontsize = 14)
    ax.set_ylabel('Versatility Scores', fontsize = 14)

    ax.set_axisbelow(True)
    ax.grid()


def optimal_kmeans(data, max_k=10):
    silhouette_scores = []
    range_n_clusters = range(2, max_k + 1)
    
    for n_clusters in tqdm(range_n_clusters, desc='Calculating clusters'):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, the average silhouette score is {silhouette_avg:.4f}")
    
    optimal_n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
    print(f"The optimal number of clusters is {optimal_n_clusters}.")
    
    # Plot the silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.grid(True)
    plt.savefig('silhouette.pdf', format='pdf', bbox_inches='tight')

    plt.show()

    return optimal_n_clusters

# Load data
file_path = '/Users/christian/Documents/Literature-Research-Master-Thesis/literature_data.csv'
data = pd.read_csv(file_path)

# Impute missing values using the mean of each column
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)
data = pd.DataFrame(data_imputed, columns=data.columns)

# Assuming the first two columns are the features we want to cluster
X = data.iloc[:, :2].values

# Find the optimal number of clusters
optimal_clusters = optimal_kmeans(X)



# Perform KMeans clustering with the optimal number of clusters
final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
final_kmeans.fit(X)
data['Cluster'] = final_kmeans.labels_

# Plotting the clusters with Voronoi regions
fig, ax = plt.subplots(figsize=(8, 8))
plot_voronoi(final_kmeans, ax, X)
plt.savefig('literature_data.pdf', format='pdf', bbox_inches='tight')
plt.show()
