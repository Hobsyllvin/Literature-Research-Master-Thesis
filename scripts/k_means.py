import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
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
    roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']
    
    # Annotating each cluster center with a Roman numeral
    plt.text(1.38, 3.2, 'I', fontsize=14, ha='center', va='center', color='black')
    plt.text(2.6, 4.2, 'II', fontsize=14, ha='center', va='center', color='black')
    plt.text(2.67, 3.2, 'III', fontsize=14, ha='center', va='center', color='black')
    plt.text(3.42, 3.2, 'IV', fontsize=14, ha='center', va='center', color='black')
    plt.text(2.68, 2.2, 'V', fontsize=14, ha='center', va='center', color='black')
    plt.text(3.22, 2.2, 'VI', fontsize=14, ha='center', va='center', color='black')
    plt.text(3.67, 2.2, 'VII', fontsize=14, ha='center', va='center', color='black')
    plt.text(3.54, 1.2, 'VIII', fontsize=14, ha='center', va='center', color='black')
    plt.text(3.61, 0.2, 'IX', fontsize=14, ha='center', va='center', color='black')
    plt.text(2.52, 1.2, 'X', fontsize=14, ha='center', va='center', color='black')
    plt.text(1.01, 2.2, 'XI', fontsize=14, ha='center', va='center', color='black')

    # Set plot limits
    ax.set_xlim(data[:, 0].min()-0.4, data[:, 0].max()+0.4)
    ax.set_ylim(data[:, 1].min()-0.4, data[:, 1].max()+0.4)
    ax.set_xlabel('Haptic Fidelity Scores', fontsize = 14)
    ax.set_ylabel('Versatility Scores', fontsize = 14)

    ax.set_axisbelow(True)
    ax.grid()


def plot_silhouette(data, n_clusters):
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(9, 6)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    
    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=14)

        y_lower = y_upper + 10

    ax1.set_title(f"k={n_clusters}", fontsize=14)
    ax1.set_xlabel("Silhouette Coefficient", fontsize=14)
    ax1.set_ylabel("Cluster Label", fontsize=14)

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

    filename = f"figures/kmeans_clusters_{n_clusters}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()


def optimal_kmeans(data, max_k=10):
    silhouette_scores = []
    range_n_clusters = range(2, max_k + 1)
    
    silhouette_max = 0
    count = 0
    optimal_n_clusters = 2
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        if silhouette_avg > silhouette_max:
            silhouette_max = silhouette_avg
            count = 0
            optimal_n_clusters = n_clusters
        else:
            count += 1
            if count == 5:
                print("Best score reached, aborting clustering")
                break
        
        print(f"For n_clusters = {n_clusters}, the average silhouette score is {silhouette_avg:.4f}")
    
    print(f"The optimal number of clusters is {optimal_n_clusters}.")
    
    # Plot the silhouette scores
    plt.figure(figsize=(7, 4))
    plt.plot(range(2, 2 + len(silhouette_scores)), silhouette_scores, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Clusters", fontsize=14)
    plt.ylabel("Silhouette Score", fontsize=14)
    plt.grid(True)
    plt.savefig('figures/silhouette_9.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    return optimal_n_clusters

# Load data
file_path = '/Users/christian/Documents/Literature-Research-Master-Thesis/figures/literature_data.csv'
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
plt.savefig('figures/k_means_9.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plot silhouette diagrams for various values of k
#for n_clusters in range(2, optimal_clusters + 1):
#    plot_silhouette(X, n_clusters)
    