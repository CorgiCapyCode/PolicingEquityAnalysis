import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def clustering(df: pd.DataFrame):
    ohe_df = one_hot_encoding(df=df)
    # ohe_df.info()
    

    k_clusters_all = kmeans_cluster_all(df=ohe_df)
    visualization_of_clusters(df=ohe_df, clusters=k_clusters_all)
    
    return ohe_df


def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    ohe_df = df.copy()
    # Converting the multi-level column index to one level & ensure all are of type str
    ohe_df.columns = ["_".join(map(str, col)) for col in df.columns]

    # Drop the first column containing the IDs   
    ohe_df.drop(columns=ohe_df.columns[0], inplace=True)
    
    ohe_df = pd.get_dummies(ohe_df)
    # Ensures that all columns are floats
    ohe_df = ohe_df.astype(float)
    return ohe_df


def kmeans_cluster_all(df: pd.DataFrame):
    kmeans = KMeans(n_clusters=6, random_state=17)
    kmeans.fit(df)
    clusters = kmeans.predict(df)
    return clusters


def visualization_of_clusters(df: pd.DataFrame, clusters: np.ndarray):
    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=17)
    tsne_features = tsne.fit_transform(df)

    # Create scatter plot of the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.colorbar(label='Cluster')
    plt.show()

    

def kmeans_cluster_group(df: pd.DataFrame, group: list):
    pass

def gmm_cluster_all(df: pd.DataFrame):
    pass

def gmm_cluster_group(df: pd.DataFrame, group: list):
    pass

def dbscan_cluster_all(df: pd.DataFrame):
    pass

def dbscan_cluster_group(df: pd.DataFrame, group: list):
    pass