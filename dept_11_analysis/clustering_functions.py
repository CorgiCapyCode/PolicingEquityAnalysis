import pandas as pd
from sklearn.cluster import KMeans

def clustering(df: pd.DataFrame):
    ohe_df = one_hot_encoding(df=df)
    ohe_df.info()

    k_clusters_all = kmeans_cluster_all(df=ohe_df)
    
    return ohe_df


def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    ohe_df = df.copy()
    # Converting the multi-level column index to one level & ensure all are of type str
    ohe_df.columns = ["_".join(map(str, col)) for col in df.columns]
    ohe_df.columns = ohe_df.columns.astype(str)
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