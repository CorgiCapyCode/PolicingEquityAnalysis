import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score

def clustering(df: pd.DataFrame, run_type: int =2):
    
    # TEMP SAMPLE
    sample_df = df.sample(frac=0.05, random_state=17)
    # sample_df = df

    if run_type == 1:
        # Results in low Silhouette Score values
        print("Run on all features")
        sample_df, k_means_nucleus, gmm_nucleus = run_complete_feature_space(sample_df=sample_df)

        return sample_df
    
    elif run_type == 2:
        # Results in low Silhouette Score values
        print("-------------------------")
        print("Run on reduced feature space")
        print("-------------------------")        
        reduced_df = sample_df.copy()
        reduced_df.drop(('LOCATION_STREET_NUMBER', 'STREET_ID'), axis=1, inplace=True)
        reduced_df.drop(('OFFICER_ID', 'OFFICER_ID'), axis=1, inplace=True)
        reduced_df.info()
        reduced_df, reduced_kmeans_nucleus, reduced_gmm_nucleus = run_complete_feature_space(sample_df=reduced_df)
        return reduced_df
    
    else:
        subject_columns = [
            ("SUBJECT_GENDER", "SEX"),
            ("SUBJECT_RACE", "DESCRIPTION"),
            ("SUBJECT_DETAILS.2", "COMPLEXION"),
            ("SUBJECT_DETAILS", "PRIORS"),
            ("OFFICER_AGE", "AGE_AT_FIO_CORRECTED")
        ]
        vehicle_columns = [
            ("VEHICLE_MAKE", "VEH_MAKE"),
            ("VEHICLE_YEAR", "VEH_YEAR_NUM"),
            ("VEHICLE_COLOR", "VEH_COLOR"),
            ("VEHICLE_DETAILS", "VEH_OCCUPANT"),
            ("VEHICLE_DETAILS.1", "VEH_STATE")
        ]
        location_columns = [
            ("LOCATION_STREET_NUMBER", "STREET_ID"),
            ("LOCATION_CITY", "CITY"),
            ("OFFICER_ASSIGNMENT", "OFF_DIST_ID")
        ]
        officer_columns = [
            ("OFFICER_SUPERVISOR", "SUPERVISOR_ID"),
            ("OFFICER_ID", "OFFICER_ID")
        ]
        action_columns = [
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_F"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_I"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_P"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_S"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_O"),
           ("SEARCH_CONDUCTED", "SEARCH_V"),
           ("SEARCH_CONDUCTED", "SEARCH_P"),
           ("SEARCH_REASON", "BASIS"),
           ("INCIDENT_REASON", "STOP_REASONS"),
           ("INCIDENT_REASON.1", "FIOFS_REASONS"),
           ('DISPOSITION', 'OUTCOME_F'),
           ('DISPOSITION', 'OUTCOME_O'),
           ('DISPOSITION', 'OUTCOME_S')
        ]
        date_columns = [
            ('Year', 'Year'),
            ('Month', 'Month'),
            ("Day", "Day")
        ]

        group_subject_df = sample_df[subject_columns].copy()
        group_vehicle_df = sample_df[vehicle_columns].copy()
        group_location_df = sample_df[location_columns].copy()
        group_officer_df = sample_df[officer_columns].copy()
        group_action_df = sample_df[action_columns].copy()
        group_date_df = sample_df[date_columns].copy()
        
        print("Run Subject cluster")
        expanded_sample_df, subject_sil_kmeans, subject_sil_gmm = grouped_df_clustering(grouped_df=group_subject_df, group_name="subject", original_df=sample_df)
        print("Run vehicle cluster")
        expanded_sample_df, vehicle_sil_kmeans, vehicle_sil_gmm = grouped_df_clustering(grouped_df=group_vehicle_df, group_name="vehicle", original_df=expanded_sample_df)
        print("Run location cluster")
        expanded_sample_df, location_sil_kmeans, location_sil_gmm = grouped_df_clustering(grouped_df=group_location_df, group_name="location", original_df=expanded_sample_df)
        print("Run officer cluster")
        expanded_sample_df, officer_sil_kmeans, officer_sil_gmm = grouped_df_clustering(grouped_df=group_officer_df, group_name="officer", original_df=expanded_sample_df)
        print("run action cluster")
        expanded_sample_df, action_sil_kmeans, action_sil_gmm = grouped_df_clustering(grouped_df=group_action_df, group_name="action", original_df=expanded_sample_df)
        print("run date cluster")
        expanded_sample_df, date_sil_kmeans, date_sil_gmm = grouped_df_clustering(grouped_df=group_date_df, group_name="date", original_df=expanded_sample_df)

        if subject_sil_gmm <= subject_sil_kmeans and 0.5 <= subject_sil_kmeans:
            print("For Subject: KMEANs")
            subject_moded_df = update_df_information_with_group_clusters(df=expanded_sample_df, cluster_column=("subject_Cluster_K_Means", ""), feature_columns=subject_columns)
        elif 0.5 <= subject_sil_gmm:
            print("For Subject: GMM")
            subject_moded_df = update_df_information_with_group_clusters(df=expanded_sample_df, cluster_column=("subject_Cluster_GMM", ""), feature_columns=subject_columns)
        else:
            print("None - no good clusters for subject")
            subject_moded_df = expanded_sample_df
             
        if vehicle_sil_gmm <= vehicle_sil_kmeans and 0.5 <= vehicle_sil_kmeans:
            print("For Vehicle: KMEANs")
            vehicle_moded_df = update_df_information_with_group_clusters(df=subject_moded_df, cluster_column=("vehicle_Cluster_K_Means", ""), feature_columns=vehicle_columns)
        elif 0.5 <= vehicle_sil_gmm:
            print("For Vehicle: GMM")
            vehicle_moded_df = update_df_information_with_group_clusters(df=subject_moded_df, cluster_column=("vehicle_Cluster_GMM", ""), feature_columns=vehicle_columns)
        else:
            print("None - no good clusters for vehicle")
            vehicle_moded_df = subject_moded_df
            
        if location_sil_gmm <= location_sil_kmeans and 0.5 <= location_sil_kmeans:
            print("For Location: KMEANs")
            location_moded_df = update_df_information_with_group_clusters(df=vehicle_moded_df, cluster_column=("location_Cluster_K_Means", ""), feature_columns=location_columns)
        elif 0.5 <= location_sil_gmm:
            print("For Location: GMM")
            location_moded_df = update_df_information_with_group_clusters(df=vehicle_moded_df, cluster_column=("location_Cluster_GMM", ""), feature_columns=location_columns)
        else:
            print("None - no good clusters for location")
            location_moded_df = vehicle_moded_df
            
        if action_sil_gmm <= action_sil_kmeans and 0.5 <= action_sil_kmeans:
            print("For Action: KMEANs")
            action_moded_df = update_df_information_with_group_clusters(df=location_moded_df, cluster_column=("action_Cluster_K_Means", ""), feature_columns=action_columns)
        elif 0.5 <= action_sil_gmm:
            print("For Action: GMM")
            action_moded_df = update_df_information_with_group_clusters(df=location_moded_df, cluster_column=("action_Cluster_K_GMM", ""), feature_columns=action_columns)
        else:
            print("None - no good clusters for action")
            action_moded_df = location_moded_df
        
        if officer_sil_gmm <= officer_sil_kmeans and 0.5 <= officer_sil_kmeans:
            print("For Officer: KMEANs")
            officer_moded_df = update_df_information_with_group_clusters(df=action_moded_df, cluster_column=("officer_Cluster_K_Means", ""), feature_columns=officer_columns)
        elif 0.5 <= officer_sil_gmm:
            print("For Officer: GMM")
            officer_moded_df = update_df_information_with_group_clusters(df=action_moded_df, cluster_column=("officer_Cluster_GMM", ""), feature_columns=officer_columns)
        else:
            print("None - no good clusters for officer")
            officer_moded_df = action_moded_df
            
        if date_sil_gmm <= date_sil_kmeans and 0.5 <= date_sil_kmeans:
            print("For Date: KMEANs")
            date_moded_df = update_df_information_with_group_clusters(df=officer_moded_df, cluster_column=("date_Cluster_K_Means", ""), feature_columns=date_columns)
        elif 0.5 <= date_sil_kmeans:
            print("For Date: GMM")
            date_moded_df = update_df_information_with_group_clusters(df=officer_moded_df, cluster_column=("date_Cluster_GMM", ""), feature_columns=date_columns)
        else:
            print("None - no good clusters for date")
            date_moded_df = officer_moded_df

        return date_moded_df


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


def visualization_of_clusters(df: pd.DataFrame, clusters: np.ndarray, name: str):
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
    plt.savefig(f"{name}.jpg")
    # plt.show()
    plt.close()
    

def get_nucleus_details(df: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    cluster_nucleus_df = df.groupby(clusters).agg(lambda x: x.mode().iloc[0])
    return cluster_nucleus_df


def check_cluster_quality(df: pd.DataFrame, labels: np.ndarray) -> float:
    score = silhouette_score(df, labels)
    return score


def run_complete_feature_space(sample_df: pd.DataFrame):
        # Running for all features
        ohe_df_original = one_hot_encoding(df=sample_df)
        # ohe_df.info()
        
        # KMeans - All
        elbow_value = kelbow_visualizer(ohe_df_original)
        print(f"Optimal clusters acc. KElbow: {elbow_value}.")  
        labels_kmeans_all = kmeans_cluster(df=ohe_df_original, k_clusters=elbow_value)
        
        print("Labels:")
        print(labels_kmeans_all)
        
        # GMM - All
        optimal_n = optimal_gmm(df=ohe_df_original)
        print(f"Optimal n for gmm: {optimal_n}")
        labels_gmm_all = gmm_cluster(df=ohe_df_original, n_components=optimal_n)
        
        # DBSCAN - All
        labels_dbscan_all = dbscan_cluster(df=ohe_df_original)
    
        # Visualization
        visualization_of_clusters(df=ohe_df_original, clusters=labels_kmeans_all, name="kmeans_all")
        visualization_of_clusters(df=ohe_df_original, clusters=labels_gmm_all, name="gmm_all")
        
        # Silhouette Value
        silhouette_value_kmeans_all = check_cluster_quality(df=ohe_df_original, labels=labels_kmeans_all)
        print(f"silhouette_value kMeans all: {silhouette_value_kmeans_all}")
        sample_df["Clusters_all_kmeans"] = labels_kmeans_all        
        silhouette_value_gmm_all = check_cluster_quality(df=ohe_df_original, labels=labels_gmm_all)
        print(f"Silhouette_value gmm all: {silhouette_value_gmm_all}")
        sample_df["Clusters_all_gmm"] = labels_gmm_all        
        k_means_nucleus_df = get_nucleus_details(df=sample_df, clusters=labels_kmeans_all) 
        gmm_nucleus_df = get_nucleus_details(df=sample_df, clusters=labels_gmm_all)
        
        if labels_dbscan_all == -1:
            pass
        else:
            visualization_of_clusters(df=ohe_df_original, clusters=labels_dbscan_all, name="dbscan_all")            
            silhouette_value_dbscan_all = check_cluster_quality(df=ohe_df_original, labels=labels_dbscan_all)
            print(f"Silhouette_value dbscan_all : {silhouette_value_dbscan_all}")
            sample_df["Clusters_all_dbscan"] = labels_dbscan_all        

     
          
        return sample_df, k_means_nucleus_df, gmm_nucleus_df


def kelbow_visualizer(df: pd.DataFrame, k_range: tuple =(2, 100), name: str="na"):
    model = KMeans(random_state=17)
    visualizer = KElbowVisualizer(model, k=k_range, timings=False)
    visualizer.fit(df)
    print(f"Create gmm plot for {name}.")
    visualizer.show(outpath=f"clustering/optimal_cluster/{name}_kelbow_visualizer.jpg")
    elbow_value = visualizer.elbow_value_
    if elbow_value is None:
        print("No clusters found - Clusters set to 2")
        return 2
    else:
        return visualizer.elbow_value_
    

def kmeans_cluster(df: pd.DataFrame, k_clusters: int):
    kmeans = KMeans(n_clusters=k_clusters, random_state=17)
    kmeans.fit(df)
    
    labels = kmeans.labels_
    
    return labels

    
def optimal_gmm(df: pd.DataFrame, g_range: tuple =(2, 100), name: str ="na") -> int:
    bics = []
    n_components_range = range(g_range[0], g_range[1])
    
    for n in n_components_range:
        print(f"Checking for n: {n}")
        gmm = GaussianMixture(n_components=n, random_state=17)
        gmm.fit(df)
        
        bics.append(gmm.bic(df))
    
    optimal_n = n_components_range[np.argmin(bics)]
    plt.plot(n_components_range, bics, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC Score')
    plt.title('BIC Score vs Number of Components')
    plt.grid(True)
    print(f"Create gmm plot for {name}.")
    plt.savefig(f"clustering/optimal_cluster/{name}_gmm_bic_values.jpg")
    plt.close()
    return optimal_n
  

def gmm_cluster(df: pd.DataFrame, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=17)
    gmm.fit(df)
    
    labels = gmm.predict(df)
    return labels


def dbscan_cluster(df: pd.DataFrame, eps: float =0.2, min_samples: int  =5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(df)
    
    labels = dbscan.labels_
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return -1
    return labels


def grouped_df_clustering(grouped_df: pd.DataFrame, group_name: str, original_df: pd.DataFrame):
    cluster_information = {}
    ohe_df = one_hot_encoding(df=grouped_df)
    elbow_value = kelbow_visualizer(df=ohe_df, k_range=(2, 100), name=group_name)
    labels_kmeans = kmeans_cluster(df=ohe_df, k_clusters=elbow_value) 
    key = f"{group_name}_elbow_value"
    cluster_information[key] = elbow_value
    # key = f"{group_name}_labels_keams"
    # cluster_information[key] = labels_kmeans
    
    optimal_n = optimal_gmm(df=ohe_df, g_range=(2, 100), name=group_name)
    labels_gmm = gmm_cluster(df=ohe_df, n_components=optimal_n)
    key = f"{group_name}_optimal_n"
    cluster_information[key] = optimal_n
    # key = f"{group_name}_labels_gmm"
    # cluster_information[key] = labels_gmm
    
    # labels_dbscan = dbscan_cluster(df=ohe_df)
    # key = f"{group_name}_labels_dbscan"
    # cluster_information[key] = labels_dbscan
    
    silhouette_kmeans = check_cluster_quality(df=ohe_df, labels=labels_kmeans)
    silhouette_gmm = check_cluster_quality(df=ohe_df, labels=labels_gmm)
    key = f"{group_name}_sil_score_kmeans"
    cluster_information[key] = silhouette_kmeans
    key = f"{group_name}_sil_score_gmm"
    cluster_information[key] = silhouette_gmm
    print(f"{group_name} Silhouette Scores:")
    print(cluster_information)
    
    # groupded_df["Cluster_K_Means"] = labels_kmeans
    # groupded_df["Cluster_GMM"] = labels_gmm
    
    # grouped_kmeans_nucleus_df = get_nucleus_details(df=groupded_df, clusters=labels_kmeans)
    # grouped_gmm_nucleus_df = get_nucleus_details(df=groupded_df, clusters=labels_gmm)
    
    original_df[f"{group_name}_Cluster_K_Means"] = labels_kmeans
    original_df[f"{group_name}_Cluster_GMM"] = labels_gmm
    
    return original_df, silhouette_kmeans, silhouette_gmm


def update_df_information_with_group_clusters(df: pd.DataFrame, cluster_column: str, feature_columns: list) -> pd.DataFrame:
    moded_df = df.copy()
    
    cluster_nucleus = df.groupby(cluster_column).agg(lambda x: x.mode().iloc[0])
    
    for cluster_label, nucleus_values in cluster_nucleus.iterrows():
        cluster_data = moded_df[moded_df[cluster_column] == cluster_label]
        
        for column in feature_columns:
            moded_df.loc[moded_df[cluster_column] == cluster_label, column] = nucleus_values[column]
            
    return moded_df


def second_clustering(df: pd.DataFrame, run_type: int =0) -> pd.DataFrame:
    if run_type != 3:
        return None
    
    df = drop_cluster_labels(df=df)
    unique_df_with_counts = count_occurances(df=df)
    
    
    unique_df_with_counts.info()
    
    ohe_df = one_hot_encoding(df=unique_df_with_counts)
    
        
    elbow_value = kelbow_visualizer(df=ohe_df, k_range=(2, 50), name="final_clustering")
    labels_kmeans = kmeans_cluster(df=ohe_df, k_clusters=elbow_value)
    print(f"Optimal clusters acc. KElbow: {elbow_value}.") 
    
    optimal_n = optimal_gmm(df=ohe_df, g_range=(2, 50), name="final_clustering")
    print(f"Optimal n for gmm: {optimal_n}")
    labels_gmm = gmm_cluster(df=ohe_df, n_components=optimal_n)
    
    silhouette_value_kmeans = check_cluster_quality(df=ohe_df, labels=labels_kmeans)    
    silhouette_value_gmm = check_cluster_quality(df=ohe_df, labels=labels_gmm)
    
    if silhouette_value_kmeans < silhouette_value_gmm:
        print(f"Silhouette Score GMM: {silhouette_value_gmm}")
        unique_df_with_counts["Cluster_gmm"] = labels_gmm
    else:
        unique_df_with_counts["Cluster_kmeans"] = labels_kmeans
        print(f"Silhouette Score KMeans: {silhouette_value_kmeans}")
    return unique_df_with_counts


def drop_cluster_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] > 32:
        df = df.iloc[:, :32]
    return df

def count_occurances(df: pd.DataFrame) -> pd.DataFrame:
    unique_df = df.drop_duplicates()

    counts = df.groupby(list(df.columns)).size().reset_index(name = ("occurances", "occurances"))
    unique_df_with_counts = pd.merge(unique_df, counts, on=list(df.columns), how="left")
    
    return unique_df_with_counts
    