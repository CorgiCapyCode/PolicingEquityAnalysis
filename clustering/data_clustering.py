import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from stats.base_statistics import analyse_feature
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
from pandas.errors import PerformanceWarning

def clustering(df: pd.DataFrame, run_type: int =2):
    
    # TEMP SAMPLE
    # sample_df = df.sample(frac=0.1, random_state=17)
    sample_df = df

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
        ]
        
        officer_columns = [
            ("OFFICER_SUPERVISOR", "SUPERVISOR_ID"),
            ("OFFICER_ID", "OFFICER_ID"),
            ("OFFICER_ASSIGNMENT", "OFF_DIST_ID")            
        ]
        
        action_columns = [
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_F"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_I"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_P"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_S"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_O"),
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
        expanded_sample_df, subject_sil,  = grouped_df_clustering(grouped_df=group_subject_df, group_name="subject", original_df=sample_df)
        print("Run vehicle cluster")
        expanded_sample_df, vehicle_sil = grouped_df_clustering(grouped_df=group_vehicle_df, group_name="vehicle", original_df=expanded_sample_df)
        print("Run location cluster")
        expanded_sample_df, location_sil = grouped_df_clustering(grouped_df=group_location_df, group_name="location", original_df=expanded_sample_df)
        print("Run officer cluster")
        expanded_sample_df, officer_sil = grouped_df_clustering(grouped_df=group_officer_df, group_name="officer", original_df=expanded_sample_df)
        print("run action cluster")
        expanded_sample_df, action_sil = grouped_df_clustering(grouped_df=group_action_df, group_name="action", original_df=expanded_sample_df)
        print("run date cluster")
        expanded_sample_df, date_sil = grouped_df_clustering(grouped_df=group_date_df, group_name="date", original_df=expanded_sample_df)

        '''
        if 0.5 <= subject_sil:
            print("Update Subject info")
            subject_moded_df = update_df_information_with_group_clusters(df=expanded_sample_df, cluster_column=("subject_cluster", "subject_cluster"), feature_columns=subject_columns)
        else:
            print("None - no good clusters for subject")
            subject_moded_df = expanded_sample_df
             
        if 0.5 <= vehicle_sil:
            print("Update Vehicle Info")
            vehicle_moded_df = update_df_information_with_group_clusters(df=subject_moded_df, cluster_column=("vehicle_cluster", "vehicle_cluster"), feature_columns=vehicle_columns)
        else:
            print("None - no good clusters for vehicle")
            vehicle_moded_df = subject_moded_df
            
        if 0.5 <= location_sil:
            print("Update Location Info")
            location_moded_df = update_df_information_with_group_clusters(df=vehicle_moded_df, cluster_column=("location_cluster", "location_cluster"), feature_columns=location_columns)

        else:
            print("None - no good clusters for location")
            location_moded_df = vehicle_moded_df
            
        if 0.5 <= action_sil:
            print("Update Action Info")
            action_moded_df = update_df_information_with_group_clusters(df=location_moded_df, cluster_column=("action_cluster", "action_cluster"), feature_columns=action_columns)

        else:
            print("None - no good clusters for action")
            action_moded_df = location_moded_df
        
        if 0.5 <= officer_sil:
            print("Update Officer Info")
            officer_moded_df = update_df_information_with_group_clusters(df=action_moded_df, cluster_column=("officer_cluster", "officer_cluster"), feature_columns=officer_columns)

        else:
            print("None - no good clusters for officer")
            officer_moded_df = action_moded_df
            
        if 0.5 <= date_sil:
            print("Update Date Info")
            date_moded_df = update_df_information_with_group_clusters(df=officer_moded_df, cluster_column=("date_cluster", "date_cluster"), feature_columns=date_columns)

        else:
            print("None - no good clusters for date")
            date_moded_df = officer_moded_df
        '''
        return expanded_sample_df


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
    plt.rcParams["font.family"] = "DejaVu Sans"
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
    count_of_increasing_values = 0
    actual_range = []
    
    for n in n_components_range:
        print(f"Checking for n: {n}")
        gmm = GaussianMixture(n_components=n, random_state=17)
        gmm.fit(df)
        
        bics.append(gmm.bic(df))
        actual_range.append(n)
        if len(bics) > 1:
            if bics[-1] > bics[-2]:
                count_of_increasing_values += 1
            else:
                count_of_increasing_values = 0
            
            if count_of_increasing_values > 10:
                print("Found minimum")
                break
    
    
    optimal_n = actual_range[np.argmin(bics)]
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.plot(actual_range, bics, marker='o')
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
    
    if silhouette_gmm <= silhouette_kmeans:
        original_df[(f"{group_name}_cluster",f"{group_name}_cluster")] = labels_kmeans
        return original_df, silhouette_kmeans
    else:
        original_df[(f"{group_name}_cluster",f"{group_name}_cluster")] = labels_gmm
        return original_df, silhouette_gmm


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
    
    clustering_df = df.copy()
    
    # Reduce to cluster features
    clustering_df = keep_only_cluster_labels(df=clustering_df)
    print("Kept df:")
    clustering_df.info()
    # Convert to object type as preparation for one hot encoded, currently all of type int
    clustering_df = clustering_df.astype(str)
    

       
    clustering_df.columns = clustering_df.columns.map("_".join)
    clustering_df = count_occurances(clustering_df)   
    
    encoded_clustering_df = pd.get_dummies(clustering_df, columns=clustering_df.columns[:])
    encoded_clustering_df = encoded_clustering_df.astype(int)
    

    
    
    elbow_value = kelbow_visualizer(df=encoded_clustering_df, k_range=(2, 20), name="final_clustering")
    labels_kmeans = kmeans_cluster(df=encoded_clustering_df, k_clusters=elbow_value)
    silhouette_value_kmeans = check_cluster_quality(df=encoded_clustering_df, labels=labels_kmeans)
    print(f"Optimal cluster for kmeans: {elbow_value} with score {silhouette_value_kmeans}")
    
    optimal_n = optimal_gmm(df=encoded_clustering_df, g_range=(2, 20), name="final_clustering")
    labels_gmm = gmm_cluster(df=encoded_clustering_df, n_components=optimal_n)
    silhouette_value_gmm = check_cluster_quality(df=encoded_clustering_df, labels=labels_gmm)
    print(f"Optimal cluster for gmm: {optimal_n} with score {silhouette_value_gmm}")
    
    if silhouette_value_kmeans < silhouette_value_gmm:
        print(f"Silhouette Score GMM: {silhouette_value_gmm}")
        clustering_df["Cluster_gmm"] = labels_gmm
    else:
        clustering_df["Cluster_kmeans"] = labels_kmeans
        print(f"Silhouette Score KMeans: {silhouette_value_kmeans}")   
    
    
    return clustering_df


def keep_only_cluster_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] > 30:
        df = df.iloc[:, 30:]
    return df


def count_occurances(df: pd.DataFrame) -> pd.DataFrame:
    
    counts = df.groupby(list(df.columns)).size().reset_index(name="occurrences")
    unique_order = df.drop_duplicates()
    unique_order_with_counts = pd.merge(unique_order, counts, on=list(df.columns), how="left")
    return unique_order_with_counts



def read_csv_file(path) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1], dtype=str)


def save_df_to_csv(df: pd.DataFrame, output_filename: str):
    df.to_csv(output_filename, index=True, mode="w")


def group_cluster_analysis(df: pd.DataFrame):
        subject_columns = [
            ("SUBJECT_GENDER", "SEX"),
            ("SUBJECT_RACE", "DESCRIPTION"),
            ("SUBJECT_DETAILS.2", "COMPLEXION"),
            ("SUBJECT_DETAILS", "PRIORS"),
            ("OFFICER_AGE", "AGE_AT_FIO_CORRECTED"),
            ("subject_cluster", "subject_cluster")
        ]
        
        vehicle_columns = [
            ("VEHICLE_MAKE", "VEH_MAKE"),
            ("VEHICLE_YEAR", "VEH_YEAR_NUM"),
            ("VEHICLE_COLOR", "VEH_COLOR"),
            ("VEHICLE_DETAILS", "VEH_OCCUPANT"),
            ("VEHICLE_DETAILS.1", "VEH_STATE"),
            ("vehicle_cluster", "vehicle_cluster")
        ]
        
        location_columns = [
            ("LOCATION_STREET_NUMBER", "STREET_ID"),
            ("LOCATION_CITY", "CITY"),
            ("location_cluster", "location_cluster")
        ]
        
        officer_columns = [
            ("OFFICER_SUPERVISOR", "SUPERVISOR_ID"),
            ("OFFICER_ID", "OFFICER_ID"),
            ("OFFICER_ASSIGNMENT", "OFF_DIST_ID"),
            ("officer_cluster", "officer_cluster")            
        ]
        
        action_columns = [
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_F"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_I"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_P"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_S"),
           ("UNKNOWN_FIELD_TYPE", "FIOFS_TYPE_O"),
           ("INCIDENT_REASON", "STOP_REASONS"),
           ("INCIDENT_REASON.1", "FIOFS_REASONS"),
           ('DISPOSITION', 'OUTCOME_F'),
           ('DISPOSITION', 'OUTCOME_O'),
           ('DISPOSITION', 'OUTCOME_S'),
           ("action_cluster", "action_cluster")
        ]
        
        date_columns = [
            ('Year', 'Year'),
            ('Month', 'Month'),
            ("Day", "Day"),
            ("date_cluster", "date_cluster")
        ]
        
        df_subject = df[subject_columns].copy()
        df_vehicle = df[vehicle_columns].copy()
        df_location = df[location_columns].copy()
        df_officer = df[officer_columns].copy()
        df_action = df[action_columns].copy()
        df_date = df[date_columns].copy()
        
        # Cluster Counts:        
        sub_data = analyse_feature(df=df_subject, feature=("subject_cluster", "subject_cluster"), location="clustering/cluster_stats")
        veh_data = analyse_feature(df=df_vehicle, feature=("vehicle_cluster", "vehicle_cluster"), location="clustering/cluster_stats")
        loc_data = analyse_feature(df=df_location, feature=("location_cluster", "location_cluster"), location="clustering/cluster_stats")
        off_data = analyse_feature(df=df_officer, feature=("officer_cluster", "officer_cluster"), location="clustering/cluster_stats")             
        act_data = analyse_feature(df=df_action, feature=("action_cluster", "action_cluster"), location="clustering/cluster_stats")
        dat_data = analyse_feature(df=df_date, feature=("date_cluster", "date_cluster"), location="clustering/cluster_stats")
        
        # Nucleus per Cluster
        subject_nuclei = df_subject.groupby(("subject_cluster", "subject_cluster")).agg(lambda x: x.mode().iloc[0])
        subject_nuclei = subject_nuclei.reset_index()
        subject_nuclei.columns = ["_".join(col) if isinstance(col, tuple) else col for col in subject_nuclei.columns]        
          
        vehicle_nuclei = df_vehicle.groupby(("vehicle_cluster", "vehicle_cluster")).agg(lambda x: x.mode().iloc[0])
        vehicle_nuclei = vehicle_nuclei.reset_index()
        vehicle_nuclei.columns = ["_".join(col) if isinstance(col, tuple) else col for col in vehicle_nuclei.columns]         
        
        location_nuclei = df_location.groupby(("location_cluster", "location_cluster")).agg(lambda x: x.mode().iloc[0])
        location_nuclei = location_nuclei.reset_index()
        location_nuclei.columns = ["_".join(col) if isinstance(col, tuple) else col for col in location_nuclei.columns]         
        
        officer_nuclei = df_officer.groupby(("officer_cluster", "officer_cluster")).agg(lambda x: x.mode().iloc[0])     
        officer_nuclei = officer_nuclei.reset_index()
        officer_nuclei.columns = ["_".join(col) if isinstance(col, tuple) else col for col in officer_nuclei.columns]           
           
        action_nuclei = df_action.groupby(("action_cluster", "action_cluster")).agg(lambda x: x.mode().iloc[0])
        action_nuclei = action_nuclei.reset_index()
        action_nuclei.columns = ["_".join(col) if isinstance(col, tuple) else col for col in action_nuclei.columns]         
        
        date_nuclei = df_date.groupby(("date_cluster", "date_cluster")).agg(lambda x: x.mode().iloc[0])        
        date_nuclei = date_nuclei.reset_index()
        date_nuclei.columns = ["_".join(col) if isinstance(col, tuple) else col for col in date_nuclei.columns]         

        # Merging columns and save them.
        final_subject_df = pd.merge(sub_data, subject_nuclei, left_on='Unique Values', right_on='subject_cluster_subject_cluster')
        final_vehicle_df = pd.merge(veh_data, vehicle_nuclei, left_on='Unique Values', right_on='vehicle_cluster_vehicle_cluster')
        final_location_df = pd.merge(loc_data, location_nuclei, left_on='Unique Values', right_on='location_cluster_location_cluster')
        final_officer_df = pd.merge(off_data, officer_nuclei, left_on='Unique Values', right_on='officer_cluster_officer_cluster')
        final_action_df = pd.merge(act_data, action_nuclei, left_on='Unique Values', right_on='action_cluster_action_cluster')
        final_date_df = pd.merge(dat_data, date_nuclei, left_on='Unique Values', right_on='date_cluster_date_cluster')
        
        save_df_to_csv(df=final_subject_df, output_filename="clustering/cluster_stats/subject_cluster_info.csv")          
        save_df_to_csv(df=final_vehicle_df, output_filename="clustering/cluster_stats/vehicle_cluster_info.csv")         
        save_df_to_csv(df=final_location_df, output_filename="clustering/cluster_stats/location_cluster_info.csv")    
        save_df_to_csv(df=final_officer_df, output_filename="clustering/cluster_stats/officer_cluster_info.csv")  
        save_df_to_csv(df=final_action_df, output_filename="clustering/cluster_stats/action_cluster_info.csv")  
        save_df_to_csv(df=final_date_df, output_filename="clustering/cluster_stats/date_cluster_info.csv")  

        # Create TNSE visualization for each group
        # Does not work - time for creation too long
        '''
        print("Visualization")
        visualization_of_group_clusters(df_subject, cluster_column="subject_cluster", name="clustering/cluster_stats/visualization/subject_cluster.jpg")
        print("Subject - Done")
        visualization_of_group_clusters(df_vehicle, cluster_column="vehicle_cluster", name="clustering/cluster_stats/visualization/vehicle_cluster.jpg")
        print("Vehicle - Done")
        visualization_of_group_clusters(df_location, cluster_column="location_cluster", name="clustering/cluster_stats/visualization/location_cluster.jpg")
        print("Location - Done")
        visualization_of_group_clusters(df_officer, cluster_column="officer_cluster", name="clustering/cluster_stats/visualization/officer_cluster.jpg")
        print("Officer - Done")
        visualization_of_group_clusters(df_action, cluster_column="action_cluster", name="clustering/cluster_stats/visualization/action_cluster.jpg")
        print("Action - Done")
        visualization_of_group_clusters(df_date, cluster_column="date_cluster", name="clustering/cluster_stats/visualization/date_cluster.jpg")
        '''
        # Visualization based on cluster results
        # Counts are represented by size of the dots
        # Cluster label is represented by color
        # Use label encoding
        
        
        
        # Subject:
        # Drop SEX since it is equal for all features
        # PRIORS determines the direction (+1/-1) of the vector for positive (TRUE) and negative (FALSE, UNKNOWN)
        
        subject_visualization(final_subject_df)
        
        # Vehicle:
        # Drop VEH_STATE 
        # OCCUPANT as PRIORS
        # Ignore first cluster with NO VEHICLE INVOVLED
        
        vehicle_visualization(final_vehicle_df)
        
        # Location:
        # 2D diagram
        
        location_visualization(final_location_df)
        
        # Officer:
        # 3D diagram
        
        officer_visualization(final_officer_df)
        
        # Action:
        # Drop FIOFS_TYPE_P
        # Use STOP_REASONS and FIOFS_REASONS as two axes
        # Put FIOFS_TYPE together and label encode it
        # Create a pointer of outcome values
        
        action_visualization(final_action_df)
        
        # Date:
        # 3D diagram
        
        date_visualization(df=final_date_df)
        
        
def visualization_of_group_clusters(df: pd.DataFrame, cluster_column: str, name: str):
    
    df.columns = [col[1] for col in df.columns]
   
    
    for col in df.columns:
        if df[col].dtype == "object":
            try: 
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass

    features = df.drop(cluster_column, axis=1)
    string_features = features.select_dtypes(include=["object"]).columns
    if len(string_features) > 0:
        features = pd.get_dummies(features, columns=string_features)
    
    tsne = TSNE(n_components=3, random_state=17)
    tsne_features = tsne.fit_transform(features)
    
    clusters = df[cluster_column]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(tsne_features[:, 0], tsne_features[:, 1], tsne_features[:, 2], c=clusters, cmap='viridis', alpha=0.5)
    ax.set_title('t-SNE Visualization of Clusters')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    plt.colorbar(ax.scatter(tsne_features[:, 0], tsne_features[:, 1], tsne_features[:, 2], c=clusters), label='Cluster')
    plt.savefig(f"{name}.jpg")
    plt.close()
    
 
def subject_visualization(df: pd.DataFrame):
    
    df["OFFICER_AGE_AGE_AT_FIO_CORRECTED"] = df["OFFICER_AGE_AGE_AT_FIO_CORRECTED"].astype(float)
    
    
    label_encoder = {}
    for feature in ["SUBJECT_RACE_DESCRIPTION", "SUBJECT_DETAILS.2_COMPLEXION"]:
        encoder = LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature])
        label_encoder [feature] = encoder
    
    scaler = MinMaxScaler()
    df[["SUBJECT_RACE_DESCRIPTION", "SUBJECT_DETAILS.2_COMPLEXION"]] = scaler.fit_transform(df[["SUBJECT_RACE_DESCRIPTION", "SUBJECT_DETAILS.2_COMPLEXION"]])
    
    df["Scalar"] = df["SUBJECT_DETAILS_PRIORS"].apply(lambda x: 1 if x == "YES" else -1)
    
    df["SUBJECT_RACE_DESCRIPTION"] *= df["Scalar"]
    df["SUBJECT_DETAILS.2_COMPLEXION"] *= df["Scalar"]
    df["OFFICER_AGE_AGE_AT_FIO_CORRECTED"] *= df["Scalar"]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        df["SUBJECT_RACE_DESCRIPTION"],
        df["SUBJECT_DETAILS.2_COMPLEXION"],
        df["OFFICER_AGE_AGE_AT_FIO_CORRECTED"],
        s=df["Percentage"] * 1000,
        c=np.arange(len(df)),
        cmap='tab20c',
        alpha=0.8 
    )
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    ax.plot([0, 0], [0, 0], zlim, color='black', linestyle='--')
    ax.plot([0, 0], ylim, [0, 0], color='black', linestyle='--')
    ax.plot(xlim, [0, 0], [0, 0], color='black', linestyle='--')
    
    ax.set_xlabel('DESCRIPTION')
    ax.set_ylabel('COMPLEXION')
    ax.set_zlabel('AGE')
    plt.savefig("clustering/cluster_stats/visualization/subject_cluster_visu.jpg")
    plt.close()
    

def vehicle_visualization(df: pd.DataFrame):
    
    df = df.drop(index=0).reset_index(drop=True)

    
    label_encoder = {}
    for feature in ["VEHICLE_MAKE_VEH_MAKE", "VEHICLE_YEAR_VEH_YEAR_NUM", "VEHICLE_COLOR_VEH_COLOR"]:
        encoder = LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature])
        label_encoder [feature] = encoder
    
    scaler = MinMaxScaler()
    df[["VEHICLE_MAKE_VEH_MAKE", "VEHICLE_YEAR_VEH_YEAR_NUM", "VEHICLE_COLOR_VEH_COLOR"]] = scaler.fit_transform(df[["VEHICLE_MAKE_VEH_MAKE", "VEHICLE_YEAR_VEH_YEAR_NUM", "VEHICLE_COLOR_VEH_COLOR"]])
    
    df["Scalar"] = df["VEHICLE_DETAILS_VEH_OCCUPANT"].apply(lambda x: 1 if x == "DRIVER" else -1)
    
    df["VEHICLE_MAKE_VEH_MAKE"] *= df["Scalar"]
    df["VEHICLE_YEAR_VEH_YEAR_NUM"] *= df["Scalar"]
    df["VEHICLE_COLOR_VEH_COLOR"] *= df["Scalar"]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        df["VEHICLE_MAKE_VEH_MAKE"],
        df["VEHICLE_YEAR_VEH_YEAR_NUM"],
        df["VEHICLE_COLOR_VEH_COLOR"],
        s=df["Percentage"] * 1000,
        c=np.arange(len(df)),
        cmap='tab20c',
        alpha=0.8 
    )
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    ax.plot([0, 0], [0, 0], zlim, color='black', linestyle='--')
    ax.plot([0, 0], ylim, [0, 0], color='black', linestyle='--')
    ax.plot(xlim, [0, 0], [0, 0], color='black', linestyle='--')
    
    ax.set_xlabel('VEHICLE_MAKE')
    ax.set_ylabel('VEHICLE_YEAR')
    ax.set_zlabel('VEHICLE_COLOR')
    plt.savefig("clustering/cluster_stats/visualization/vehicle_cluster_visu.jpg")
    plt.close()
    

def location_visualization(df: pd.DataFrame):
    df['LOCATION_STREET_NUMBER_STREET_ID'] = pd.to_numeric(df['LOCATION_STREET_NUMBER_STREET_ID'], errors='coerce')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df['LOCATION_STREET_NUMBER_STREET_ID'],
        df['LOCATION_CITY_CITY'],
        s=df['Percentage'] * 1000,
        c=np.arange(len(df)),
        cmap='viridis',
        alpha=0.8
    )
    
    plt.xlabel('LOCATION_STREET_NUMBER_STREET_ID')
    plt.ylabel('LOCATION_CITY_CITY')
    plt.title('Location Visualization')
    
    plt.colorbar(label='Index')
    
    plt.savefig("clustering/cluster_stats/visualization/location_cluster_visu.jpg")
    plt.close()


def officer_visualization(df: pd.DataFrame):
    
    encoder = LabelEncoder()

    df['OFFICER_ID_OFFICER_ID'] = encoder.fit_transform(df['OFFICER_ID_OFFICER_ID'])
    
    scaler = MinMaxScaler()
    df[['OFFICER_SUPERVISOR_SUPERVISOR_ID', 'OFFICER_ID_OFFICER_ID', 'OFFICER_ASSIGNMENT_OFF_DIST_ID']] = scaler.fit_transform(df[['OFFICER_SUPERVISOR_SUPERVISOR_ID', 'OFFICER_ID_OFFICER_ID', 'OFFICER_ASSIGNMENT_OFF_DIST_ID']])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        df['OFFICER_SUPERVISOR_SUPERVISOR_ID'],
        df['OFFICER_ID_OFFICER_ID'],
        df['OFFICER_ASSIGNMENT_OFF_DIST_ID'],
        s=df['Percentage'] * 1000,
        c=np.arange(len(df)),
        cmap='viridis',
        alpha=0.8
    )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    ax.plot([0, 0], [0, 0], zlim, color='black', linestyle='--')
    ax.plot([0, 0], ylim, [0, 0], color='black', linestyle='--')
    ax.plot(xlim, [0, 0], [0, 0], color='black', linestyle='--')

    ax.set_xlabel('SUPERVISOR_ID')
    ax.set_ylabel('OFFICER_ID')
    ax.set_zlabel('OFF_DIST_ID')
    ax.set_title('Officer Visualization')

    plt.savefig("clustering/cluster_stats/visualization/officer_cluster_visu.jpg")
    plt.close()        
    

def action_visualization(df: pd.DataFrame):
    # Put FIOFS_TYPE columns back together - ignore P since always 0
    df["UNKNOWN_FIELD_TYPE_FIOFS_TYPE_F"] = df["UNKNOWN_FIELD_TYPE_FIOFS_TYPE_F"].replace({"0":"", "1":"F"})
    df["UNKNOWN_FIELD_TYPE_FIOFS_TYPE_I"] = df["UNKNOWN_FIELD_TYPE_FIOFS_TYPE_I"].replace({"0":"", "1":"I"})
    df["UNKNOWN_FIELD_TYPE_FIOFS_TYPE_S"] = df["UNKNOWN_FIELD_TYPE_FIOFS_TYPE_S"].replace({"0":"", "1":"S"})
    df["UNKNOWN_FIELD_TYPE_FIOFS_TYPE_O"] = df["UNKNOWN_FIELD_TYPE_FIOFS_TYPE_O"].replace({"0":"", "1":"O"})
    df['FIOFS_TYPE'] = df[['UNKNOWN_FIELD_TYPE_FIOFS_TYPE_F', 'UNKNOWN_FIELD_TYPE_FIOFS_TYPE_I', 'UNKNOWN_FIELD_TYPE_FIOFS_TYPE_S']].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
    
    encoder = LabelEncoder()
    df["FIOFS_TYPE"] = encoder.fit_transform(df["FIOFS_TYPE"])
    df["INCIDENT_REASON_STOP_REASONS"] = encoder.fit_transform(df["INCIDENT_REASON_STOP_REASONS"])
    df["INCIDENT_REASON.1_FIOFS_REASONS"] = encoder.fit_transform(df["INCIDENT_REASON.1_FIOFS_REASONS"])    
    scaler = MinMaxScaler()
    df["FIOFS_TYPE"] = scaler.fit_transform(df[["FIOFS_TYPE"]])
    df["INCIDENT_REASON_STOP_REASONS"] = scaler.fit_transform(df[["INCIDENT_REASON_STOP_REASONS"]])  
    df["INCIDENT_REASON.1_FIOFS_REASONS"] = scaler.fit_transform(df[["INCIDENT_REASON.1_FIOFS_REASONS"]])
    
    df["Scalar_1"] = df["DISPOSITION_OUTCOME_F"].apply(lambda x: 1 if x == "1" else -1)
    df["Scalar_2"] = df["DISPOSITION_OUTCOME_O"].apply(lambda x: 1 if x == "1" else -1)   
    df["Scalar_3"] = df["DISPOSITION_OUTCOME_S"].apply(lambda x: 1 if x == "1" else -1)
    
    df["FIOFS_TYPE"] *= df["Scalar_1"]
    df["INCIDENT_REASON_STOP_REASONS"] *= df["Scalar_2"]
    df["INCIDENT_REASON.1_FIOFS_REASONS"] *= df["Scalar_3"]   
     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        df["FIOFS_TYPE"],
        df["INCIDENT_REASON_STOP_REASONS"],
        df["INCIDENT_REASON.1_FIOFS_REASONS"],
        s=df["Percentage"] * 1000,
        c=np.arange(len(df)),
        cmap='tab20c',
        alpha=0.8 
    )
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    ax.plot([0, 0], [0, 0], zlim, color='black', linestyle='--')
    ax.plot([0, 0], ylim, [0, 0], color='black', linestyle='--')
    ax.plot(xlim, [0, 0], [0, 0], color='black', linestyle='--')
    
    ax.set_xlabel('FIOFS_TYPE')
    ax.set_ylabel('STOP_REASON')
    ax.set_zlabel('FIOFS_REASONS')
    plt.savefig("clustering/cluster_stats/visualization/action_cluster_visu.jpg")
    plt.close()    
    
    
    df.info()


def date_visualization(df: pd.DataFrame):
    df["Year_Year"] = df["Year_Year"].astype(float)
    df["Month_Month"] = df["Month_Month"].astype(float)
    df["Day_Day"] = df["Day_Day"].astype(float)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        df['Year_Year'],
        df['Month_Month'],
        df['Day_Day'],
        s=df['Percentage'] * 1000,
        c=np.arange(len(df)),
        cmap='viridis',
        alpha=0.8
    )


    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    ax.plot([0, 0], [0, 0], zlim, color='black', linestyle='--')
    ax.plot([0, 0], ylim, [0, 0], color='black', linestyle='--')
    ax.plot(xlim, [0, 0], [0, 0], color='black', linestyle='--')

    ax.set_xlabel('Year')
    ax.set_ylabel('Month')
    ax.set_zlabel('Day')
    ax.set_title('Date Visualization')

    plt.savefig("clustering/cluster_stats/visualization/date_cluster_visu.jpg")
    plt.close()        


if __name__ == "__main__":
    
    path = "clustering/first_clustering.csv"    
    df = read_csv_file(path)
    
    #df.iloc[:, 1:]
    
    #group_cluster_analysis(df=df)
    
    warnings.simplefilter(action='ignore', category=PerformanceWarning)

    print("Only second round.")

    df = second_clustering(df=df, run_type=3)
    save_df_to_csv(df=df, output_filename="final_values.csv")
    
    analyse_feature(df=df, feature="Cluster_kmeans")