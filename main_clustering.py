from dept_11_analysis.clustering_functions import clustering
import pandas as pd
from dept_11_analysis.data_cleaning import get_unique_value_df_for_features
from dept_11_analysis_main import save_df_to_csv, read_csv_file




def main_clustering():
    show_results = True
    path = "prepared_dataframe.csv"    
    df = read_csv_file(path)
    
    print("Start Clustering")
    if show_results:
        print("Input stats for Clustering")
        df.info()
        clustering_unique_counts, clustering_unique_values = get_unique_value_df_for_features(df=df)
        print("Input unique value list for clustering:")
        print(clustering_unique_values)
    
    testing = clustering(df=df)
    save_df_to_csv(df=testing, output_filename="testing.csv")
    

if __name__ == "__main__":
    main_clustering()
    print("Finished process")