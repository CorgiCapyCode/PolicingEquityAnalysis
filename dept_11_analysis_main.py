import pandas as pd

from data_exploration.data_pre_filtering import pre_filtering_dataset

# from dept_11_analysis.data_pre_filtering import redundant_feature_filtering, delete_duplicated_data_points
from dept_11_analysis.data_cleaning import feature_value_cleaning, get_unique_value_df_for_features, save_dataframes_to_csv
from dept_11_analysis.feature_filtering_2 import further_feature_filtering
from dept_11_analysis.feature_selection import feature_selection, feature_scaling
from dept_11_analysis.data_imputing import data_imputing
from dept_11_analysis.base_statistics import calculate_univariate_statistics, calculate_multivariate_statistics
from dept_11_analysis.clustering_functions import clustering

def read_csv_file(path) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1])

def save_df_to_csv(df: pd.DataFrame, output_filename: str):
    df.to_csv(output_filename, index=True, mode="w")


def dept_11_analysis_main():
    show_results = False
    threshold_to_drop = 30
    
    path = "raw_data/Dept_11-00091/11-00091_Field-Interviews_2011-2015.csv"    
    df = read_csv_file(path)

    print("------------------------------------")
    print("Start Feature and Data Pre-Filtering")
    print("------------------------------------")
    # Find redundant columns and delete them.
    # Find duplicanted data rows and delete them.
    pre_filtering_dataset(df=df, show_results=show_results)
    
    if show_results:
        df.info()
        save_df_to_csv(df=df, output_filename="data_exploration/pre_filtered_dataframe.csv")
        print("Saved results of feature and data pre-filtering in /data_exploration/pre_filtered_dataframe.csv")

    print("------------------------------------")
    print("Start Data Cleaning")
    print("------------------------------------")

    simple_feature_value_modification_list = [
        (("SUBJECT_GENDER", "SEX"), [("UNKNOWN", pd.NA)]),
        (("SUBJECT_RACE", "DESCRIPTION"), [("NO DATA ENTERED", pd.NA)]), 
        (("SUBJECT_RACE", "DESCRIPTION"), [("B(Black)", "Black")]),
        (("SUBJECT_RACE", "DESCRIPTION"), [("W(White)", "White")]),
        (("SUBJECT_RACE", "DESCRIPTION"), [("H(Hispanic)", "Hispanic")]),
        (("SUBJECT_RACE", "DESCRIPTION"), [("A(Asian or Pacific Islander)","Asian or Pacific Islander")]),
        (("SUBJECT_RACE", "DESCRIPTION"), [("M(Middle Eastern or East Indian)", "Middle Eastern or East Indian")]),
        (("SUBJECT_RACE", "DESCRIPTION"), [("I(American Indian or Alaskan Native)", "American Indian or Alaskan Native")]),
        (("INCIDENT_REASON", "STOP_REASONS"), [("OTHER (SPECIFY)", "OTHER")]),
        (("SUBJECT_DETAILS.2", "COMPLEXION"), [("NO DATA ENTERED", pd.NA)]),
        (("VEHICLE_MAKE", "VEH_MAKE"), [("NO DATA ENTERED", pd.NA)]),
        (("VEHICLE_YEAR", "VEH_YEAR_NUM"), [(0, pd.NA), (2016, pd.NA), (2020, pd.NA), (2017, pd.NA)]),
        (("VEHICLE_COLOR", "VEH_COLOR"), [("NO DATA ENTERED", pd.NA)]),
        (("VEHICLE_DETAILS.1", "VEH_STATE"), [("NO DATA ENTERED", pd.NA), ("OTHER", pd.NA)]),
        (("OFFICER_SUPERVISOR", "SUPERVISOR_ID"), [(1, pd.NA)]),
        (("OFFICER_ID", "OFFICER_ID"), [(1, pd.NA), (2, pd.NA)]),
        (("LOCATION_CITY", "CITY"), [("NO DATA ENTERED", pd.NA)]),
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("INVESTIGATE", "INVESTIGATION")])
    ]
           
    df = feature_value_cleaning(
        df=df,
        threshold=threshold_to_drop,
        feature_value_modification=simple_feature_value_modification_list,
        show_results=show_results
    )
    
    if show_results:
        num_complete_data_points = df.dropna().shape[0]
        print(f"Number of data points with no missing data: {num_complete_data_points}")
        df_complete_values = df.dropna(how="all")
        save_df_to_csv(df=df_complete_values, output_filename="dept_11_analysis/data_files/only_complete_after_cleaning.csv")
        save_df_to_csv(df=df, output_filename="dept_11_analysis/data_files/after_data_cleaning.csv")
    print("End Data Cleaning")

    
    # region - Feature Filtering - 2nd iteration
    print("Start Featre Filtering - 2nd iteration")
    further_feature_filtering(df=df, show_results=show_results)
    if show_results:
        new_unique_value_counts, new_num_unique_values = get_unique_value_df_for_features(df=df)
        print(f"New unique value list: {new_num_unique_values}")        
        num_complete_data_points = df.dropna().shape[0]
        print(f"Number of data points with no missing data: {num_complete_data_points}")
    print("End Feature Filtering - 2nd iteration")
    # endregion
    
    # region - Imputation
    print("Start Imputation")    
    if show_results:
        df.info()
    data_imputing(df=df, show_results=show_results)
    if show_results:
        print("DF info:")
        print(df.info())
    print("End Imputation")
    # endregion

    # region - Feature Selection
    print("Start Feature Selection")
    comparison_results = feature_selection(df=df, run_all=False)
    if comparison_results is None:
        pass
    elif not comparison_results.empty:
        save_df_to_csv(df=comparison_results, output_filename="comparison_results.csv")

    if show_results:
        print("")
        print("FINAL FEATURE INFORMATION")
        print("")
        new_unique_value_counts, new_num_unique_values = get_unique_value_df_for_features(df=df)
        print(f"New unique value list: {new_num_unique_values}")     
    
          
   
    print("End Feature Selection")
    # endregion
    
    # region - Base statistics
    print("Start Base Statistics")
    
    univariate_statistics = calculate_univariate_statistics(df=df)
    if show_results:
        for key, value in univariate_statistics.items():
            output_filename = f"dept_11_analysis/statistics_and_graphs/univariate/univariate_statistics_{key}.csv"
            save_df_to_csv(df=value, output_filename=output_filename)
    
    # Creates 2D plots (option for 3D as well)
    # calculate_multivariate_statistics(df=df)
    print("End Base Statistics")
    
    # endregion
    # region - Feature Scaling
    
    feature_scaling(df=df)     
    
    # endregion
    
    print("")
    print("################################")
    print("Final df-info before clustering:")
    print("################################")
    print("")
    df.info()
    save_df_to_csv(df=df, output_filename="prepared_dataframe.csv")    

    # region - Clustering

    
    print("Start Clustering")
    
    if show_results:
        print("Input stats for Clustering")
        df.info()
        clustering_unique_counts, clustering_unique_values = get_unique_value_df_for_features(df=df)
        save_dataframes_to_csv(dict_with_df=clustering_unique_counts, name="final_unique_values", sub_directory_name="final_values")
        print("Input unique value list for clustering:")
        print(clustering_unique_values)
    
    testing = clustering(df=df, run_type=3)
    save_df_to_csv(df=testing, output_filename="testing_compl.csv")

    
    print("End Clustering")
    # endregion

    
if __name__ == "__main__":
    dept_11_analysis_main()
    print("Finished process")
    
