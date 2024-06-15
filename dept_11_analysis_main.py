import pandas as pd

from data_exploration.data_pre_filtering import pre_filtering_dataset
from data_cleaning.data_clean import feature_value_cleaning, get_unique_value_df_for_features, save_dataframes_to_csv
from feature_selec.feature_selection import select_features
from final_preparation.final_prep import feature_final_preparation, feature_scaling
from data_imputation.data_imputing import impute_data
from stats.base_statistics import calculate_univariate_statistics, calculate_multivariate_statistics
from clustering.data_clustering import clustering, second_clustering, group_cluster_analysis

def read_csv_file(path) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1])

def save_df_to_csv(df: pd.DataFrame, output_filename: str):
    df.to_csv(output_filename, index=True, mode="w")


def dept_11_analysis_main():
    show_results = True
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
    
    # Features that will use simple value adjustments.
    # 1. Feature name
    # 2. List of pairs (old value -> new value)
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
    # Apply standard feature cleaning by replacing specific value (above) with specific value (above)
    # Run special cleanings for:
    # ("LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION", "LOCATION")
    # Date and Time
    # Clothing
    # Vehicle Features
    # Ages
    # FIOFS_Reasons
    original_unique_value_counts, num_unique_values, df, new_unique_value_counts, new_num_unique_values = feature_value_cleaning(
        df=df,
        threshold=threshold_to_drop,
        feature_value_modification=simple_feature_value_modification_list,
        show_results=show_results
    )
    
    if show_results:
        print("")
        print("*******************")
        print("Results from data_clean.py")
        print("*******************")
        print("")
        save_dataframes_to_csv(dict_with_df=original_unique_value_counts, name="_original_unique_values", sub_directory_name="data_cleaning/original_unique_value_lists")
        print("Original unique value lists saved in data_cleaning/original_unique_value_lists")
        print("Number of unique values:")
        print(num_unique_values)
        save_dataframes_to_csv(dict_with_df=new_unique_value_counts, name="_new_unique_values", sub_directory_name="data_cleaning/cleaned_unique_value_lists")
        print("Updated unique value lists saved in data/cleaning/cleaned_unique_value_lists")
        print("Number of new unique values:")
        print(new_num_unique_values)
        
        num_complete_data_points = df.dropna().shape[0]
        print(f"Number of data points with no missing data: {num_complete_data_points}")
        df_complete_values = df.dropna(how="all")
        save_df_to_csv(df=df_complete_values, output_filename="data_cleaning/data_points_with_complete_data.csv")
        save_df_to_csv(df=df, output_filename="data_cleaning/dataframe_after_cleaning.csv")    
    

    print("------------------------------------")
    print("Start Feature Selection")
    print("------------------------------------") 
    
    # Drop features with inconsistent data.
    # Compare pairs of features and drop them.
    # For comparison: Use three methods of Chi-Square Test
    select_features(df=df, show_results=show_results)
    
    if show_results:
        new_unique_value_counts, new_num_unique_values = get_unique_value_df_for_features(df=df)
        print(f"New unique value list: {new_num_unique_values}")        
        num_complete_data_points = df.dropna().shape[0]
        print(f"Number of data points with no missing data after feature selection: {num_complete_data_points}")
        df.info()
        save_df_to_csv(df=df, output_filename="feature_selec/df_with_selected_features.csv")

    print("------------------------------------")
    print("Start Imputation")
    print("------------------------------------") 
  
    # Simple Probabilistic Imputer:
    # Imputes the missing values based on the probability distribution of values of a feature.
    # Multivariate imputing methods:
    # Complex Imputer:
    # Imputes the missing values based on the probability distribution of exclusive values of a feature. Ignores specific values.
    # Dependent Imputer:
    # Ensures data consistency by imputing a specific value based on the value of another column.
    # In this case: Ensure that the searched object cannot be a vehicle (value v or vp) if there is no vehicle involved.
    # Multivariate Bayesian Imputer:
    # Imputes values using the conditional probability distribution of a feature and a conditional feature.
    impute_data(df=df, show_results=show_results)
    
    if show_results:
        print("DF info:")
        print(df.info())
        save_df_to_csv(df=df, output_filename="data_imputation/df_with_imputed_data.csv")

    print("------------------------------------")
    print("Start Final Preparation")
    print("------------------------------------")
    run_comparison = False

    mutual_comparison_results, chi_comparison_results = feature_final_preparation(df=df, run_all=run_comparison)
    
    if mutual_comparison_results is None:
        pass
    elif not mutual_comparison_results.empty:
        print("Saving mutual feature comparison results.")
        save_df_to_csv(df=mutual_comparison_results, output_filename="final_preparation/mutual_feature_comparison_results.csv")

    if chi_comparison_results is None:
        pass
    elif not chi_comparison_results.empty:
        print("Saving chi feature comparison results.")
        save_df_to_csv(df=chi_comparison_results, output_filename="final_preparation/chi_feature_comparison_results.csv")    
    
    
    if show_results:
        print("")
        print("FINAL FEATURE INFORMATION")
        print("")
        new_unique_value_counts, new_num_unique_values = get_unique_value_df_for_features(df=df)
        print(f"New unique value list: {new_num_unique_values}")     
        
      
    print("------------------------------------")
    print("Start Base Statistics")
    print("------------------------------------") 
    
    univariate_statistics = calculate_univariate_statistics(df=df)
    if show_results:
        for key, value in univariate_statistics.items():
            output_filename = f"stats/statistic_values/univariate_statistics_{key}.csv"
            save_df_to_csv(df=value, output_filename=output_filename)
    
    # Creates 2D plots (option for 3D as well) -> time intensive process!
    # calculate_multivariate_statistics(df=df)

    print("------------------------------------")
    print("Start Feature Scaling")
    print("------------------------------------") 

    feature_scaling(df=df)
    save_df_to_csv(df=df, output_filename="final_preparation/input_for_clustering.csv")

    print("------------------------------------")
    print("Start Clustering")
    print("------------------------------------") 
    # Run types:
    # 1: The dataframe as it is right now -> No good results. Might crash due to high complexity of dataframe.
    # 2: The dataframe without STREET_ID and OFFICER_ID. -> No good resutls. Less complex than 1.
    # 3: Grouping the features. -> Under investigation.
    run_type = 3
    clustered_df = clustering(df=df, run_type=run_type)
    save_df_to_csv(df=clustered_df, output_filename="clustering/first_clustering.csv")

    group_cluster_analysis(clustered_df)
    
    final_df = second_clustering(clustered_df, run_type=run_type)
    
    
    if final_df is None:
        pass
    else:
        save_df_to_csv(final_df, "final_values.csv")
    
    
if __name__ == "__main__":
    dept_11_analysis_main()
    print("Finished process")
    
