import pandas as pd

from dept_11_analysis.data_pre_filtering import redundant_feature_filtering, delete_duplicated_data_points
from dept_11_analysis.data_cleaning import feature_value_cleaning, get_unique_value_df_for_features
from dept_11_analysis.feature_filtering_2 import further_feature_filtering
from dept_11_analysis.feature_selection import feature_selection
from dept_11_analysis.data_imputing import data_imputing
from dept_11_analysis.base_statistics import calculate_univariate_statistics

def read_csv_file(path) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1])

def save_df_to_csv(df: pd.DataFrame, output_filename: str):
    '''
    Saves the dataframe as CSV-file.
    '''
    df.to_csv(output_filename, index=True, mode="w")


def dept_11_analysis_main():
    show_results = False
    threshold_to_drop = 30
    
    path = "raw_data/Dept_11-00091/11-00091_Field-Interviews_2011-2015.csv"    
    df = read_csv_file(path)
    # general_df_info(df)

    # region - Feature and Data Pre-Filtering
    redundant_feature_filtering(df=df, show_results=show_results)
    delete_duplicated_data_points(df=df, show_results=show_results)
    
    # endregion
    # region - Data Cleaning
    
    simple_feature_value_modification_list = [
        (("SUBJECT_GENDER", "SEX"), [("UNKNOWN", pd.NA)]),
        (("SUBJECT_RACE", "DESCRIPTION"), [("NO DATA ENTERED", pd.NA)]),
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
    
    # complex_feature_value_modification_list = [
    #    ("LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION", "LOCATION")
    #    ("INCIDENT_DATE", "FIO_DATE")
    #    ("SUBJECT_DETAILS.1", "CLOTHING")
    #    ("VEHICLE_MAKE", "VEH_MAKE") and other vehicle related features
    #    ("OFFICER_AGE", "AGE_AT_FIO_CORRECTED")
    # ]
    # see data_cleaning for complex feature value modification
    
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
    
    # endregion
    
    # region - Feature Filtering - 2nd iteration
    further_feature_filtering(df=df, show_results=show_results)
    
    if show_results:
        new_unique_value_counts, new_num_unique_values = get_unique_value_df_for_features(df=df)
        print(f"New unique value list: {new_num_unique_values}")        
        num_complete_data_points = df.dropna().shape[0]
        print(f"Number of data points with no missing data: {num_complete_data_points}")
    # endregion
    
    # region - Imputation
    data_imputing(df=df, show_results=show_results)
    if show_results:
        print("DF info:")
        print(df.info())
    # endregion

    # region - Feature Selection
    comparison_results = feature_selection(df=df)
    save_df_to_csv(df=comparison_results, output_filename="comparison_results.csv")
    # endregion
    
    # region - Base statistics
    calculate_univariate_statistics(df=df)
    # endregion
    
    #save_df_to_csv(df=df, output_filename="test.csv")
    # df.info()
    
if __name__ == "__main__":
    dept_11_analysis_main()
    print("Finished process")
    
