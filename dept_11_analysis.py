import pandas as pd

from dept_11_analysis.data_pre_filtering import redundant_feature_filtering, delete_duplicated_data_points
from dept_11_analysis.data_cleaning import data_imputing, feature_value_cleaning
from standard_functions import save_df_to_csv

def read_csv_file(path) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1])


def dept_11_analysis_main():
    show_results = False
    threshold_to_drop = 30
    
    path = "raw_data/Dept_11-00091/11-00091_Field-Interviews_2011-2015.csv"    
    df = read_csv_file(path)
    # general_df_info(df)

    redundant_feature_filtering(df=df, show_results=show_results)
    
    delete_duplicated_data_points(df=df, show_results=show_results)
    
    feature_value_cleaning(df=df, show_results=show_results)
    
    # data_imputing(df=df, threshold=threshold_to_drop, show_results=True)

    
    #print("Final df.info:")
    #print(df.info())
    #print("final df:")
    #print(df)


    
if __name__ == "__main__":
    dept_11_analysis_main()
    print("Finished process")