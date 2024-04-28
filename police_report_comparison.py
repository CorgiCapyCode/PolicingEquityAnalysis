from typing import Dict, List
import pandas as pd
from standard_functions import save_df_to_csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def add_sum_of_usages(df: pd.DataFrame):
    df_with_sum = df.copy()
    df_with_sum["Total"] = df_with_sum.sum(axis=1)
    return df_with_sum


def average_of_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return np.mean(df.values[np.triu_indices_from(df, k=1)])


def check_for_features_and_add_to_dataframe(
    df: pd.DataFrame,
    df_dict: Dict[str, pd.DataFrame]
):
    for dept_name, dept_df in df_dict.items():
        for feature in df.index:
            if feature in dept_df.columns:
                df.at[feature, dept_name] = 1
            else:
                df.at[feature, dept_name] = 0


def create_dataframe(
    feature_list: list,
    department_names: list
) -> pd.DataFrame:
    return pd.DataFrame(columns=department_names, index=feature_list)


def create_department_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr()


def get_unique_feature_list(df_dict: Dict[str, pd.DataFrame]) -> List[str]:
    unique_features: set[str] = set()
    for df in df_dict.values():
        unique_features.update(df.columns)
    return list(unique_features)


def read_police_data_files(path_list: list) -> Dict[str, pd.DataFrame]:
    df_dictionary = {}
    for path in path_list:
        dept_name = path.split("/")[-2]
        df = pd.read_csv(path, low_memory=False)
        df_dictionary[dept_name] = df
    return df_dictionary


def plot_and_save_heatmap(correlation_matrix: pd.DataFrame, save_name: str):
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Deptartment Correlation")
    plt.xlabel("Departments")
    plt.ylabel("Departments")
    plt.savefig(f"{save_name}.jpg")
    plt.show()


def police_report_comparison_main():
    # region - Create unique feature list from all police reports
    # Ignoring the 49-009_UOF.csv since it is equal to the 49-0009_UOF.csv
    police_reports = [
        "raw_data/cpe-data/Dept_11-00091/11-00091_Field-Interviews_2011-2015.csv",
        "raw_data/cpe-data/Dept_23-00089/23-00089_UOF-P.csv",
        "raw_data/cpe-data/Dept_24-00013/24-00013_UOF_2008-2017_prepped.csv",
        "raw_data/cpe-data/Dept_24-00098/24-00098_Vehicle-Stops-data.csv",
        "raw_data/cpe-data/Dept_35-00016/35-00016_UOF-OIS-P.csv",
        "raw_data/cpe-data/Dept_35-00103/35-00103_UOF-OIS-P_prepped.csv",
        "raw_data/cpe-data/Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv",
        "raw_data/cpe-data/Dept_37-00049/37-00049_UOF-P_2016_prepped.csv",
        "raw_data/cpe-data/Dept_49-00009/49-00009_UOF.csv",
        "raw_data/cpe-data/Dept_49-00033/49-00033_Arrests_2015.csv",
        "raw_data/cpe-data/Dept_49-00035/49-00035_Incidents_2016.csv",
        "raw_data/cpe-data/Dept_49-00081/49-00081_Incident-Reports_2012_to_May_2015.csv"
    ]
    df_dictionary = read_police_data_files(path_list=police_reports)
    # print(df_dictionary.keys())

    unique_feature_list = get_unique_feature_list(df_dict=df_dictionary)
    # print(unique_feature_list)

    department_names = list(df_dictionary.keys())
    feature_analysis_dataframe = create_dataframe(
        feature_list=unique_feature_list,
        department_names=department_names
    )
    # print(feature_analysis_dataframe)

    check_for_features_and_add_to_dataframe(
        df=feature_analysis_dataframe,
        df_dict=df_dictionary
    )
    # print(feature_analysis_dataframe)
    feature_analysis_df_with_sum = add_sum_of_usages(
        feature_analysis_dataframe
    )
    # print(feature_analysis_df_with_sum)
    save_df_to_csv(
        df=feature_analysis_df_with_sum,
        output_filename="police_report_comparison/feature_analysis_dataframe.csv"
    )
    # endregion
    # region - Check police reports for reports using the same features

    corr_matrix = create_department_correlation_matrix(
        feature_analysis_dataframe
    )

    save_df_to_csv(
        df=corr_matrix,
        output_filename="police_report_comparison/original_corr_matrix.csv"
    )
    average_corr = average_of_correlation_matrix(corr_matrix)
    # print(average_corr)
    plot_and_save_heatmap(
        correlation_matrix=corr_matrix,
        save_name="police_report_comparison/original_department_correlation"
    )
    # endregion


if __name__ == "__main__":
    police_report_comparison_main()
    print("Police report analysis ended")
