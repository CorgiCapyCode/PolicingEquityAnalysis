from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt


def feature_value_cleaning(df: pd.DataFrame, show_results: bool =False):
    unique_value_counts, num_unique_values = get_unique_value_df_for_features(df)
    
    show_results = True
    if show_results:       
        save_dataframes_to_csv(unique_value_counts)
        print("Number of unique values:")
        print(num_unique_values)
    
    
    # Last step!!!
    fill_missing_with_na(df=df)
    

def fill_missing_with_na(df: pd.DataFrame):
    df.fillna(value=pd.NA, inplace=True)


def get_unique_value_df_for_features(df: pd.DataFrame) -> Tuple[dict, dict]:
    unique_value_counts = {}
    num_unique_values_dict = {}
    for column in df.columns:
        unique_values = df[column].unique()
        num_unique_values = len(unique_values)
        unique_values, value_counts = df[column].value_counts().index.tolist(), df[column].value_counts().tolist()
        num_unique_values = len(unique_values)
        unique_value_table = pd.DataFrame({'Unique_Values': unique_values, 'Value_Counts': value_counts})
        unique_value_counts[column] = unique_value_table
        num_unique_values_dict[column] = num_unique_values
    return unique_value_counts, num_unique_values_dict


def save_dataframes_to_csv(dict_with_df: dict):
    for column, dataframe in dict_with_df.items():
        dataframe.to_csv(f"dept_11_analysis/data_files/{column}_unique_values.csv")


def data_imputing(df: pd.DataFrame, threshold: float =30, show_results: bool =False):
    number_of_dropped_data_points, data_completion_perc = drop_not_filled_data(df=df, threshold=threshold)
    if show_results:
        print(f"Dropped data points because of missing input: {number_of_dropped_data_points}")
        plot_histogram_for_dp_completness(dp_completness=data_completion_perc)
        print("Histrogram for data completness saved.")


def drop_not_filled_data(df: pd.DataFrame, threshold: float) -> int:
    data_completion_perc = df.notna().mean(axis=1) * 100
    data_to_drop = data_completion_perc <= threshold
    number_of_data_to_be_dropped = data_to_drop.sum()
    df.drop(df[data_to_drop].index, inplace=True)
    return number_of_data_to_be_dropped, data_completion_perc


def plot_histogram_for_dp_completness(dp_completness: pd.Series):
    plt.hist(dp_completness, bins=100, edgecolor="black")
    plt.xlabel("Filling Grade [%]")
    plt.ylabel("Frequency")
    plt.title("Filling Grade Distribution")
    plt.savefig("dept_11_analysis/histogram_data_completness.jpg")