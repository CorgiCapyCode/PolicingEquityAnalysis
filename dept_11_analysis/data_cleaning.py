from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


def feature_value_cleaning(df: pd.DataFrame, threshold: float =30, feature_value_modification: list =[], show_results: bool =False):
    
    number_of_dropped_data_points, data_completion_perc = drop_not_filled_data(df=df, threshold=threshold)
    original_unique_value_counts, num_unique_values = get_unique_value_df_for_features(df=df)

    # simple standard modification: replace a value with another one
    for feature_name, modification in feature_value_modification:
        df = simple_data_preprocessing(df=df, feature_name=feature_name, value_modification=modification)

    # complex modification: specific for a feature   
    # modify_location_full_street(df=df)
    # modify_date_and_time_entries(df=df)
    

    new_unique_value_counts, new_num_unique_values = get_unique_value_df_for_features(df=df)
    print(new_num_unique_values)
    
    
    
    
    # Last step!!!
    # fill_missing_with_na(df=df)    

    if show_results:       
        print(f"Dropped data points because of missing input: {number_of_dropped_data_points}")
        plot_histogram_for_dp_completness(dp_completness=data_completion_perc)
        print("Histrogram for data completness saved.") 
        
        save_dataframes_to_csv(dict_with_df=original_unique_value_counts, name="_original_unique_values")
        print("Number of unique values:")
        print(num_unique_values)
        # save_dataframes_to_csv(dict_with_df=new_unique_value_counts, name="_new_unique_values")
    
    return df

# region 1-Simple operations
def drop_not_filled_data(df: pd.DataFrame, threshold: float) -> int:
    data_completion_perc = df.notna().mean(axis=1) * 100
    data_to_drop = data_completion_perc <= threshold
    number_of_data_to_be_dropped = data_to_drop.sum()
    df.drop(df[data_to_drop].index, inplace=True)
    return number_of_data_to_be_dropped, data_completion_perc


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


def simple_data_preprocessing(df: pd.DataFrame, feature_name: str, value_modification: list) -> pd.DataFrame:
    for orginal_value, target_value in value_modification:
        df[feature_name] = df[feature_name].replace(orginal_value,target_value)
    return df  


def fill_missing_with_na(df: pd.DataFrame):
    df.fillna(value=pd.NA, inplace=True)

# endregion
# region 2-Complex operations
def modify_location_full_street(df: pd.DataFrame):
    progress_bar_location_street = tqdm(total=len(df), desc="Adjusting Intersections: ", unit="entries")
    feature_name = ("LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION", "LOCATION")
    for i, entry in df[feature_name].items():
        entry = str(entry)
        if " at " in entry:
            entry = modify_intersection_street_numbers(entry=entry)
            entry = sort_intersection_names(entry=entry)
        df.loc[i, feature_name] = entry.strip()
        progress_bar_location_street.update(1)
    progress_bar_location_street.close()


def modify_intersection_street_numbers(entry: str) -> str:
    if entry[0].isdigit():
        for i, character in enumerate(entry):
            if not character.isdigit():
                entry = entry[i:]
                break
    return entry


def sort_intersection_names(entry: str) -> str:
    parts = entry.split(" at ")
    stripped_parts = [part.strip() for part in parts]
    sorted_parts = sorted(stripped_parts)
    sorted_entry = " at ".join(sorted_parts)
    return sorted_entry


def modify_date_and_time_entries(df: pd.DataFrame):
    progress_bar_date_and_time = tqdm(total=len(df), desc="Processing date and time: ", unit="entries")
    feature_name = ("INCIDENT_DATE", "FIO_DATE")
    
    for i, entry in df[feature_name].items():
        entry = str(entry)
        entry = clear_unplausible_dates(entry=entry)
        if entry is not pd.NA:
            entry = remove_time_from_date(entry=entry)
        df.loc[i, feature_name] = entry
        
        progress_bar_date_and_time.update(1)
            
    progress_bar_date_and_time.close()


def clear_unplausible_dates(entry: str) -> str:
    date = datetime.strptime(entry, "%m/%d/%y %H:%M")
    if 2011 <= date.year <= 2015:
        return entry
    else:
        return pd.NA

def remove_time_from_date(entry: str) -> str:
    date_parts = entry.split()
    date = date_parts[0]
    return date













# endregion
# region 3-Show details
def save_dataframes_to_csv(dict_with_df: dict):
    for column, dataframe in dict_with_df.items():
        dataframe.to_csv(f"dept_11_analysis/data_files/{column}_unique_values.csv")


def plot_histogram_for_dp_completness(dp_completness: pd.Series):
    plt.hist(dp_completness, bins=100, edgecolor="black")
    plt.xlabel("Filling Grade [%]")
    plt.ylabel("Frequency")
    plt.title("Filling Grade Distribution")
    plt.savefig("dept_11_analysis/histogram_data_completness.jpg")

# endregion