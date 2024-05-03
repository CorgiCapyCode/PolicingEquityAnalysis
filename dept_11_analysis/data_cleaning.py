from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def feature_value_cleaning(df: pd.DataFrame, threshold: float =30, feature_value_modification: list =[], show_results: bool =False):
    
    number_of_dropped_data_points, data_completion_perc = drop_not_filled_data(df=df, threshold=threshold)
    original_unique_value_counts, num_unique_values = get_unique_value_df_for_features(df=df)

    # simple standard modification: replace a value with another one
    for feature_name, modification in feature_value_modification:
        df = data_preprocessing(df=df, feature_name=feature_name, value_modification=modification)

    # complex modification: specific for a feature   
    modify_location_full_street(df=df)
    
    
    
    


    # Last step!!!
    # fill_missing_with_na(df=df)
    new_unique_value_counts, new_num_unique_values = get_unique_value_df_for_features(df=df)
    print(new_num_unique_values)
    

    if show_results:       
        print(f"Dropped data points because of missing input: {number_of_dropped_data_points}")
        plot_histogram_for_dp_completness(dp_completness=data_completion_perc)
        print("Histrogram for data completness saved.") 
        
        save_dataframes_to_csv(dict_with_df=original_unique_value_counts, name="_original_unique_values")
        print("Number of unique values:")
        print(num_unique_values)
        # save_dataframes_to_csv(dict_with_df=new_unique_value_counts, name="_new_unique_values")
    
    return df


def data_preprocessing(df: pd.DataFrame, feature_name: str, value_modification: list) -> pd.DataFrame:
    for orginal_value, target_value in value_modification:
        # df[feature_name].replace(orginal_value, target_value, inplace=True)
        df[feature_name] = df[feature_name].replace(orginal_value,target_value)
    return df


def drop_not_filled_data(df: pd.DataFrame, threshold: float) -> int:
    data_completion_perc = df.notna().mean(axis=1) * 100
    data_to_drop = data_completion_perc <= threshold
    number_of_data_to_be_dropped = data_to_drop.sum()
    df.drop(df[data_to_drop].index, inplace=True)
    return number_of_data_to_be_dropped, data_completion_perc
  

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


def modify_intersection_street_numbers(entry: str): #(df: pd.DataFrame, feature_name: str):
    # if " at " in entry and entry[0].isdigt():
    if entry[0].isdigit():
        for i, character in enumerate(entry):
            if not character.isdigit():
                entry = entry[i:]
                break
    return entry
    '''
    progress_bar_street_numbers = tqdm(total=len(df), desc="Processing st. no. removal from intersections.", unit="entries")
    for i, entry in df[feature_name].items():
        entry = df.loc[i, feature_name]
        entry = str(entry)
        if " at " in entry and entry[0].isdigit():
            for j, character in enumerate(entry):
                if not character.isdigit():
                    #df.loc[i, feature_name] = entry[j:]
                    entry = entry[j:]
                    break
        df.loc[i, feature_name] = entry.strip()
        progress_bar_street_numbers.update(1)
    
    progress_bar_street_numbers.close()
    '''


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
    
    #modify_intersection_street_numbers(df=df, feature_name=("LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION", "LOCATION"))
    #sort_intersection_names(df=df, feature_name=("LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION", "LOCATION"))


def save_dataframes_to_csv(dict_with_df: dict):
    for column, dataframe in dict_with_df.items():
        dataframe.to_csv(f"dept_11_analysis/data_files/{column}_unique_values.csv")


def sort_intersection_names(entry: str): # (df: pd.DataFrame, feature_name: str):
    parts = entry.split(" at ")
    stripped_parts = [part.strip() for part in parts]
    sorted_parts = sorted(stripped_parts)
    sorted_entry = " at ".join(sorted_parts)
    return sorted_entry
    
    '''
    progress_bar_intersection_sort = tqdm(total=len(df), desc="Sorting intersection streets.", unit="entries")
    for i, entry in df[feature_name].items():
        if " at " in entry:
            parts = entry.split(" at ")
            sorted_parts = sorted(parts)
            sorted_entry = " at ".join(sorted_parts)
            df.loc[i, feature_name] = sorted_entry
        progress_bar_intersection_sort.update(1)
    progress_bar_intersection_sort.close()
    '''

def plot_histogram_for_dp_completness(dp_completness: pd.Series):
    plt.hist(dp_completness, bins=100, edgecolor="black")
    plt.xlabel("Filling Grade [%]")
    plt.ylabel("Frequency")
    plt.title("Filling Grade Distribution")
    plt.savefig("dept_11_analysis/histogram_data_completness.jpg")