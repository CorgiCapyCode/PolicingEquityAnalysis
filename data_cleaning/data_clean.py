# Module containg all functions for Data Cleaning
# See: DEPT_11_POLICING_REPORT_ANALYSIS

from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt

def feature_value_cleaning(df: pd.DataFrame, threshold: float =30, feature_value_modification: list =[], show_results: bool =False):
    
    # Get List of unique values and their counts.
    original_unique_value_counts, num_unique_values = get_unique_value_df_for_features(df=df)

    # Simple standard modification: replace a value with another one
    for feature_name, modification in feature_value_modification:
        df = simple_data_preprocessing(df=df, feature_name=feature_name, value_modification=modification)

    # Complex modification: specific for a feature
    
    # Remove street numbers from intersections
    # Sort intersections alphabetically
    modify_location_full_street(df=df)
    
    # Remove time from the FIO_DATE, since it is always 0:00
    # Set unplausible dates to n/a
    modify_date_and_time(df=df)
    
    # Remove special characters from entries
    # Write all in upper case
    # Adjust most common color namings (standardize)
    # Adjust most common clothing namings (standardize)
    modify_clothing(df=df)
    
    # Fill data based on other columns with NO VEHICLE INVOLVED values
    # Write all in upper case
    # Remove signs
    # Adjust model names -> not used, due to different values and details
    modify_vehicle_features(df=df)
    
    # Filter for unplausible ages: 10 <= x <= 80
    modfiy_ages(df=df)
    
    # Remove signs
    # Split at ,
    # Replace namings using simple standard cleaning
    modify_fiofs_reasons(df=df)
    post_processing_list =[
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("209A INTIMIDATION OF WITNESS", "209A")]),
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("A&B PO", "A&B")]),
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("ASSAULTW", "ASSAULT")]),
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("DISTURBING", "DISTURBANCE")]),
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("DISTURBING PUBLIC ASSEMBLY", "DISTURBANCE")]),
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("IMPERSONATING PO", "IMPERSONATION")]),
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("IMPERSONATING", "IMPERSONATION")]),
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("LARCENY OF LEASED OR RENTED PROPERTY", "LARENCY")]),
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("WARRANT ARREST", "WARRANTS")]),
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("CITY ORDINANCES", "CITY ORDINANCE")]),
        (("INCIDENT_REASON.1", "FIOFS_REASONS"), [("CITY ORDINANCE OTHER", "CITY ORDINANCE")]),
    ]
    for feature_name, modification in post_processing_list:
        df = simple_data_preprocessing(df=df, feature_name=feature_name, value_modification=modification)
    
    # Fill missing data
    fill_missing_with_na(df=df)
    
    # Drop missing data below threshold (30% filling)
    number_of_dropped_data_points, data_completion_perc = drop_not_filled_data(df=df, threshold=threshold)
    print(f"Dropped data points because of missing input: {number_of_dropped_data_points}")
    
    # Get update about unique values
    new_unique_value_counts, new_num_unique_values = get_unique_value_df_for_features(df=df)
    
    if show_results:
        plot_histogram_for_dp_completness(dp_completness=data_completion_perc)
        print("Histrogram for data completness saved in data_cleaning.") 
    
    return original_unique_value_counts, num_unique_values, df, new_unique_value_counts, new_num_unique_values


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
        df[feature_name] = df[feature_name].replace(orginal_value, target_value)
    return df  


def modify_location_full_street(df: pd.DataFrame) -> pd.DataFrame:
    feature_name = ("LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION", "LOCATION")    
    
    df[feature_name] = df[feature_name].str.lstrip()
    df[feature_name] = df[feature_name].apply(modify_intersection_numbers)
    df[feature_name] = df[feature_name].str.lstrip()
    df[feature_name] = df[feature_name].apply(sort_intersection_names)


def modify_intersection_numbers(entry: str):
    if pd.isna(entry):
        return pd.NA
    if " at " in entry:
        if entry[0].isdigit():
            for i, character in enumerate(entry):
                if not character.isdigit():
                    entry = entry[i:]
                    break
    return entry


def sort_intersection_names(entry: str) -> str:
    if pd.isna(entry):
        return pd.NA
    if " at " in entry:
        street_split = entry.split(" at ")
        stripped_streets = [street.strip() for street in street_split]
        sorted_streets = sorted(stripped_streets)
        entry = " at ".join(sorted_streets)
    return entry


def modify_date_and_time(df: pd.DataFrame):
    feature_name = ("INCIDENT_DATE", "FIO_DATE")
    df[feature_name] = df[feature_name].str.replace(" 0:00", "")

    df[feature_name] = pd.to_datetime(df[feature_name], errors="coerce", format="%m/%d/%y")
    start_date = pd.to_datetime("2011-01-01")
    end_date = pd.to_datetime("2015-12-31")
    
    df.loc[(df[feature_name] < start_date) | (df[feature_name] > end_date), feature_name] = pd.NA


def modify_clothing(df: pd.DataFrame):
    feature_name = ("SUBJECT_DETAILS.1", "CLOTHING")
    df[feature_name] = df[feature_name].str.lstrip()
    df[feature_name] = df[feature_name].apply(remove_signs)
    df[feature_name] = df[feature_name].apply(writte_all_in_upper_case)
    df[feature_name] = df[feature_name].apply(adjust_color_naming_for_clothing)
    df[feature_name] = df[feature_name].apply(adjust_cloths_naming_for_clothing)


def remove_signs(entry: str) -> str:
    if pd.isna(entry):
        return pd.NA
    
    sings_correction = {
        "/": ", ",
        ".": "",
        ";": "",
        "(":"",
        ")": "",
        "\"": "",
    }
    for original_value, corrected_value in sings_correction.items():
        if original_value in entry:
            entry = entry.replace(original_value, corrected_value)    

    return entry


def writte_all_in_upper_case(entry: str) -> str:
    if pd.isna(entry):
        return pd.NA
    return entry.upper()


def adjust_color_naming_for_clothing(entry: str) -> str:
    if pd.isna(entry):
        return pd.NA
    
    color_correction = {
        "BLK": "BLACK",
        "WHT": "WHITE",
        "GRY": "GREY",
        "BLU": "BLUE",
        "BRN": "BROWN",
        "DRK": "DARK",
        "WHTE": "WHITE",
        "GRN": "GREEN",
        "BLUEE": "BLUE",
        "BKL" : "BLACK",
        "B LUE": "BLUE",
    }
    for original_value, corrected_value in color_correction.items():
        if original_value in entry:
            entry = entry.replace(original_value, corrected_value)
    return entry


def adjust_cloths_naming_for_clothing(entry: str) -> str:
    if pd.isna(entry):
        return pd.NA
    
    cloths_correction = {
        "JCK": "JACKET",
        "JKT": "JACKET",
        "SHT": "SHIRT",
        "SWTSHT": "SWEATSHIRT",
        "JNS": "JEANS",
        "HOODY": "HOODIE",
        "TSHIRT": "T-SHIRT",
        "SWEAT-SHIRT": "SWEATSHIRT",
        "SNKRS": "SNEAKERS",
        "T SHIRT": "T-SHIRT",
        "SWEAT SHIRT": "SWEATSHIRT"
        
    }
    for original_value, corrected_value in cloths_correction.items():
        if original_value in entry:
            entry = entry.replace(original_value, corrected_value)
    return entry


def modify_vehicle_features(df: pd.DataFrame):
    vehicle_related_columns = [
        ("VEHICLE_DETAILS.1", "VEH_STATE"),
        ("VEHICLE_COLOR", "VEH_COLOR"),
        ("VEHICLE_YEAR", "VEH_YEAR_NUM"),
        ("VEHICLE_MAKE", "VEH_MAKE"),
        ("VEHICLE_DETAILS", "VEH_OCCUPANT"),
        ("VEHICLE_MODEL", "VEH_MODEL")
    ]
    
    search_not_v = ~df[[("SEARCH_CONDUCTED", "SEARCH")]].isin(["VP", "V"]).any(axis=1)
        
    all_empty_data_points = df[vehicle_related_columns].isna().all(axis=1)
    final_condition = search_not_v & all_empty_data_points
    df.loc[final_condition, vehicle_related_columns] = "NO VEHICLE INVOLVED"
    
    df[("VEHICLE_MODEL", "VEH_MODEL")] = df[("VEHICLE_MODEL", "VEH_MODEL")].apply(writte_all_in_upper_case)
    df[("VEHICLE_MODEL", "VEH_MODEL")] = df[("VEHICLE_MODEL", "VEH_MODEL")].apply(remove_signs)
    # df[("VEHICLE_MODEL", "VEH_MODEL")] = df[("VEHICLE_MODEL", "VEH_MODEL")].apply(adjust_vehicle_model)
    
# NOT USED
def adjust_vehicle_model(entry: str) -> str:
    if pd.isna(entry):
        return pd.NA
    
    door_numbers = {
        "4": "4 ",
        " DR": " DOOR",
        "  ": " "
    }
    for original_value, corrected_value in door_numbers.items():
        if original_value in entry:
            entry = entry.replace(original_value, corrected_value)    
    
    return entry
# END NOT USED


def modfiy_ages(df: pd.DataFrame):
    feature_name = ("OFFICER_AGE", "AGE_AT_FIO_CORRECTED")
    df[feature_name] = df[feature_name].apply(filter_unplausible_ages)


def filter_unplausible_ages(entry: float) -> float:
    if pd.isna(entry):
        return pd.NA
    
    if 10 <= entry <= 80:
        return entry
    else:
        return pd.NA


def modify_fiofs_reasons(df: pd.DataFrame):
    feature_name = ("INCIDENT_REASON.1", "FIOFS_REASONS")
    df[feature_name] = df[feature_name].apply(remove_signs)
    df[feature_name] = df[feature_name].apply(lambda x: x.split(",")[0].strip())


def fill_missing_with_na(df: pd.DataFrame):
    df.fillna(value=pd.NA, inplace=True)


def drop_not_filled_data(df: pd.DataFrame, threshold: float) -> int:
    data_completion_perc = df.notna().mean(axis=1) * 100
    data_to_drop = data_completion_perc <= threshold
    number_of_data_to_be_dropped = data_to_drop.sum()
    df.drop(df[data_to_drop].index, inplace=True)
    return number_of_data_to_be_dropped, data_completion_perc


def plot_histogram_for_dp_completness(dp_completness: pd.Series):
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.hist(dp_completness, bins=100, edgecolor="black")
    plt.xlabel("Filling Grade [%]")
    plt.ylabel("Frequency")
    plt.title("Filling Grade Distribution")
    plt.savefig("data_cleaning/histogram_data_completness.jpg")


def save_dataframes_to_csv(dict_with_df: dict, name: str, sub_directory_name: str):
    for column, dataframe in dict_with_df.items():
        dataframe.to_csv(f"{sub_directory_name}/{column}{name}.csv")




















