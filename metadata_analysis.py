from typing import Dict
import os
import pandas as pd
import matplotlib.pyplot as plt
from dept_11_analysis_main import save_df_to_csv


def add_filename_as_feature(
    metadata: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    '''
    Adds a feature called Filename to the dataframe.
    Uses the relational path for input values.
    '''
    for file_path, df in metadata.items():
        df["Filename"] = file_path
    return metadata


def categorize_dataframe(
    df: pd.DataFrame,
    categories: list,
    new_feature_name: str
) -> pd.DataFrame:
    '''
    Categorizes the file path. Used to categorize the statistical groups
    (e.g. education-attainment) and the departments.
    '''
    categorized_df = df.copy()
    categorized_df[new_feature_name] = categorized_df["Filename"].apply(
        lambda filename: next(
            (category for category in categories if category in filename),
            "No-category"))
    return categorized_df


def combine_dataframes(metadata: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    '''
    Takes the dictionary entries and creates one dataframe.
    '''
    combined_df = pd.concat(metadata.values(), ignore_index=True)
    return combined_df


def create_feature_dict(df: pd.DataFrame, feature: str) -> Dict[str, int]:
    '''
    Creates a dictionary with unique values for one specific feature.
    '''
    unique_values = df[feature].unique()
    feature_dict = {value: idx for idx, value in enumerate(unique_values)}
    return feature_dict


def deltete_no_category_entries(
    df: pd.DataFrame,
    feature_name: str
) -> pd.DataFrame:
    '''
    Deletes the No-category. Cleans the data since two departments stored
    some files doubled.
    '''
    filtered_df = df[df[feature_name] != "No-category"]
    return filtered_df


def filter_dataframe_for_value(df: pd.DataFrame,
                               feature_name: str, value
                               ) -> pd.DataFrame:
    '''
    Filters the data for a specific category.
    Is used to prepare the visualization.
    '''
    filtered_df = df[df[feature_name] == value]
    return filtered_df


def plot_features(df: pd.DataFrame, x_axis: str, y_axis: str, save_name: str):
    '''
    Creates scatter plots for different feature
    combinations and saves them as JPEG-files.
    '''
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_axis], df[y_axis])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{x_axis} vs. {y_axis}')
    plt.xticks(rotation=90)
    # plt.tight_layout()
    plt.savefig(f"metadata_analysis/{save_name}.jpg")
    # plt.show()


def search_and_read_metadata_files(root_dir: str) -> Dict[str, pd.DataFrame]:
    '''
    Searches and reads all csv-files which are in the root directory.
    The content of the csv-files is stored as a pandas dataframe in a
    dictionary.
    '''
    metadata = {}
    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("_metadata.csv"):
                file_path = os.path.join(dirpath, file)
                df = pd.read_csv(file_path, names=["Code", "Description"])
                metadata[file_path] = df
    return metadata


def metadata_analysis_main():
    '''
    The main function for the metadata analysis.
    '''
    root_dir = "raw_data/cpe-data"

    metadata = search_and_read_metadata_files(root_dir=root_dir)

    metadata_with_filename = add_filename_as_feature(metadata)

    # region - Check for unique codes and descriptions in the metadata.
    combined_df = combine_dataframes(metadata_with_filename)
    save_df_to_csv(
        df=combined_df,
        output_filename="metadata_analysis/combined_df.csv"
        )
    code_dict = create_feature_dict(df=combined_df, feature="Code")
    description_dict = create_feature_dict(
        df=combined_df,
        feature="Description"
        )
    print(f"Code dict of length: {len(code_dict)}")
    # print(code_dict)
    print(f"Description dict of length: {len(description_dict)}")
    # print(description_dict)
    # endregion

    # region - Check for same codes per category
    statistic_categories = [
        "ACS_education-attainment-over-25",
        "ACS_education-attainment",
        "ACS_employment",
        "ACS_income",
        "ACS_owner-occupied-housing",
        "ACS_poverty",
        "ACS_race-sex-age"
        ]
    stat_categories_name = "Stat_Categories"
    df_categorized_by_statistic_types = categorize_dataframe(
        df=combined_df,
        categories=statistic_categories,
        new_feature_name=stat_categories_name
        )
    reduced_df = deltete_no_category_entries(
        df=df_categorized_by_statistic_types,
        feature_name=stat_categories_name
        )

    dept_list = [
        "Dept_11-00091",
        "Dept_23-00089",
        "Dept_24-00013",
        "Dept_24-00098",
        "Dept_35-00016",
        "Dept_35-00103",
        "Dept_37-00027",
        "Dept_37-00049",
        "Dept_49-00009",
        "Dept_49-00033",
        "Dept_49-00035",
        "Dept_49-00081",
        ]
    dept_name = "Dept_name"
    df_categorized_by_departments = categorize_dataframe(
        df=reduced_df,
        categories=dept_list,
        new_feature_name=dept_name
        )
    save_df_to_csv(df=df_categorized_by_departments,
                   output_filename="metadata_analysis/dept_categories.csv"
                   )
    # endregion

    # region - Visualization
    dfs_filtered_by_category = {}
    for category in statistic_categories:
        dfs_filtered_by_category[category] = filter_dataframe_for_value(
            df=df_categorized_by_departments,
            feature_name=stat_categories_name,
            value=category
            )

    for category, df in dfs_filtered_by_category.items():
        save_name = f"{category.replace('/', '_')}_Code_plot"
        plot_features(df=df,
                      x_axis="Code",
                      y_axis=dept_name,
                      save_name=save_name
                      )
    for category, df in dfs_filtered_by_category.items():
        save_name = f"{category.replace('/', '_')}_Description_plot"
        plot_features(df=df,
                      x_axis="Description",
                      y_axis=dept_name,
                      save_name=save_name
                      )
    # endregion


if __name__ == "__main__":
    metadata_analysis_main()
    print("Metadata analysis ended")
