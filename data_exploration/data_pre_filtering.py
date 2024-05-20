# Chapter 2.1 Data Exploration
import pandas as pd

def pre_filtering_dataset(df: pd.DataFrame, show_results: bool=False):
    redundant_feature_filtering(df=df, show_results=show_results)
    delete_duplicated_data_points(df=df, show_results=show_results)
    
    
def redundant_feature_filtering(df: pd.DataFrame, show_results: bool =False):
    redundant_features = filter_for_redundant_features(df)
    delete_feature(df=df, redundant_feature_list=redundant_features)
    if show_results:
        print("")
        print("*******************")
        print("Results from data_pre_filtering.py")
        print("*******************")
        print("")
        print("Redundant feature pairs:")
        print(redundant_features)
        print("Dataframe after removing features:")
        print(df)    


def filter_for_redundant_features(df: pd.DataFrame) -> list:
    redundant_features = []
    number_of_features = len(df.columns)
    for i in range(number_of_features):
        for j in range(i+1, number_of_features):
            feature_1 = df.iloc[:, i]
            feature_2 = df.iloc[:, j]
            if feature_1.equals(feature_2):
                redundant_features.append((df.columns[i], df.columns[j]))               
    return redundant_features


def delete_feature(df: pd.DataFrame, redundant_feature_list: tuple):
    for pair_of_features in redundant_feature_list:  
        if pair_of_features[1] in df.columns:
            del df[pair_of_features[1]]


def delete_duplicated_data_points(df: pd.DataFrame, show_results: bool =False):
    initial_rows = len(df)
    df.drop_duplicates(subset=df.columns[2:], inplace=True)
    new_rows = len(df)
    deleted_rows = initial_rows-new_rows
    if show_results:
        print(f"{deleted_rows} data points have been deleted from the dataset.")
        print("Dataframe after removing duplicants:")
        print(df)
