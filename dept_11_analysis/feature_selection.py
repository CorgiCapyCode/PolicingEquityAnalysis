import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from dept_11_analysis.feature_filtering_2 import chi_square_test, chi_square_test_label_encoder, chi_square_test_one_hot_encoder
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder

def feature_selection(df: pd.DataFrame, run_all: bool =True):
    print("")
    print("*****************************")
    print("Input for Feature Selection")
    print("*****************************")
    print("")
    df.info()
    
    split_date_and_time(df=df, feature=("INCIDENT_DATE", "FIO_DATE"))
    # Splitting features that use combinations of shortcuts (e.g. F, I, FI, ...)
    char_feature_split(df=df, feature_name=('UNKNOWN_FIELD_TYPE', 'FIOFS_TYPE'))
    char_feature_split(df=df, feature_name=('SEARCH_CONDUCTED', 'SEARCH'))
    char_feature_split(df=df, feature_name=('DISPOSITION', 'OUTCOME'))
    
    # Group insignificant features <25 occurences
    for feature_name in df.columns:
        group_insignificant_values(df=df, feature_name=feature_name)
    
    # Label encode binary feature - can be interpreted as "MALE?" 1: True, 0: False
    label_encode_feature(df=df, feature_name=('SUBJECT_GENDER', 'SEX'))
    
    if run_all:
        comparison_results = feature_comparison(df=df)
        plot_comparison_results(df=comparison_results)
    else: comparison_results = None
    
    print("")
    print("*****************************")
    print("Output for Feature Selection")
    print("*****************************")
    print("")
    df.info()
    
    return comparison_results
    

def feature_comparison(df: pd.DataFrame) -> pd.DataFrame:
    
    comp_df = df.copy()
    comp_df = comp_df.astype(str)
    feature_names = comp_df.columns[1:]
    comparison_results = pd.DataFrame(index=feature_names, columns=feature_names)
    for i in range(len(feature_names)):
        
        for j in range(i+1, len(feature_names)):
            feature_1 = feature_names[i]
            feature_2 = feature_names[j]
            # print(f"Comparing feature {feature_1}, no {i},  with feature {feature_2}, no {j}")
            mutual_value = mutual_info_comparison(df=comp_df, feature_1=feature_1, feature_2=feature_2)
            # p_value_1, _ = chi_square_test(df=comp_df, feature_1=feature_1, feature_2=feature_2)
            # p_value_2 = chi_square_test_label_encoder(df=comp_df, feature_1=feature_1, feature_2=feature_2)
            # p_value_3 = chi_square_test_one_hot_encoder(df=comp_df, feature_1=feature_1, feature_2=feature_2)
            # final_p_value = max(p_value_1, p_value_2, p_value_3)
            comparison_results.loc[feature_2, feature_1] = mutual_value
    return comparison_results
    


def mutual_info_comparison(df: pd.DataFrame, feature_1: str, feature_2: str) -> float:
    feature_1_values = df[feature_1].values
    feature_2_values = df[feature_2].values
    mutual_info_value = adjusted_mutual_info_score(feature_1_values, feature_2_values)
    # print(mutual_info_value)
    return mutual_info_value

    
def plot_comparison_results(df: pd.DataFrame):
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.astype(float), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Feature Mutual Information Comparison Results")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.savefig("dept_11_analysis/comparison_heatmap.jpg")


def split_date_and_time(df: pd.DataFrame, feature: str):
    df[("Year", "Year")] = df[feature].dt.year
    df[("Month", "Month")] = df[feature].dt.month
    # df[("Day", "Day")] = df[feature].dt.day
    df.drop(columns=[feature], inplace=True)


def char_feature_split(df: pd.DataFrame, feature_name: str):
    modified_feature = df[feature_name]
    unique_chars = set()
    for entry in modified_feature:
        if isinstance(entry, str):
            unique_chars.update(entry)
            
    for char in unique_chars:
        mask = modified_feature.apply(lambda x: char in str(x))
        new_column_name = (feature_name[0], f"{feature_name[1]}_{char}")
        df[new_column_name] = 0
        df.loc[mask, new_column_name] = 1
    
    df.drop(columns=[feature_name], inplace=True)        


def group_insignificant_values(df: pd.DataFrame, feature_name: str):
    value_counts = df[feature_name].value_counts()
    unique_values = value_counts[value_counts < 25].index.tolist()
    # Ensure that binary features are not considered.
    if len(unique_values) > 2:
        df.loc[df[feature_name].isin(unique_values), feature_name] = "OTHER"
        
def label_encode_feature(df: pd.DataFrame, feature_name: str):
    label_encoder = LabelEncoder()
    df[feature_name] = label_encoder.fit_transform(df[feature_name])