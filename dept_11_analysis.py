import pandas as pd

def delete_feature(df: pd.DataFrame, feature):
    pass

def filter_for_redundant(df: pd.DataFrame) -> list:
    redundant_features = []
    number_of_features = len(df.columns)
    for i in range(number_of_features):
        for j in range(i+1, number_of_features):
            feature_1 = df.iloc[:, i]
            feature_2 = df.iloc[:, j]
            # print(f"Comparing {df.columns[i]} with {df.columns[j]}")
            if feature_1.equals(feature_2):
                redundant_features.append((df.columns[i], df.columns[j]))               
    return redundant_features

def general_df_info(df: pd.DataFrame):
    print("General information:")
    print(df.info())   

def read_csv_file(path) -> pd.DataFrame:
    return pd.read_csv(path, header=[0, 1])

def dept_11_analysis_main():
    path = "raw_data/Dept_11-00091/11-00091_Field-Interviews_2011-2015.csv"    
    df = read_csv_file(path)
    
    general_df_info(df)
    
    redundant_features = filter_for_redundant(df)
    print(redundant_features)
    
    # print(df)


if __name__ == "__main__":
    dept_11_analysis_main()