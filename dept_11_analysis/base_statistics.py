import pandas as pd
import matplotlib.pyplot as plt

def calculate_univariate_statistics(df: pd.DataFrame):
    
    for feature in df.columns[1:]:
        _ = analyse_feature(df=df, feature=feature)

def analyse_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    value_counts = df[feature].value_counts()
    total_count = value_counts.sum()
    percentage = value_counts/total_count
    unique_values_df = pd.DataFrame({"Count": value_counts, "Percentage": percentage})
    unique_values_df.index.name = "Unique Values"
    
    plt.figure(figsize=(16, 8))
    df[feature].value_counts().plot(kind="bar", color="skyblue")
    plt.title(f"Historgram for {feature}")
    plt.xlabel("Unique values")
    plt.ylabel("Counts")
    plt.savefig(f"dept_11_analysis/statistics_and_graphs/histogram_{feature}.jpg")
    plt.close()
    
    plt.figure(figsize=(12, 12))
    plt.pie(percentage, labels=percentage.index, autopct="%1.1f%%", startangle=140)
    plt.title(f"Pie Chart for {feature}")
    plt.axis("equal")
    plt.savefig(f"dept_11_analysis/statistics_and_graphs/pie_chart_{feature}.jpg")
    plt.close()
    return unique_values_df