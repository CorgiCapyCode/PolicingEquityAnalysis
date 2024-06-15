import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import LabelEncoder

def calculate_univariate_statistics(df: pd.DataFrame) -> dict:
    univariate_statistics = {}
    for feature in df.columns[1:]:
        unique_values_df = analyse_feature(df=df, feature=feature)
        univariate_statistics[feature] = unique_values_df
    return univariate_statistics
    

def analyse_feature(df: pd.DataFrame, feature: str, location: str ="stats") -> pd.DataFrame:
    value_counts = df[feature].value_counts()
    total_count = value_counts.sum()
    percentage = value_counts/total_count
    unique_values_df = pd.DataFrame({"Count": value_counts, "Percentage": percentage})
    unique_values_df.index.name = "Unique Values"
    
    # Limiting the number of values added to the plots to ensure readability.
    # The dataframe is not influenced by this.
    top_values = value_counts.head(25)
    high_count_values = value_counts[value_counts / total_count >= 0.01]
    plotted_values = top_values.combine_first(high_count_values)
    plotted_values["MINORS"] = total_count - plotted_values.sum()
    plt.rcParams["font.family"] = "DejaVu Sans"
    
    plt.figure(figsize=(16, 8))
    plotted_values.plot(kind="bar", color="skyblue")
    plt.title(f"Historgram for {feature}")
    plt.xlabel("Unique values")
    plt.ylabel("Counts")
    plt.savefig(f"{location}/univariate_histograms/histogram_{feature}.jpg")
    plt.close()
    
    plt.figure(figsize=(12, 12))
    plotted_values.plot(kind="pie", autopct="%1.1f%%", startangle=140)
    plt.title(f"Pie Chart for {feature}")
    plt.axis("equal")
    plt.savefig(f"{location}/univariate_pie_charts/pie_chart_{feature}.jpg")
    plt.close()
    return unique_values_df

def calculate_multivariate_statistics(df: pd.DataFrame):
    encoder = LabelEncoder()
    encoded_df = df.copy()
    encoded_df = encoded_df.astype(str)
    for column in encoded_df.columns[1:]:
        encoded_df[column]= encoder.fit_transform(encoded_df[column])
        
    feature_names = encoded_df.columns[1:]
    combinations = itertools.combinations(feature_names, 2)
    for combination in combinations:
        plot_two_dimensional(encoded_df, *combination)
        


def plot_two_dimensional(df: pd.DataFrame, feature_1: str, feature_2: str):
    plt.figure(figsize=(16, 12))
    plt.scatter(df[feature_1], df[feature_2], alpha=0.5)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.title(f"Scatter Plot of {feature_1} and {feature_2}")
    plt.grid(True)
    plt.savefig(f"stats/multivariate_2D_scatters/2D_scatter_{feature_1}_{feature_2}.jpg")
    plt.close()
  

# Only run for specific!    
def plot_three_dimensional(df: pd.DataFrame, feature_1: str, feature_2: str, feature_3: str):

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df[feature_1], df[feature_2], df[feature_3], alpha=0.5)
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_zlabel(feature_3)
    ax.set_title(f"Scatter Plot of {feature_1}, {feature_2} and {feature_3}")
    plt.savefig(f"stats/multivariate_3D_scatters/3D_scatter_{feature_1}_{feature_2}_{feature_3}.jpg")
    plt.close()