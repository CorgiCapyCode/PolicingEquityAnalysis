import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
import numpy as np


def further_feature_filtering(df: pd.DataFrame, show_results: bool =False):
    # Dropping the second unique identifier from the dataset.
    df.drop(("INCIDENT_UNIQUE_IDENTIFIER.1", "FIO_ID"), axis=1, inplace=True)
    # (LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION, LOCATION) and (LOCATION_STREET_NUMBER, STREET_ID)
    # contain information about the location.
    # See as well the feature description.
    # Due to the high number of unique values in the first feature,
    # this will be dropped and accepting the loss of information (complete address)
    df.drop(("LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION", "LOCATION"), axis=1, inplace=True)
    # Dropping (VEHICLE_MODEL, VEH_MODEL) because of its high variation of namings etc.
    df.drop(("VEHICLE_MODEL", "VEH_MODEL"), axis=1, inplace=True)
    # Dropping the clothing feature, due to its remaing inconsistency, even after cleaning.
    df.drop(("SUBJECT_DETAILS.1", "CLOTHING"), axis=1, inplace=True)
    # Dropping (OFFICER_ETHNICITY, ETHNICITY) because of inconsistent data and not interpretable values.
    df.drop(("OFFICER_ETHNICITY", "ETHNICITY"), axis=1, inplace=True)
    # Dropping feature (UNKNOWN_FIELD_TYPE, TERRORISM) due to low variance.
    df.drop(("UNKNOWN_FIELD_TYPE", "TERRORISM"), axis=1, inplace=True)
    # Compare (LOCATION_DISTRICT, DIST) & (LOCATION_DISTRICT.1, DIST_ID)
    # Compare (OFFICER_ASSIGNMENT.1, OFF_DIST) & (OFFICER_ASSIGNMENT, OFF_DIST_ID)
    p_value_loc_dist, cont_table_loc_dist = chi_square_test(df=df, feature_1=("LOCATION_DISTRICT", "DIST"), feature_2=("LOCATION_DISTRICT.1", "DIST_ID"))
    ohe_p_value_loc_dist = chi_square_test_one_hot_encoder(df=df, feature_1=("LOCATION_DISTRICT", "DIST"), feature_2=("LOCATION_DISTRICT.1", "DIST_ID"))
    le_p_value_loc_dist = chi_square_test_label_encoder(df=df, feature_1=("LOCATION_DISTRICT", "DIST"), feature_2=("LOCATION_DISTRICT.1", "DIST_ID"))
    max_p_value_loc_dist = max(p_value_loc_dist, ohe_p_value_loc_dist, le_p_value_loc_dist)
    if max_p_value_loc_dist < 0.01:
        df.drop(("LOCATION_DISTRICT", "DIST"), axis=1, inplace=True)
      
    p_value_off_dist, cont_table_off_dist = chi_square_test(df=df, feature_1=("OFFICER_ASSIGNMENT", "OFF_DIST_ID"), feature_2=("OFFICER_ASSIGNMENT.1", "OFF_DIST"))
    ohe_p_value_off_dist = chi_square_test_one_hot_encoder(df=df, feature_1=("OFFICER_ASSIGNMENT", "OFF_DIST_ID"), feature_2=("OFFICER_ASSIGNMENT.1", "OFF_DIST"))
    le_p_value_off_dist = chi_square_test_label_encoder(df=df, feature_1=("OFFICER_ASSIGNMENT", "OFF_DIST_ID"), feature_2=("OFFICER_ASSIGNMENT.1", "OFF_DIST")) 
    max_p_value_off_dist = max(p_value_off_dist, ohe_p_value_off_dist, le_p_value_off_dist)    
    if max_p_value_off_dist < 0.01:
        df.drop(("OFFICER_ASSIGNMENT.1", "OFF_DIST"), axis=1, inplace=True)
        
    p_value_dist, cont_table_dist = chi_square_test(df=df,feature_1=("LOCATION_DISTRICT.1", "DIST_ID"), feature_2=("OFFICER_ASSIGNMENT", "OFF_DIST_ID"))
    ohe_p_value_dist = chi_square_test_one_hot_encoder(df=df,feature_1=("LOCATION_DISTRICT.1", "DIST_ID"), feature_2=("OFFICER_ASSIGNMENT", "OFF_DIST_ID"))
    le_p_value_dist = chi_square_test_label_encoder(df=df,feature_1=("LOCATION_DISTRICT.1", "DIST_ID"), feature_2=("OFFICER_ASSIGNMENT", "OFF_DIST_ID"))
    max_p_value_dist = max(p_value_dist, ohe_p_value_dist, le_p_value_dist)   
    if max_p_value_dist < 0.01:
        df.drop(("LOCATION_DISTRICT.1", "DIST_ID"), axis=1, inplace=True)

    if show_results:
        print("")
        print("*******************")
        print("Results from feature_filtering_2.py")
        print("*******************")
        print("")
        print(f"The p-value for the Location District information is: {max_p_value_loc_dist}")
        print("The related contigency matrix is:")
        print(cont_table_loc_dist)
        print("-------------")
        print(f"The p-value for the Officer District information is: {max_p_value_off_dist}")
        print("The related contigency matrix is:")
        print(cont_table_off_dist)
        print("-------------")
        print(f"The p-value for the District information is: {max_p_value_dist}")
        print("The related contigency matrix is:")
        print(cont_table_dist)            
    

def chi_square_test(df: pd.DataFrame, feature_1: str, feature_2: str):
    contigency_table = pd.crosstab(df[feature_1], df[feature_2])
    chi2_stat, p_value, _, _ = chi2_contingency(contigency_table)
    return p_value, contigency_table


def chi_square_test_one_hot_encoder(df: pd.DataFrame, feature_1: str, feature_2: str):
    encoded_data = pd.get_dummies(df[[feature_1, feature_2]], drop_first=True)
    chi2_stats, p_values = chi2(encoded_data, df[feature_1])
    p_value = np.mean(p_values)
    return p_value
    

def chi_square_test_label_encoder(df: pd.DataFrame, feature_1: str, feature_2: str):
    
    df_1 = df[[feature_1]]
    df_2 = df[feature_2]
    
    df_1.loc[:, feature_1] = LabelEncoder().fit_transform(df_1.loc[:, feature_1])
    
    selector = SelectKBest(chi2, k=len(df_1.columns))
    selector.fit(df_1, df_2)

    p_value = np.mean(selector.pvalues_)

    return p_value