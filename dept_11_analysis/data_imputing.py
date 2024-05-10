import pandas as pd
import numpy as np



def data_imputing(df: pd.DataFrame, show_results: bool =False):
  # Univariate imputing methods
  prob_distribution_sex = simple_probalistic_imputer(df=df, feature_name=("SUBJECT_GENDER", "SEX"))
  prob_distribution_date = simple_probalistic_imputer(df=df, feature_name=("INCIDENT_DATE", "FIO_DATE"))
  prob_distribution_priors = simple_probalistic_imputer(df=df, feature_name=("SUBJECT_DETAILS", "PRIORS"))
  prob_distribution_search = simple_probalistic_imputer(df=df, feature_name=("SEARCH_CONDUCTED", "SEARCH"))  
  prob_distribution_basis = simple_probalistic_imputer(df=df, feature_name=("SEARCH_REASON", "BASIS"))
  prob_distribution_stop_reasons = simple_probalistic_imputer(df=df, feature_name=("INCIDENT_REASON", "STOP_REASONS"))
  prob_distribution_outcome = simple_probalistic_imputer(df=df, feature_name=("DISPOSITION", "OUTCOME"))
  prob_distribution_age = simple_probalistic_imputer(df=df, feature_name=("OFFICER_AGE", "AGE_AT_FIO_CORRECTED"))
  
  not_considered_list = ["NO VEHICLE INVOLVED"]
  prob_distribution_veh_make = complex_probalistic_imputer(df=df, feature_name=("VEHICLE_MAKE", "VEH_MAKE"), not_considered_values=not_considered_list)
  prob_distribution_veh_year = complex_probalistic_imputer(df=df, feature_name=("VEHICLE_YEAR", "VEH_YEAR_NUM"), not_considered_values=not_considered_list)
  prob_distribution_veh_color = complex_probalistic_imputer(df=df, feature_name=("VEHICLE_COLOR", "VEH_COLOR"), not_considered_values=not_considered_list)
  prob_distribution_veh_occ = complex_probalistic_imputer(df=df, feature_name=("VEHICLE_DETAILS", "VEH_OCCUPANT"), not_considered_values=not_considered_list)
  prob_distribution_veh_state = complex_probalistic_imputer(df=df, feature_name=("VEHICLE_DETAILS.1", "VEH_STATE"), not_considered_values=not_considered_list)
  
  # Multivariate imputing methods
  multivariate_bayesian_imputer(df=df, target_feature=("SUBJECT_RACE", "DESCRIPTION"), cond_feature=("SUBJECT_DETAILS.2", "COMPLEXION"))
  multivariate_bayesian_imputer(df=df, target_feature=("SUBJECT_DETAILS.2", "COMPLEXION"), cond_feature=("SUBJECT_RACE", "DESCRIPTION"))
  _ = simple_probalistic_imputer(df=df, feature_name=("SUBJECT_RACE", "DESCRIPTION"))
  _ = simple_probalistic_imputer(df=df, feature_name=("SUBJECT_DETAILS.2", "COMPLEXION"))  
  
  
  multivariate_bayesian_imputer(df=df, target_feature=("OFFICER_SUPERVISOR", "SUPERVISOR_ID"), cond_feature=("OFFICER_ID", "OFFICER_ID"))
  multivariate_bayesian_imputer(df=df, target_feature=("OFFICER_ID", "OFFICER_ID"), cond_feature=("OFFICER_SUPERVISOR", "SUPERVISOR_ID")) 
  _ = simple_probalistic_imputer(df=df, feature_name=("OFFICER_ID", "OFFICER_ID"))
  _ = simple_probalistic_imputer(df=df, feature_name=("OFFICER_SUPERVISOR", "SUPERVISOR_ID"))
  
  
  multivariate_bayesian_imputer(df=df, target_feature=("LOCATION_CITY", "CITY"), cond_feature=("LOCATION_STREET_NUMBER", "STREET_ID"))
  _ = simple_probalistic_imputer(df=df, feature_name=("LOCATION_CITY", "CITY"))
  
  
  
  if show_results:
    print("")
    print("*******************")
    print("Results from feature_filtering_2.py")
    print("*******************")
    print("")
    print(f"The probalbity distribution for feature ('SUBJECT_GENDER', 'SEX') is:")
    print(prob_distribution_sex)
    print(f"The probalbity distribution for feature ('INCIDENT_DATE', 'FIO_DATE') is:")
    print(prob_distribution_date)
    print(f"The probalbity distribution for feature ('SUBJECT_DETAILS', 'PRIORS') is:")
    print(prob_distribution_priors)
    print(f"The probalbity distribution for feature ('SEARCH_CONDUCTED', 'SEARCH') is:")
    print(prob_distribution_search)    
    print(f"The probalbity distribution for feature ('SEARCH_REASON', 'BASIS') is:")
    print(prob_distribution_basis)
    print(f"The probalbity distribution for feature ('INCIDENT_REASON', 'STOP_REASONS') is:")
    print(prob_distribution_stop_reasons)
    print(f"The probalbity distribution for feature ('DISPOSITION', 'OUTCOME') is:")
    print(prob_distribution_outcome)   
    print(f"The probalbity distribution for feature ('OFFICER_AGE', 'AGE_AT_FIO_CORRECTED') is:")
    print(prob_distribution_age)
    
    print(f"The probalbity distribution for feature ('VEHICLE_MAKE', 'VEH_MAKE') is:")
    print(prob_distribution_veh_make)    
    print(f"The probalbity distribution for feature ('VEHICLE_YEAR', 'VEH_YEAR_NUM') is:")
    print(prob_distribution_veh_year)    
    print(f"The probalbity distribution for feature ('VEHICLE_COLOR', 'VEH_COLOR') is:")
    print(prob_distribution_veh_color)    
    print(f"The probalbity distribution for feature ('VEHICLE_DETAILS', 'VEH_OCCUPANT') is:")
    print(prob_distribution_veh_occ)    
    print(f"The probalbity distribution for feature ('VEHICLE_DETAILS.1', 'VEH_STATE') is:")
    print(prob_distribution_veh_state)
  
  
        
def simple_probalistic_imputer(df: pd.DataFrame, feature_name: str):
  # Ensure reproducibility
  np.random.seed(17)
  
  prob_distribution = df[feature_name].value_counts(normalize=True)
  unique_values = prob_distribution.index.tolist()
  probabilites = prob_distribution.values
  generated_values = np.random.choice(unique_values, size=df[feature_name].isnull().sum(), p=probabilites)
  df.loc[df[feature_name].isnull(), feature_name] = generated_values
  
  return prob_distribution

   
def complex_probalistic_imputer(df: pd.DataFrame, feature_name: str, not_considered_values: list):
  # Ensure reproducibility
  np.random.seed(42)
  
  filtered_df = df[~df[feature_name].isin(not_considered_values)]

  prob_distribution = filtered_df[feature_name].value_counts(normalize=True)
  unique_values = prob_distribution.index.tolist()
  probabilites = prob_distribution.values
  generated_values = np.random.choice(unique_values, size=df[feature_name].isnull().sum(), p=probabilites)
  df.loc[df[feature_name].isnull(), feature_name] = generated_values
  
  return prob_distribution


def multivariate_bayesian_imputer(df: pd.DataFrame, target_feature: str, cond_feature: str):
  # Ensure reproducibility
  np.random.seed(42)
  
  for cond_value in df[cond_feature].dropna().unique():
    filtered_df = df[df[cond_feature] == cond_value]
    target_prob_distribution = filtered_df[target_feature].value_counts(normalize=True)
    unique_target_values = target_prob_distribution.index.tolist()
    probabilities = target_prob_distribution.values
    
    if not unique_target_values:
      # print(f"For conditional value {cond_value} no target values available. No imputing.")
      continue
    
    # Get the indicies from the original mask to ensure matching.
    missing_mask = df[target_feature].isnull() & (df[cond_feature] == cond_value)
    missing_indicies = missing_mask[missing_mask].index
    
    generated_values = np.random.choice(unique_target_values, size=missing_mask.sum(), p=probabilities)
    df.loc[missing_indicies, target_feature] = generated_values
  