import pandas as pd
from fancyimpute import KNN as fancyKNN



def data_imputing(df: pd.DataFrame) -> pd.DataFrame:
  feature_name_list = df.columns.tolist()
  
  return df


    
def simple_stochastical_imputer(df: pd.DataFrame):
    pass


def complex_imputer(df: pd.DataFrame):
    pass
  