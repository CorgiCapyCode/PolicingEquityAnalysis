import pandas as pd

def save_df_to_csv(df: pd.DataFrame, output_filename: str):
    '''
    Saves the dataframe as CSV-file.
    '''
    df.to_csv(output_filename, index=True, mode="w")