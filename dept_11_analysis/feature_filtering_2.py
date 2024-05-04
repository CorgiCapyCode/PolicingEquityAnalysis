import pandas as pd


def further_feature_filtering(df: pd.DataFrame):
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
