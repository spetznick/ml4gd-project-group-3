#
import pandas as pd
import numpy as np
from typing import Tuple

def preprocess_csv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the given dataframe by converting the time column to datetime format, removing duplicate values (keeping the last one) and pivoting the table to a
        'DateTime' based indexing.

        Args:
        df (pd.DataFrame): DateTime-indexed pivotted dataframe with the values of 'KWH/hh (per half hour) ' in the field specified by columns 'LCLid'.
    """
    assert 'DateTime' in df.columns, "The DataFrame must contain a column named 'DateTime'."
    assert 'KWH/hh (per half hour) ' in df.columns, "The DataFrame must contain a column named 'KWH/hh (per half hour) '."
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['KWH/hh (per half hour) '] = pd.to_numeric(df['KWH/hh (per half hour) '], errors='coerce')
    # Might need to remove
    # df.drop(columns=['stdorToU'], inplace=True)
    #
    new_df = df.drop_duplicates(subset=['DateTime', 'LCLid'], keep='last')
    new_df = new_df.pivot(index='DateTime', columns='LCLid', values='KWH/hh (per half hour) ')

    return new_df


def find_largest_continuous_timeseries(df: pd.DataFrame, value_axis: str) -> Tuple[np.datetime64, np.datetime64]:
    """Find the longest continuous time series in the dataframe and return the start and end dates.

        Args:
        df (pd.DataFrame): 1D timeseries given as a pandas dataframe with the values in the field specified by value_axis.
    """
    pass


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the dataframe with the last known value of the timeseries. If the previous value is missing, fill with the next known value.

        Args:
        df (pd.DataFrame): 1D timeseries given as a pandas dataframe with the values in the field specified by value_axis.
    """
    # Find last value that is not NaN or zero before a value that is NaN or zero
    df.replace(0, np.nan, inplace=True)
    df.ffill(inplace=True, axis=0)
    df.bfill(inplace=True, axis=0)
    return df

