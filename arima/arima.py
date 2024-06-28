import statsmodels.api as sm
import pandas as pd
import numpy as np
# Import typing option
from typing import Union
import warnings

def arima_aic(data: pd.DataFrame, aic_full: pd.DataFrame) -> None:
    """Train 3 times 3 ARIMA models with different p and q values and store the Akaike information criterion values in the
      dataframe passed as an argument.

    Args:
        data (_type_): Time series data to train the ARIMA models on.
        aic_full (_type_): DataFrame to store the AIC values of the ARIMA models.
    """
    warnings.simplefilter('ignore')
    # Based on prior analysis, we know that the seasonality is 24 hours
    seasonal_order = (1,0,1,24) # (Seasonal AR specification, Seasonal Integration order, Seasonal MA, Seasonal periodicity)
    # Iterate over all ARMA(p,q) models with p,q in [0,6]
    for p in range(3):
        for q in range(3):
            # Baseline model does not have any AR or MA terms
            if p == 0 and q == 0:
                continue

            # Estimate the model with no missing datapoints
            mod = sm.tsa.statespace.SARIMAX(_impute_missing_values(data),
                                            order=(p,0,q),
                                            seasonal_order=seasonal_order,
                                            enforce_invertibility=False)
            try:
                res = mod.fit(disp=False)
                aic_full.iloc[p,q] = res.aic
            except:
                aic_full.iloc[p,q] = np.nan


def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the dataframe with the last known value of the timeseries. If the previous value is missing, fill with the next known value.

        Args:
        df (pd.DataFrame): 1D timeseries given as a pandas dataframe with the values in the field specified by value_axis.
    """
    # Find last value that is not NaN or zero before a value that is NaN or zero
    df.replace(0, np.nan, inplace=True)
    df.ffill(inplace=True, axis=0)
    df.bfill(inplace=True, axis=0)
    return df


def _get_timeseries_for_node(data: pd.DataFrame, node: Union[str, int]) -> pd.DataFrame:
    """Get the timeseries for a specific node from the data.

    Args:
        data (pd.DataFrame): Dataframe containing the timeseries data.
        node (Union[str, int]): Node to get the timeseries for. Either passed as an LCLid or as the index of the node in
            the adjacency matrix. Note that the DataFrame should have the corresponding format to fit `node`.

    Returns:
        pd.DataFrame: Timeseries for the specified node.
    """
    if isinstance(node, (int, np.integer, str)):
        node_data = data[node]
    else:
        raise ValueError("Node should be either an integer (index in adjacency matrix) or a string (LCLid).")
    return node_data


def _build_arima_model(data, impute_data=False, **kwargs) -> sm.tsa.SARIMAX:
    """Build a SARIMAX model and fit it to the given data.

    Args:
        data (pandas.DataFrame): Data to fit the model to. Should only include the data to be used in the model.
        impute_data (bool, optional): If the timeseries has missing values, they can imputed by simply inserting the
        last known value or the first known value if the timeseries starts with missing data. Defaults to False.
        **kwargs: Additional arguments and to be passed to the statsmodels.tsa.SARIMAX model with their default values like
            `order`=(1,0,1),
            `seasonal_order`=(1,0,1,24),
            `trend`='const',
            `freq`='hour' etc.
        For a full list see https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html.

    Returns:
        sm.tsa.SARIMAX: Fitted SARIMAX model.
    """
    if impute_data:
        print("Imputing missing values.")
        data = _impute_missing_values(data)

    print("Create SARIMAX model.")
    sarimax = sm.tsa.statespace.SARIMAX(data,
                                        trend=kwargs.get("trend", "c"),
                                        order=kwargs.get("order", (1, 0, 1)),
                                        seasonal_order=kwargs.get("seasonal_order", (1, 0, 1, 24)),
                                        enforce_invertibility=False,
                                        freq=kwargs.get("freq", "h"),
    )
    print("Fitting SARIMAX model.")
    # sarimax_res = sarimax.fit(disp=False)
    sarimax_res = sarimax.fit(disp=False)
    print("Done.")
    return sarimax_res


def build_arima_model_for_node(data: pd.DataFrame, node: Union[str, int], impute_data=False, **kwargs) -> sm.tsa.statespace.MLEResults:
    """Build a SARIMAX model for a specific node and fit it to the given data.

    Args:
        data (pandas.DataFrame): Full dataframe containing the timeseries data of all nodes and index `DateTime`. Either with columns for each
            node by index of the adjacency matrix (`df_agg`) or with a column 'LCLid' containing the LCLids and columns for the timeseries data.
        node (Union[str, int]): Node to model. Either passed as an LCLid or as the index of the node in the adjacency
            matrix. Note that the DataFrame should have the corresponding format to fit the representation.
        impute_data (bool, optional): If the timeseries has missing values, they can imputed by simply inserting the
            last known value or the first known value if the timeseries starts with missing data. Defaults to False.
        **kwargs: Additional arguments and to be passed to the statsmodels.tsa.SARIMAX model with their default values like
            `order`=(1,0,1),
            `seasonal_order`=(1,0,1,24),
            `trend`='const',
            `freq`='hour' etc.
        For a full list see https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html.

    Returns:
        sm.tsa.statespace.MLEResults: Fitted SARIMAX model result for the specified node on which one can call `.predict(start, end)`.
    """
    node_data = _get_timeseries_for_node(data, node)
    return _build_arima_model(node_data, impute_data, **kwargs)



if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    sns.set_style("darkgrid")
    plt.rc("figure", figsize=(6, 6))
    plt.rc("font", size=13)

    filepath = "./uk-smart-meter-aggregated/"
    filename = "df_agg.pkl"
    # load pickle file
    with open(filepath+filename, "rb") as f:
        df = pickle.load(f)
    # Set seed for reproducibility
    seed = 1997
    np.random.seed(seed)
    random_households = np.random.choice(df.columns, 5)
    # Rows for DateTime = 2012-01-01 00:00:00 to 2012-12-31 23:00:00
    df_range = df[(df.index >= "2012-09-01 00:00:00") & (df.index <= "2012-12-31 23:00:00")]
    five_households = df_range[random_households]
    five_households.plot(subplots=True, figsize=(8, 10), title="Electricity consumption of 5 random households in 2012")
    train_start = np.argmax(five_households.index >= "2012-09-01 00:00:00")
    train_end = np.argmin(five_households.index < "2012-11-30 00:00:00")
    val_start = train_end + 1
    # Select node as
    node = int(random_households[1])

    # Build ARIMA model for node
    sarimax_res = build_arima_model_for_node(five_households, node, impute_data=True)

    predictions = []
    # Plot seven last 7 days of fitted data and 7 days ahead (24 hours * 7 = 168)
    predictions.append(sarimax_res.predict(start=train_end-168, end=val_start+168))

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    # five_households.iloc[:-720, :].plot(ax= axes, subplots=True, figsize=(8, 10), title="Electricity consumption of 5 random households in 2012")
    for i in range(1):
        axes.plot(five_households[node].iloc[val_start-168:val_start+168], label="Data", color="blue")
        # axes[i].plot(predictions[i], label="Forecast", color="red")
        axes.plot(predictions[i], label="Fitted data", color="orange")
        # Plot horizontal line between part of the training and part of the predicted data
        axes.axvline(x=five_households.index[train_end], color="black", linestyle="--", label="Start forecast")
        axes.legend()
        # turn off the x-axis labels
        axes.set_xlabel("")
    # plt.tight_layout()
    plt.tick_params(axis='x', labelrotation=45)
    plt.show()