import pandas as pd
import numpy as np
from collections import Counter
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

def rename_columns(df: pd.DataFrame, col_map: dict = None) -> pd.DataFrame:
    """
    Rename columns of a dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to rename the columns of.

    col_map: dict
        A dictionary mapping the current column names to the new column names.

    Returns
    -------
    df: pd.DataFrame
        The dataframe with renamed columns.
    """

    if col_map is None:
        # If col_map is None, just lowercase all the column names
        df.columns = df.columns.str.lower()
    else:
        df.rename(columns=col_map, inplace=True)

    return df

def time_processing(df: pd.DataFrame, time_column:  str = 'time') -> pd.DataFrame:
    """
    Process the time column of the dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to process the time column of.
    
    time_column: str
        The name of the column to process.

    Returns
    -------
    df: pd.DataFrame
        The dataframe with the processed time column.
    """

    # Convert to datetime
    df[time_column] = pd.to_datetime(df[time_column])

    # Extract hour and minute
    df['hour'] = df[time_column].dt.hour
    df['minute'] = df[time_column].dt.minute

    # Drop time column
    df.drop(time_column, axis=1, inplace=True)

    return df 

def convert_minutes(x: int) -> int:
    """
    Convert the given number of minutes to the nearest multiple of 5.

    Parameters
    ----------
    x: int
        The number of minutes to convert.

    Returns
    -------
    int
        The converted number of minutes.
    """

    # Define a list of multiples of 5 from 5 to 55
    min = list(range(5, 56, 5))

    # Check if the input value is greater than the largest multiple of 5
    if x in [56, 57, 58, 59]:
        return 0

    # Check if the input value is already a multiple of 5 or 0
    if x in min + [0]:
        return x

    # Find the nearest multiple of 5 greater than the input value
    for m in min:
        if x % m == x and x > m - 5:
            return m

def drop_null_cols(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Drops columns from a dataframe that have null value count greater than the threshold.
    
    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to process the time column of.

    threshold: int 
        The maximum allowed null value count for a column.
    
    Returns
    -------
    df: pd.DataFrame
        The dataframe with the processed time column.
    """
    # get the null value count for each column
    null_counts = df.isnull().sum()
    
    # get the column names where null value count is greater than the threshold
    cols_to_drop = null_counts[null_counts > threshold].index.tolist()
    
    # drop the columns and return the modified dataframe
    return df.drop(cols_to_drop, axis=1)

def ordinal_encoder(df: pd.DataFrame, feats: pd.DataFrame.columns) -> pd.DataFrame:
    """
    Perform Ordinal Encoder to the dataset

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to perform the ordinal encoder on.

    feats: pd.DataFrame.columns
        The columns to perform the ordinal encoder on.

    Returns
    -------
    df: pd.DataFrame
        The dataframe with the ordinal encoded columns.
    """
    for feat in feats:
        feat_val = list(np.arange(df[feat].nunique()))
        feat_key = list(df[feat].sort_values().unique())
        feat_dict = dict(zip(feat_key, feat_val))
        df[feat] = df[feat].map(feat_dict)
    return df

def knn_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute the null values using KNN Imputer.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to impute the null values of.

    Returns
    -------
    df: pd.DataFrame
        The dataframe with the imputed null values.
    """
    # Create a KNN Imputer
    knn_imputer = KNNImputer(n_neighbors=5)
    # Fit the KNN Imputer
    df_imputed = knn_imputer.fit_transform(df)
    # Return the imputed dataframe
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    return df_imputed

def smote_upsample(X_train, y_train):
    """
    Upsample the minority class using SMOTE.

    Parameters
    ----------
    X_train: numpy.ndarray
        The feature matrix of the training data.

    y_train: numpy.ndarray
        The target vector of the training data.

    Returns
    -------
    tuple of numpy.ndarray
        The upsampled feature matrix and target vector.
    """

    # Print class distribution before upsampling
    print("=============================")
    counter = Counter(y_train)
    for k,v in counter.items():
        per = 100*v/len(y_train)
        print(f"Class= {k}, n={v} ({per:.2f}%)")

    # Perform SMOTE upsampling
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    # Print class distribution after upsampling
    print("=============================")
    counter = Counter(y_train)
    for k,v in counter.items():
        per = 100*v/len(y_train)
        print(f"Class= {k}, n={v} ({per:.2f}%)")
    print("=============================")

    print("Upsampled data shape: ", X_train.shape, y_train.shape)

    return X_train, y_train

def preprocess(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Preprocess the whole dataset

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to preprocess.

    threshold: int 
        The maximum allowed null value count for a column.
    
    Returns
    -------
    df: pd.DataFrame
        The preprocessed dataframe.
    """
    # Rename columns
    df = rename_columns(df)
    # Drop null columns
    df = drop_null_cols(df, threshold)

    # Convert time column to datetime
    df = time_processing(df)

    # Convert minutes column to nearest multiple of 5
    df['minute'] = df['minute'].apply(convert_minutes)

    # Convert categorical columns to numeric
    df = ordinal_encoder(df, df.select_dtypes(include=['object']).columns)

    # Impute null values using KNN Imputer
    df = knn_impute(df)

    return df
