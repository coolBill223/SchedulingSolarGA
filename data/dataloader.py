import pandas as pd
import numpy as np
import os
import torch
import re
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import re

def clean_angle(value):
    """change Angles to the integer data we need"""
    if pd.isna(value):
        return np.nan  #keep nan for nan
    value = str(value).replace("°", "").strip()  # move the sign °
    
    if "/" in value:  # changing the angels with "/"
        try:
            values = list(map(float, value.split("/")))
            return sum(values) / len(values)  # get the average
        except:
            return np.nan  # read input failed, return nan
    
    try:
        return float(value)  # translate into integer
    except:
        return np.nan  # read input failed, return nan

def convert_time_to_minutes(value):
    """ change time to the numerical data we need """
    if pd.isna(value) or value in ["nan", "None", ""]:
        return None  #fill none if there is nan, none or no value

    if isinstance(value, pd.Timedelta): 
        return value.total_seconds() / 60
    elif isinstance(value, datetime.datetime) or isinstance(value, datetime.time):  
        return value.hour * 60 + value.minute
    elif isinstance(value, str):  # change the str to datetime
        value = value.strip().lower()

        # **if time is going to the format with hh: mm**
        match = re.match(r"(\d+):(\d+)(?::(\d+))?", value)
        if match:
            h, m, s = map(lambda x: int(x) if x else 0, match.groups())
            return h * 60 + m

        # **dealing with the time format such as 2h 15m**
        match = re.match(r"(\d+)\s*h\s*(\d*)\s*m?", value, re.IGNORECASE)
        if match:
            h = int(match.group(1))
            m = int(match.group(2)) if match.group(2) else 0
            return h * 60 + m

        # **change the 15mins to 15**
        match = re.match(r"(\d+)\s*mins?", value)
        if match:
            return int(match.group(1))  # just return the numerical time 

        # **dealing with the time format with integer**
        if value.isnumeric():
            return int(value)

    return None  # read input failed, return none

def convert_dhm_to_minutes_strict(value):
    """
    Converts values in 'D:H:M', 'D:H', or 'H' format into total minutes.
    - Handles Excel auto-formatted cells (datetime.datetime or time)
    - Treats 1:9:30 as 1 day, 9 hours, 30 minutes (NOT hours:minutes:seconds!)
    """
    import datetime

    if pd.isna(value) or value in ["", "nan", "None"]:
        return None

    try:
        # Case 1: datetime.datetime from Excel (e.g. 1900-01-02 09:30:00)
        if isinstance(value, datetime.datetime):
            base = datetime.datetime(1899, 12, 30)  # Excel zero-date origin
            delta = value - base
            return delta.total_seconds() / 60

        # Case 2: datetime.time → just hours, minutes, seconds
        elif isinstance(value, datetime.time):
            return value.hour * 60 + value.minute + value.second / 60

        # Case 3: string (D:H:M or D:H)
        parts = list(map(int, str(value).strip().split(":")))

        if len(parts) == 3:
            d, h, m = parts
        elif len(parts) == 2:
            d, h = parts
            m = 0
        elif len(parts) == 1:
            d = 0
            h = parts[0]
            m = 0
        else:
            return None

        return d * 24 * 60 + h * 60 + m
    except Exception as e:
        print(f"[WARN] Failed to convert value: {value} ({type(value)}) → {e}")
        return None






"""
    Load and preprocess the solar installation dataset from Excel.

    This function reads the dataset from an Excel file, performs the following steps:
    - Skips irrelevant rows and columns
    - Cleans column names
    - Parses time fields (e.g. drive time, total install time)
    - Converts angle values (e.g. tilt, azimuth)
    - Encodes categorical and boolean features(change into integers)
    - Selects numeric features and drops excluded columns(users can edit this part)
    - Standardizes input features (X) and normalizes target variable (y)
    - Converts both to PyTorch tensors

    Parameters:
    None

    Returns:
    --------
    X : torch.Tensor, shape (n_samples, n_features)
        Standardized input features used for modeling.
    
    y : torch.Tensor, shape (n_samples, 1)
        Normalized target variable (installation duration in minutes, sqrt-transformed).
    
    X_scaler : sklearn.preprocessing.StandardScaler
        Scaler used to standardize X (useful for inverse-transforming predictions).
    
    y_scaler : sklearn.preprocessing.StandardScaler
        Scaler used to normalize y (useful for inverse-transforming predictions).

    Example:
    --------
    >>> X, y, X_scaler, y_scaler = load_data()
    >>> print(X.shape, y.shape)
    torch.Size([277, 21]) torch.Size([277, 1])
    """
def load_data():
    """ read the excel data and return to the format we need"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "..", "data", "UPDATED Dataset - Predictive Tool Development for Residential Solar Installation Duration - REV1-3.xlsx")  

    if not os.path.exists(file_path):
        print(f"File didn't find: {file_path}")
        return None, None, None, None 

    df = pd.read_excel(file_path, engine="openpyxl")
    # delete the first row and the first column
    df = pd.read_excel(file_path, engine="openpyxl", skiprows=2)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.columns = df.columns.str.strip().str.replace("\n", " ")

    if df.empty:
        print("The file is empty, please check the file！")
        return None, None, None, None  

    print(f"Data load successfully! Shape: {df.shape}")

    
    # target 
    target = "Total Direct Time for Project for Hourly Employees (Including Drive Time)"
    
    if target not in df.columns:
        #print(f"Target {target} doesn't exist！")
        return None, None, None, None  

    #print(f"Target {target}'s unique values: {df[target].unique()}")

    if pd.api.types.is_timedelta64_dtype(df[target]):
        #print(f"`{target}` is timedelta64 type, convert to minutes")
        df[target] = df[target].dt.total_seconds() / 60
        
    if df[target].apply(lambda x: isinstance(x, (datetime.datetime, datetime.time, str, pd.Timedelta))).any():
        #print(f"`{target}` includes datetime, time, str or timedelta, convert to minutes")
        df[target] = df[target].apply(convert_dhm_to_minutes_strict)

    # **make sure target is numeric to train**
    df[target] = pd.to_numeric(df[target], errors="coerce").astype("float64")
    df[target] = np.sqrt(df[target])  # apply  log(y+1) transforme to avoid negative values

    if df[target].isnull().sum() > 0:
        df[target] = df[target].fillna(df[target].mean())

    #print(f"Target{target} is null: {df[target].isnull().sum()}")
    #print(df[target].dtype)
    #print(df[target].head(10))

    # change Drive Time
    #print(f"Original Drive Time first 10 rows:\n{df['Drive Time'].head(10)}")
    #print(f"Original Drive Time unique values: {df['Drive Time'].unique()[:20]}")

    if "Drive Time" in df.columns:

        # If it is timedelta64，change it into minutes
        if pd.api.types.is_timedelta64_dtype(df["Drive Time"]):
            #print("`Drive Time` is timedelta64 type, need to convert to minutes")
            df["Drive Time"] = df["Drive Time"].dt.total_seconds() / 60
        else:
            df["Drive Time"] = df["Drive Time"].astype(str).str.strip()  # delete the space
            df["Drive Time"] = df["Drive Time"].replace(["", "nan", "None"], pd.NA)  # deal with the string
            df["Drive Time"] = df["Drive Time"].apply(convert_time_to_minutes)

        # make sure Drive Time is in minutes
        #print(f"Drive Time first 10 rows after conversion:\n{df['Drive Time'].head(10)}")

    # **fill nan with 0**
    df["Drive Time"] = df["Drive Time"].fillna(0)

    if "Tilt" in df.columns:
        df["Tilt"] = df["Tilt"].apply(clean_angle)

    if "Azimuth" in df.columns:
        df["Azimuth"] = df["Azimuth"].apply(clean_angle)
    
    # change yes/no to 1/0
    boolean_cols = [col for col in df.columns if df[col].dropna().astype(str).apply(lambda x: x.lower() in ["yes", "no"]).all()]
    for col in boolean_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0})  

    # use Label Encoding
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in boolean_cols + [target]]  # exclude boolean and target

    for col in categorical_cols:
        df[col] = df[col].astype(str)  # make sure the data is string
        df[col] = df[col].factorize()[0] + 1  # start given the label start from 1

    # change to numeric
    df[target] = pd.to_numeric(df[target], errors="coerce")

    #print(f"Target {target} is null: {df[target].isnull().sum()}")
    #print(df[target].dtype)
    #print(df[target].head(10))

    df = df.dropna(subset=[target])  # delete the row that y is null


    #print(f"Data column: {df.columns.tolist()}")

    # **check nan/null**
    #print(f"Data is null:\n{df.isnull().sum()}")

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #CHANGE THE EXCLUDE COLUMNS HERE!!!!!!
    exclude_columns = [
        "Project ID", "Notes", "Total # of Days on Site",
        "Estimated # of Salaried Employees on Site",
        "Estimated Salary Hours",
        "Estimated Total Direct Time",
        "Estimated Total # of People on Site"
    ]


    # reget all the features
    features = [col for col in df.columns if col != target and col not in exclude_columns]
    missing_features = [col for col in features if col not in df.columns]

    if missing_features:
        print(f"Columns with missing features: {missing_features}")
        return None, None, None, None  

    # **check the feature head is null**
    #print(f"Selected features:\n{df[features].head()}")

    # **make sure all the data is numeric**
    for col in features:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            #print(f"`{col}` is timedelta64 type，translate into minutes")
            df[col] = df[col].dt.total_seconds() / 60

    pd.set_option("display.max_columns", None)
    print(f"The first 10 rows of the data:\n{df[features].head(10)}")
    
# **Standardization**
    X_scaler = StandardScaler()
    df[features] = df[features].fillna(0)  # fill nan with 0
    df[features] = X_scaler.fit_transform(df[features])


    # ** normalization y**
    y_scaler = StandardScaler()
    y = df[[target]].values
    y = pd.DataFrame(y).fillna(0).values  # ✅ fill NaN with 0
    y = y_scaler.fit_transform(y)

    # **turn into torch**
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    print(f"Selected features: {features}")
    print(f"Data after standardization and normalization:\n{df[features].head(10)}")
    print(f"Data loaded successfully! X shape: {X.shape}, y shape: {y.shape}")

    # Optional: show a few original y values (after inverse transform and square)
    y_sqrt = y_scaler.inverse_transform(y.numpy())  # shape: (N, 1)
    y_real_minutes = y_sqrt ** 2                    # undo sqrt()

    # Convert to pandas for pretty display
    y_real_minutes = pd.DataFrame(y_real_minutes, columns=["Install Time (min)"])
    print("\nFirst 10 rows of real target values (in minutes):")
    print(y_real_minutes.head(10))
    return X, y, X_scaler, y_scaler  

if __name__ == "__main__":
    X, y, X_scaler, y_scaler = load_data()
