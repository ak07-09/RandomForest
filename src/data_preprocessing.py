import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)


def load_data(path):

    df=pd.read_csv(path)

    return df


def clean_data(df):

    if "Unnamed: 0" in df.columns:
        df=df.drop(
            "Unnamed: 0",
            axis=1
        )

    df=df.drop_duplicates()

    return df


def prepare_classifier_data(df):

    X=df.drop(
        "Company",
        axis=1
    )

    y=df["Company"]

    encoder=LabelEncoder()

    y=encoder.fit_transform(y)

    X=pd.get_dummies(X)

    scaler=StandardScaler()

    X=scaler.fit_transform(X)

    X_train,X_test,y_train,y_test=train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoder
    )


def prepare_regressor_data(df):

    X=df.drop(
        "Price",
        axis=1
    )

    y=df["Price"]

    X=pd.get_dummies(X)

    scaler=StandardScaler()

    X=scaler.fit_transform(X)

    X_train,X_test,y_train,y_test=train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )