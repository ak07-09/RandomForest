import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)


DATA_PATH="data/laptop_price.csv"


def load_and_preprocess_data():

    df=pd.read_csv(
        DATA_PATH,
        encoding="latin1"
    )

    if "Unnamed: 0" in df.columns:

        df=df.drop(
            "Unnamed: 0",
            axis=1
        )

    df=df.drop_duplicates()

    df["Ram"]=df["Ram"].str.replace(
        "GB",
        "",
        regex=False
    ).astype(int)

    df["Weight"]=df["Weight"].str.replace(
        "kg",
        "",
        regex=False
    ).astype(float)

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

    return(
        df,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoder,
        X.columns
    )