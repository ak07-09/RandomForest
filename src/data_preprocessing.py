import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)


DATA_PATH="data/laptop_price.csv"


def extract_cpu_brand(text):

    text=text.split()

    if text[0]=="Intel":

        return " ".join(text[:3])

    elif text[0]=="AMD":

        return "AMD Processor"

    else:

        return text[0]


def extract_gpu_brand(text):

    return text.split()[0]


def process_memory(value):

    value=str(value)

    hdd=0
    ssd=0

    if "HDD" in value:

        parts=value.split("+")

        for item in parts:

            if "HDD" in item:

                number=item.strip().split()[0]

                if "TB" in number:

                    hdd+=int(
                        float(
                            number.replace(
                                "TB",
                                ""
                            )
                        )*1000
                    )

                else:

                    hdd+=int(
                        number.replace(
                            "GB",
                            ""
                        )
                    )

    if "SSD" in value:

        parts=value.split("+")

        for item in parts:

            if "SSD" in item:

                number=item.strip().split()[0]

                if "TB" in number:

                    ssd+=int(
                        float(
                            number.replace(
                                "TB",
                                ""
                            )
                        )*1000
                    )

                else:

                    ssd+=int(
                        number.replace(
                            "GB",
                            ""
                        )
                    )

    return hdd,ssd


def load_and_preprocess_data():

    df=pd.read_csv(
        DATA_PATH,
        encoding="latin1"
    )

    df=df.drop_duplicates()

    if "laptop_ID" in df.columns:

        df=df.drop(
            "laptop_ID",
            axis=1
        )

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

    df["Cpu brand"]=df["Cpu"].apply(
        extract_cpu_brand
    )

    df["Gpu brand"]=df["Gpu"].apply(
        extract_gpu_brand
    )

    df["Touchscreen"]=df[
        "ScreenResolution"
    ].apply(
        lambda x:1 if "Touchscreen" in x else 0
    )

    df["IPS"]=df[
        "ScreenResolution"
    ].apply(
        lambda x:1 if "IPS" in x else 0
    )

    memory_features=df["Memory"].apply(
        process_memory
    )

    df["HDD"]=memory_features.apply(
        lambda x:x[0]
    )

    df["SSD"]=memory_features.apply(
        lambda x:x[1]
    )

    df=df.drop(
        [
            "Cpu",
            "Gpu",
            "Memory",
            "ScreenResolution",
            "Product"
        ],
        axis=1
    )

    X=df.drop(
        "Company",
        axis=1
    )

    y=df["Company"]

    encoder=LabelEncoder()

    y=encoder.fit_transform(y)

    X=pd.get_dummies(X)

    feature_columns=X.columns

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
        feature_columns
    )