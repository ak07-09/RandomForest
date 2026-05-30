import pandas as pd
import streamlit as st
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import (
    load_and_preprocess_data
)

(
    df,
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    encoder,
    columns
)=load_and_preprocess_data()

model=joblib.load(
    "models/rf_classifier.pkl"
)

st.set_page_config(
    page_title="Random Forest Classifier",
    layout="wide"
)

st.title(
    "Laptop Company Prediction"
)

ram=st.slider(
    "RAM",
    2,
    64,
    8
)

weight=st.slider(
    "Weight",
    1.0,
    5.0,
    2.0
)

inches=st.slider(
    "Screen Size",
    10.0,
    20.0,
    15.6
)

sample=pd.DataFrame({

    "TypeName":["Notebook"],

    "Inches":[inches],

    "Ram":[ram],

    "Weight":[weight],

    "Touchscreen":[0],

    "Ips":[1],

    "Ppi":[140],

    "Cpu brand":[
        "Intel Core i5"
    ],

    "HDD":[1000],

    "SSD":[256],

    "Gpu brand":[
        "Intel"
    ],

    "os":[
        "Windows"
    ],

    "Price":[50000]
})

sample=pd.get_dummies(sample)

sample=sample.reindex(
    columns=columns,
    fill_value=0
)

sample=scaler.transform(sample)

if st.button(
    "Predict Company"
):

    prediction=model.predict(sample)

    result=encoder.inverse_transform(
        prediction
    )[0]

    st.success(
        f"Predicted Company: {result}"
    )

st.subheader(
    "Company Distribution"
)

fig,ax=plt.subplots()

sns.countplot(
    x="Company",
    data=df,
    ax=ax
)

plt.xticks(rotation=90)

st.pyplot(fig)

st.subheader(
    "RAM vs Price"
)

fig,ax=plt.subplots()

sns.scatterplot(
    x="Ram",
    y="Price",
    hue="Company",
    data=df,
    ax=ax
)

st.pyplot(fig)

st.subheader(
    "Correlation Heatmap"
)

fig,ax=plt.subplots(
    figsize=(10,6)
)

sns.heatmap(
    df.select_dtypes(
        include="number"
    ).corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)