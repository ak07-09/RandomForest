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

print(df.head())

print(df.info())

print(df.describe())

print(
    "Missing Values:"
)

print(
    df.isnull().sum()
)

plt.figure(figsize=(10,5))

sns.countplot(
    x="Company",
    data=df
)

plt.xticks(rotation=90)

plt.title(
    "Laptop Company Distribution"
)

plt.show()

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Ram",
    y="Price_euros",
    hue="Company",
    data=df
)

plt.title(
    "RAM vs Price"
)

plt.show()

plt.figure(figsize=(10,6))

sns.heatmap(
    df.select_dtypes(
        include="number"
    ).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["Price_euros"]
)

plt.title(
    "Price Outlier Detection"
)

plt.show()