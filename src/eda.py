import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(
    "data/laptop_data.csv"
)

print(df.head())

plt.figure(figsize=(8,4))

sns.countplot(
    x="Company",
    data=df
)

plt.xticks(rotation=90)

plt.title(
    "Laptop Company Distribution"
)

plt.show()

plt.figure(figsize=(6,4))

sns.histplot(
    df["Price"],
    kde=True
)

plt.title(
    "Laptop Price Distribution"
)

plt.show()

plt.figure(figsize=(6,4))

sns.scatterplot(
    x="Ram",
    y="Price",
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