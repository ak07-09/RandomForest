import pandas as pd
import joblib

model=joblib.load(
    "models/rf_classifier.pkl"
)

scaler=joblib.load(
    "models/scaler.pkl"
)

encoder=joblib.load(
    "models/encoder.pkl"
)

columns=joblib.load(
    "models/columns.pkl"
)

ram=int(
    input("Enter RAM:")
)

weight=float(
    input("Enter Weight:")
)

inches=float(
    input("Enter Screen Size:")
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

prediction=model.predict(sample)

result=encoder.inverse_transform(
    prediction
)[0]

print(
    "Predicted Company:",
    result
)