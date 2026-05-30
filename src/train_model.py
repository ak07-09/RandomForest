import os
import joblib

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.model_selection import (
    GridSearchCV
)

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

param_grid={

    "n_estimators":[
        100,
        200
    ],

    "max_depth":[
        10,
        20,
        None
    ],

    "min_samples_split":[
        2,
        5
    ],

    "min_samples_leaf":[
        1,
        2
    ],

    "max_features":[
        "sqrt",
        "log2"
    ],

    "bootstrap":[
        True,
        False
    ],

    "criterion":[
        "gini",
        "entropy"
    ]
}

rf=RandomForestClassifier(
    random_state=42
)

grid_search=GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=2
)

grid_search.fit(
    X_train,
    y_train
)

best_model=grid_search.best_estimator_

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/rf_classifier.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    encoder,
    "models/encoder.pkl"
)

joblib.dump(
    columns,
    "models/columns.pkl"
)

print(
    "Best Parameters:"
)

print(
    grid_search.best_params_
)

print(
    "Best CV Score:",
    grid_search.best_score_
)

print(
    "Train Accuracy:",
    best_model.score(
        X_train,
        y_train
    )
)

print(
    "Test Accuracy:",
    best_model.score(
        X_test,
        y_test
    )
)