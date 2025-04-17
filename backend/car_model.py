# backend/car_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def clean_data():

    df = pd.read_csv("data/cars.csv")
  
    df['horsepower'] = df['horsepower'].replace('?', np.nan)
    df['horsepower'] = df['horsepower'].astype(float)
    df.dropna(inplace=True)
    df.drop("car_name", axis=1, inplace=True)
    
    return df


def split_data():

    df = clean_data()

    X = df.drop('mpg', axis=1)
    y = df['mpg']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def model_pipeline():
    
    numeric_features = ['displacement', 'horsepower', 'weight', 'acceleration']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_features = ['cylinders', 'model', 'origin']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return model


def train_model():
    
    X_train, X_test, y_train, y_test = split_data()
    model = model_pipeline()
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model MSE: {mse:.4f}")
    print(f"Model R²: {r2:.4f}")

    joblib.dump(model, "model/car_mpg_model.pkl")
    
    feature_names = list(X_train.columns)
    return model, feature_names


def load_car_model():

    model_path = "model/car_mpg_model.pkl"
        
    model = joblib.load(model_path)
    
    feature_names = ['cylinders', 'displacement', 'horsepower', 'weight', 
                    'acceleration', 'model', 'origin']
    
    return model, feature_names

if __name__ == "__main__":
    #train_model()
    # model, feature_names = load_car_model()
    # input_data = pd.DataFrame([[4, 200.0, 150.0, 3000.0, 15.0, 76, 1]], 
    #                          columns=feature_names)
    
    #input_data = np.array([[4, 200.0, 150.0, 3000.0, 15.0, 76, 1]])
    # print(model.predict(input_data))
    pass