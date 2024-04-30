from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Load dataset
df = pd.read_csv('../data/ar_properties.csv')
df['created_on']= pd.to_datetime(df['created_on'])
df['start_date']= pd.to_datetime(df['start_date'])
df1 = df[df.currency == 'USD']
df2 = df1[(df1['price_period'] == "Mensual") & (df1['property_type'] == "Casa")].drop(['id', 'end_date', 'l6'], axis=1)
df2.dropna(subset=["rooms","surface_total"], inplace=True)

data = df2

numerical_features = []
categorical_features = []
datetime_features = []

# Classify features based on their data type
for column in data.columns:
    dtype = data[column].dtype
    if dtype == 'object' or dtype == 'string':
        categorical_features.append(column)
    elif dtype == 'datetime64[ns]':
        datetime_features.append(column)
    elif column != 'price':  # Assuming 'price' should be excluded from features
        numerical_features.append(column)

# Pipeline for numeric features: imputation + scaling
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical features: imputation + one-hot encoding
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer to apply the appropriate transformations
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Complete pipeline including preprocessing and a regressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X = data.drop('price', axis=1)
y = np.log(data['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)



from skl2onnx.common.data_types import FloatTensorType, StringTensorType

initial_types = []
for name, dtype in zip(X_train.columns, X_train.dtypes):
    if "float" in dtype.name or "int" in dtype.name:
        initial_types.append((name, FloatTensorType([None, 1])))
    else:
        initial_types.append((name, StringTensorType([None, 1])))

onnx_model = convert_sklearn(pipeline, initial_types=initial_types)

with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

