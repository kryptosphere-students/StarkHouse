import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()

model.fit(X_train, y_train)


initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

