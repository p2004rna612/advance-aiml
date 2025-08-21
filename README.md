import numpy as np

# Sample data
# Reshape the data for scikit-learn
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction
prediction = model.predict([[6]])

print(f"Prediction for X=6: {prediction[0]:.2f}")
