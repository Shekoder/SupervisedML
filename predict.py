import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset from the provided URL
url = "http://bit.ly/w-data"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Features and target variable
X = data[['Hours']]  # Feature
y = data['Scores']   # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Prepare the input for prediction with the correct feature name
hours_studied = pd.DataFrame([[9.25]], columns=['Hours'])
predicted_score = model.predict(hours_studied)

print(f"Predicted Score for studying 9.25 hours/day: {predicted_score[0]:.2f}")

# Optional: Evaluate the model and plot the regression line
y_pred = model.predict(X_test)
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Hours vs Percentage Score')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.legend()
plt.show()

