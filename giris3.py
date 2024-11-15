import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
mileage = np.array([5000, 6000, 8000, 10000, 12000, 15000, 18000, 20000, 22000, 25000])
price = np.array([25000, 24000, 22000, 20000, 18000, 16000, 15000, 14000, 13000, 12000])
# Reshape the data, reshape(-1, 1) means that, --> you are asking numpy to reshape
# your array with 1 column and as many rows as necessary to accommodate the data
mileage = mileage.reshape(-1, 1)
price = price.reshape(-1, 1)
# Split the data into training and testing sets, 0.70 for traning and 0.3 for test
# random_state it ensures that the same randomization is used each time you run the code,
X_train, X_test, y_train, y_test = train_test_split(mileage, price, test_size=0.3, random_state=42)
print("x_train is : ", X_train)
print("x_test is : ", X_test)
# Create a linear regression model
model = LinearRegression()
# Train the model on the training data
model.fit(X_train, y_train)
# Make predictions on the testing data
predictions = model.predict(X_test)
# Calculate the coefficient of determination (R^2) to evaluate the model
r_squared = model.score(X_test, y_test)
print("Coefficient of determination (R^2):", r_squared)
plt.plot(mileage, price)
plt.show() 
