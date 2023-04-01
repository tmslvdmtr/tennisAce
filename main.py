import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read the CSV data and assign DataFrame and create a LR 
readCSV = pd.read_csv('tennis_stats.csv')
data = pd.DataFrame(readCSV)
lm = LinearRegression()

# Linear Regression Model
# Feature - Break points converted
# Independent Value - Wins
X = data[['BreakPointsConverted']]
y = data['Wins']

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = lm.fit(X_train, y_train)
y_predict = lm.predict(X_test)

print("Train score of first LR:")
print(lm.score(X_train, y_train))
print("Test score of first LR:")
print(lm.score(X_test, y_test))

# Multiple(2) Linear Regression 
# Features - First Serve and Aces
# Independent Value - Wins
features = ['FirstServe', 'Aces']
X = data[features]
y = data['Wins']

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = lm.fit(X_train, y_train)
y_predict = lm.predict(X_test)

print("Train score of second LR:")
print(lm.score(X_train, y_train))
print("Test score of second LR:")
print(lm.score(X_test, y_test))

# Multiple Linear Regression 
# Features - Data set
# Independent Value - Wins
x = data[['Aces', 'DoubleFaults', 'FirstServe', 'FirstServePointsWon', 'SecondServePointsWon', 'BreakPointsFaced', 'BreakPointsSaved', 'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalServicePointsWon']]
y = data[['Wins']]

x_train, x_test, y_train, y_test = train_test_split(x, y)
model = lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)

print("Train score of last MLR:")
print(lm.score(x_train, y_train))
print("Test score of last MLR:")
print(lm.score(x_test, y_test))

plt.scatter(y_test, y_predict)
plt.title('Predicted Wins vs. Actual Wins')
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.plot(x_test, y_predict, alpha=0.4)
plt.show()