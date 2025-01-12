import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# load data
df = pd.read_csv('prices.csv')

#  convert date format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# handle missing values
df = df.dropna()

# data sorted in ascending date
df = df.sort_values(by='date')

# select required features
df = df[['date', 'symbol', 'open', 'close', 'low', 'high', 'volume']]

# creat moving average features
df['MA10'] = df['close'].rolling(window=10).mean()
df['MA20'] = df['close'].rolling(window=20).mean()

# calculate daily price change rate
df['price_change'] = df['close'].pct_change()

# add 7-day moving average of trading volune
df['volume_MA7'] = df['volume'].rolling(window=7).mean()

# handle NaN
df = df.dropna()

# Set stock movement as the target variable (1 for up, 0 for down)
df['direction'] = (df['close'].shift(-1) > df['close']).astype(int)

# Feature columns and target column
features = ['open', 'low', 'high', 'volume', 'MA10', 'MA20', 'price_change', 'volume_MA7']
target = 'direction'

# Split data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)

# Define the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)

# Use GridSearchCV for parameter tuning
param_grid = {
    'C': [0.1, 1, 10],  # Regularization Parameter
    'solver': ['liblinear', 'saga'],  # select algorithm
    'penalty': ['l2'],  # L2 regularization
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the model
grid_search.fit(train_data[features], train_data[target])

print("Best Parameters: ", grid_search.best_params_)

# use best parameter train ultimate model
best_model = grid_search.best_estimator_

test_data['predicted_direction'] = best_model.predict(test_data[features])

# Make predictions on the test set
test_data['actual_direction'] = (test_data['close'].shift(-1) > test_data['close']).astype(int)

# evaluate model
accuracy = accuracy_score(test_data[target], test_data['predicted_direction'])
f1 = f1_score(test_data[target], test_data['predicted_direction'])

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')

# Plot the comparison of predictions and actual values
plt.figure(figsize=(10, 6))
plt.plot(test_data['date'], test_data[target], label='Actual Direction', color='blue')
plt.plot(test_data['date'], test_data['predicted_direction'], label='Predicted Direction', color='red', linestyle='--')
plt.title('Actual vs Predicted Direction (Up = 1, Down = 0)')
plt.xlabel('Date')
plt.ylabel('Direction')
plt.legend()
# Save the comparison plot
plt.savefig('actual_vs_predicted_direction.png')
plt.close()

# Confusion matrix
cm = confusion_matrix(test_data['actual_direction'], test_data['predicted_direction'])
cm_display = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
cm_display.set_xlabel('Predicted')
cm_display.set_ylabel('Actual')
plt.title('Confusion Matrix')
# Save the confusion matrix plot
plt.savefig('confusion_matrix.png')
plt.close()

# Classification report
print(classification_report(test_data['actual_direction'], test_data['predicted_direction']))

# Calculate prediction errors: error in direction prediction
test_data['error'] = test_data['actual_direction'] - test_data['predicted_direction']

# Plot the residual plot
plt.figure(figsize=(10, 6))
plt.plot(test_data['date'], test_data['error'], label='Prediction Error', color='green')
plt.axhline(0, color='black', linestyle='--')  # Reference Line
plt.title('Prediction Error (Residuals)')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
# Save the residual plot
plt.savefig('prediction_error_residuals.png')
plt.close()
