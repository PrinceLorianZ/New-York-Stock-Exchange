# Stock Price Prediction Project README

## Project Overview
— This project employs Logistic Regression model to make an estimate of the direction of the stock price going up or down on previous stock (e.g., open price, close price, high, low, and volume). The ultimate aim is to use machine learning algorithms to categorize and make predictions on the stock market movements so users can get to know the stock market trend in the future.

## Project Structure
- **prices.csv**: CSV file of historical stock data including date, symbol, open price, close price, high, low, and volume.  
- **stock_price_prediction.py**: The main project code that load data, parse features, learn the model, test the model and outputs visual output.  
- **actual_vs_predicted_direction.png**: Actual stock direction vs. Predicted stock direction Comparison.  
- **confusion_matrix.png**: Confusion matrix of model prediction.  
- **prediction_error_residuals.png**: Residual graph of model’s prediction errors.  
- **README.md**: This README file.

## Project Dependencies
Python libraries and versions needed to build this project:

- `pandas`: For processing and data.
- `scikit-learn`: For ML models creation and testing.
- `matplotlib`: For plotting graphs.
- `seaborn`: For visualizing confusion matrix.
- `numpy`: For scientific computation.

Install the dependencies using this command:
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

## Dataset Description
The `prices.csv` file stores stock key market information. The fields are as follows:

- **date**: Date (format: YYYY-MM-DD)
- **symbol**: Stock identifier (stock code)
- **open**: Open price
- **close**: Close price
- **low**: Lowest price
- **high**: Highest price
- **volume**: Trading volume

Each row includes market data from the previous trading day: price, closing price, highest and lowest price, volume.

## Project Flow

### Data Preprocessing
1. **Date Conversion**: Update date column to datetime type.  
2. **Missing Values Handling**: Removing missing rows.  
3. **Feature Engineering**:
   - Determine the 10-day and 20-day MA10,MA20.  
   - Determine the daily price change rate (price_change).  
   - Estimate 7-day volume moving average (volume_MA7).

### Feature and Target Definition
- **Feature Columns**:  
  `open`, `low`, `high`, `volume`, `MA10`, `MA20`, `price_change`, `volume_MA7`
- **Target Column**:  
  `direction`: Shows whether the stock price is +1 or -1 next day.

### Model Training
- Classification prediction is made by Logistic Regression.  
- **GridSearchCV**: Tuning hyperparameters for the model parameters (Regularization parameter C, solver algorithm, and penalty algorithm) using GridSearchCV.

### Model Evaluation and Visualization
- **Accuracy**: Estimate model prediction accuracy.  
- **F1 Score**: Determine model accuracy and memory.  
- **Confusion Matrix**: Observe the classification output (actual vs. expected classifications).  
- **Residual Plot (Prediction Error)**: Show prediction error of the model and bias.

### Saving Results
- **Plot of Comparison**: Compare the real-time and predicted stock direction.  
- **Matrix of Confusion plot**: plot the matrix of confusion.  
- **Residual Plot**: Display the prediction error.

## How to Use
1. Save the dataset `prices.csv` in the same folder as `stock_price_prediction.py` and open it.
2. Run `stock_price_prediction.py` to begin model training and test.
```bash
python stock_price_prediction.py
```

You should see these outputs after you run the script:

- Best Parameters (GrooveSearchCV best hyperparameters).
- Accuracy and F1 Score.
- Classification Report: Performance report of the model.
- Images will be made and saved:  
  - **actual_vs_predicted_direction.png**: Comparison of actual vs predicted stock direction.  
  - **confusion_matrix.png**: The confusion matrix plot.  
  - **prediction_error_residuals.png**: The residual plot.

## Project Output

### Actual vs Predicted Direction Plot
This plot shows the difference between the stock direction and the predicted stock direction of the test dataset and gives a visual representation of how accurately the model predicts.

### Confusion Matrix
This confusion matrix describes in a more detailed manner what the model was predicting: True positives (actually forecasted upwards movements), True negatives (actually forecasted downwards movements), False positives and False negatives.

### Residual Plot
In the residual plot we can see prediction error (the deviation between predicted and real direction) for each prediction, which provides information about the model bias.

## Contributions and Future Improvements
- **Enhancement to Model**: Currently, the model being used in the project is a very simple logistic regression model but other complex models could be implemented like Support Vector Machines (SVM) or Deep Learning model to gain better prediction.  
- **Extensions of Features**: There are also technical indicators (RSI, MACD), sentiment, macroeconomic indicators that could be added to increase model precision.

## License
This is a project licensed under the MIT License.

