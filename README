Stock Price Prediction Using LSTM
This project is a stock price prediction model built using Long Short-Term Memory (LSTM) networks. The goal of the project is to predict the future stock prices of a company based on its historical stock prices. We utilize time series data and apply LSTM, a type of recurrent neural network (RNN), to model the sequential nature of the stock price data.

Table of Contents
Project Overview
Technologies Used
Dataset
Model Architecture
Data Preprocessing
Model Training
Model Evaluation
Usage
Results and Visualizations
Future Work
License
Project Overview
Stock market prediction is a challenging problem due to the volatile and non-linear nature of financial markets. This project focuses on using machine learning techniques to predict stock prices using historical data. Specifically, we implement a LSTM-based model, which is well-suited for time series forecasting tasks due to its ability to capture long-term dependencies in sequential data.

The stock price data is preprocessed, normalized, and used to train an LSTM model to predict the closing price of a stock for the next day. The model is evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) metrics, and its predictions are compared to the actual stock prices.

Technologies Used
Python: The primary programming language for implementing the machine learning model and preprocessing the data.
Keras: A high-level neural networks API, used for building and training the LSTM model.
TensorFlow: Backend framework for Keras, used for training the model.
Pandas: Data manipulation library used for loading and preprocessing the stock price data.
NumPy: Used for numerical computations and handling arrays.
Matplotlib: A library for creating static, animated, and interactive visualizations in Python.
Scikit-learn: A library for machine learning, used for scaling and evaluation metrics (e.g., MSE, RMSE).
Jupyter Notebook: Used for running the code and visualizing the results interactively.
Dataset
The dataset used in this project consists of historical stock prices of a company. The dataset includes the following columns:

Date: The date of the stock data entry.
Open: The opening price of the stock on that day.
High: The highest price reached during the day.
Low: The lowest price reached during the day.
Close: The closing price of the stock on that day.
Volume: The number of shares traded on that day.
For the purpose of this project, we use only the Close price to predict the future stock price. The dataset is available in CSV format and can be loaded into a Pandas DataFrame for further processing.

Model Architecture
We use a Long Short-Term Memory (LSTM) model for stock price prediction. LSTM is a type of recurrent neural network (RNN) that is capable of learning long-term dependencies in sequential data, making it well-suited for time series forecasting tasks.

Model Structure:
Input Layer: Takes in the historical stock prices as input.
LSTM Layers: One or more LSTM layers are used to capture temporal patterns in the stock prices.
Dense Layer: The output layer is a dense layer with a single neuron that predicts the stock price for the next day.
Activation Functions: The LSTM layers use the tanh activation function, and the output layer uses a linear activation function to predict a continuous value (stock price).
Data Preprocessing
Steps:
Loading Data: The stock price data is loaded into a Pandas DataFrame from a CSV file.
Data Cleaning: Missing or erroneous data points are handled, and only the "Close" price is retained for prediction.
Feature Scaling: Since stock prices can have a large range, we normalize the data using MinMaxScaler from Scikit-learn. This scales the data to a range between 0 and 1, which helps the model learn more effectively.
Train-Test Split: The data is split into training and testing sets. We use the last 100 days of data from the training set to predict future stock prices.
Windowing: We create a sliding window of 100 consecutive days to predict the stock price on the next day. This results in a sequence of input-output pairs for model training.
python
Copy code
# Scaling the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Preparing the training data
x_train, y_train = [], []
for i in range(100, len(scaled_data)):
    x_train.append(scaled_data[i-100:i])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
Model Training
The LSTM model is trained on the preprocessed training data. We use the following parameters for the LSTM model:

Batch Size: 32
Epochs: 20
Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
The model is trained using the training data, and its performance is evaluated on the test data.

python
Copy code
# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(x_train, y_train, epochs=20, batch_size=32)
Model Evaluation
After training the model, we evaluate its performance on the test dataset. The evaluation is done by comparing the predicted stock prices to the actual stock prices. The key metrics used for evaluation are:

Mean Squared Error (MSE): Measures the average of the squares of the errors (difference between predicted and actual values).
Root Mean Squared Error (RMSE): The square root of MSE, providing a more interpretable measure of prediction error.
R-squared (R²): Measures how well the model's predictions match the actual values, with a value closer to 1 indicating a better fit.
python
Copy code
# Model prediction and evaluation
y_pred = model.predict(x_test)

# Inverse scaling
y_pred = scaler.inverse_transform(y_pred)
y_actual = scaler.inverse_transform(y_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_pred)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the model: You can run the code in a Jupyter Notebook or as a Python script to train the model and evaluate its performance on a test dataset.

Input Data: Place your stock price dataset (in CSV format) in the appropriate folder, ensuring it contains at least the "Date" and "Close" columns.

Results and Visualizations
Once the model has been trained, the results are evaluated and visualized. Below is an example of how the actual and predicted stock prices can be plotted:

python
Copy code
import matplotlib.pyplot as plt

plt.plot(y_actual, label='Actual Stock Price')
plt.plot(y_pred, label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
The model's predictions are plotted against the actual stock prices, allowing you to visually assess its performance.

Future Work
Hyperparameter Tuning: Further optimization of the model's hyperparameters (e.g., number of LSTM units, batch size, learning rate) could improve its performance.
Additional Features: Incorporating other features, such as trading volume, opening price, and technical indicators (e.g., moving averages), may improve the accuracy of predictions.
Model Improvements: Trying other time series forecasting models, such as GRU (Gated Recurrent Units) or Transformer models, could lead to better performance.
