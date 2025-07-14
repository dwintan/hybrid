
import os
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from dateutil.relativedelta import relativedelta

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx'}

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def split_data(stock, lookback, test_percentage):
    data_raw = stock
    data = []
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])
    data = np.array(data)
    test_set_size = int(np.round(test_percentage * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]
    return [x_train, y_train, x_test, y_test, train_set_size]

@app.route('/prediksi', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        steps = int(request.form['horizon'])

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_excel(filepath)
            df = df.dropna()
            df.columns = ['y']

            model_arima = SARIMAX(df['y'], order=(2, 2, 1))
            model_arima_fit = model_arima.fit(disp=False)
            preds = model_arima_fit.predict(1, len(df), typ='levels')
            preds.index -= 1

            arima_result = pd.DataFrame({'raw': df['y'], 'predicted': preds})
            arima_result['residuals'] = arima_result['raw'] - arima_result['predicted']
            residuals = arima_result['residuals'].values.astype(float)

            scaler = MinMaxScaler(feature_range=(0, 1))
            residuals_scaled = scaler.fit_transform(residuals.reshape(-1, 1))

            lookback = 7
            x_train, y_train, x_test, y_test, train_set_size = split_data(residuals_scaled, lookback, 0.2)

            model = Sequential()
            model.add(LSTM(128, input_shape=(lookback - 1, 1)))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation="sigmoid"))
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

            arima_forecast = model_arima_fit.get_forecast(steps=steps)
            arima_predicted_mean = arima_forecast.predicted_mean.values

            lstm_forecast_scaled = []
            input_seq = np.copy(x_test[-1:])

            for _ in range(steps):
                pred = model.predict(input_seq, verbose=0)[0, 0]
                lstm_forecast_scaled.append(pred)
                input_seq = np.roll(input_seq, -1)
                input_seq[0, -1, 0] = pred

            lstm_forecast_actual = scaler.inverse_transform(np.array(lstm_forecast_scaled).reshape(-1, 1)).flatten()
            final_forecast = arima_predicted_mean + lstm_forecast_actual

            start_date = datetime(2024, 7, 1)
            bulan_prediksi = [(start_date + relativedelta(months=i)).strftime('%B %Y') for i in range(steps)]
            prediction = list(zip(bulan_prediksi, final_forecast.tolist()))

    return render_template('index.html', prediction=prediction)

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

if __name__ == '__main__':
    app.run(debug=True)
