import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

class LSTM:
    def __init__(self, n_in, n_out, n_units, l2_lambda=0.001, grad_clip=1.0):
        self.n_in = n_in
        self.n_out = n_out
        self.n_units = n_units
        self.l2_lambda = l2_lambda
        self.grad_clip = grad_clip

        self.Wf = np.random.randn(n_units, n_in + n_units) * np.sqrt(2 / (n_in + n_units))
        self.bf = np.zeros((n_units, 1))
        self.Wi = np.random.randn(n_units, n_in + n_units) * np.sqrt(2 / (n_in + n_units))
        self.bi = np.zeros((n_units, 1))
        self.Wc = np.random.randn(n_units, n_in + n_units) * np.sqrt(2 / (n_in + n_units))
        self.bc = np.zeros((n_units, 1))
        self.Wo = np.random.randn(n_units, n_in + n_units) * np.sqrt(2 / (n_in + n_units))
        self.bo = np.zeros((n_units, 1))
        self.Wy = np.random.randn(n_out, n_units) * np.sqrt(2 / n_units)
        self.by = np.zeros((n_out, 1))

        self.dropout_rate = 0.2

    def forward(self, x, prev_h, prev_c):
        concat = np.concatenate((prev_h, x), axis=0)

        ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        cct = np.tanh(np.dot(self.Wc, concat) + self.bc)
        c_next = ft * prev_c + it * cct
        ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        h_next = ot * np.tanh(c_next)

        if self.training:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=h_next.shape)
            h_next *= dropout_mask

        y = np.dot(self.Wy, h_next) + self.by
        return y, h_next, c_next, concat, cct, it, ft, ot

    def backward(self, x, y, y_pred, h_prev, c_prev, h_next, c_next, concat, cct, it, ft, ot, dh_next, dc_next):
        dy = y_pred - y
        dh_next += np.dot(self.Wy.T, dy)

        dWy = np.dot(dy, h_next.T) + self.l2_lambda * self.Wy
        dby = dy
        dht = np.dot(self.Wy.T, dy)

        dot = dht * np.tanh(c_next)
        dot = self.sigmoid_derivative(ot) * dot
        dWo = np.dot(dot, concat.T) + self.l2_lambda * self.Wo
        dbo = dot

        dc_next += dh_next * ot * self.tanh_derivative(np.tanh(c_next)) + dc_next
        dcct = dc_next * it
        dcct = self.tanh_derivative(cct) * dcct
        dWc = np.dot(dcct, concat.T) + self.l2_lambda * self.Wc
        dbc = dcct

        dit = dc_next * cct
        dit = self.sigmoid_derivative(it) * dit
        dWi = np.dot(dit, concat.T) + self.l2_lambda * self.Wi
        dbi = dit

        dft = dc_next * c_prev
        dft = self.sigmoid_derivative(ft) * dft
        dWf = np.dot(dft, concat.T) + self.l2_lambda * self.Wf
        dbf = dft

        dconcat = (np.dot(self.Wf.T, dft)
                   + np.dot(self.Wi.T, dit)
                   + np.dot(self.Wc.T, dcct)
                   + np.dot(self.Wo.T, dot))
        dh_prev = dconcat[:self.n_units, :]
        dc_prev = ft * dc_next

        return dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby, dh_prev, dc_prev

    def update(self, dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby, lr):
        dWf, dWi, dWc, dWo, dWy = self.clip_grads(dWf, dWi, dWc, dWo, dWy)
        self.Wf -= lr * dWf
        self.Wi -= lr * dWi
        self.Wc -= lr * dWc
        self.Wo -= lr * dWo
        self.Wy -= lr * dWy
        self.bf -= lr * dbf
        self.bi -= lr * dbi
        self.bc -= lr * dbc
        self.bo -= lr * dbo
        self.by -= lr * dby

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh_derivative(self, x):
        return 1 - x ** 2

    def clip_grads(self, *grads):
        return [np.clip(grad, -self.grad_clip, self.grad_clip) for grad in grads]

def preprocess_data(file_path):
    data = pd.read_csv(file_path, index_col="Date/Time")
    data = data.drop(columns=['Wind Direction (°)', 'Wind Speed (m/s)'])
    data = data.dropna()
    data = data[:-10]

    raw_values = data['LV ActivePower (kW)'].values.reshape(-1, 1)

    diff_values = difference(raw_values, 1)
    supervised = timeseries_to_supervised(diff_values, lag=24)
    train, test = supervised[:-168], supervised[-168:]

    min_val, max_val, train_scaled, test_scaled = scale(train, test)

    return raw_values, min_val, max_val, train_scaled, test_scaled

def difference(dataset, interval=1):
    return np.diff(dataset, n=interval, axis=0)

def scale(train, test):
    min_val = np.min(train)
    max_val = np.max(train)
    train_scaled = (train - min_val) / (max_val - min_val)
    test_scaled = (test - min_val) / (max_val - min_val)
    return min_val, max_val, train_scaled, test_scaled

def invert_scale(min_val, max_val, scaled_value):
    return scaled_value * (max_val - min_val) + min_val

def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df.values

def train_lstm(train_scaled, n_epochs=1, n_in=24, n_out=1, n_units=50, lr=0.001):
    lstm = LSTM(n_in, n_out, n_units)
    batch_size = 1

    for epoch in range(n_epochs):
        lstm.training = True
        h_prev = np.zeros((n_units, batch_size))
        c_prev = np.zeros((n_units, batch_size))
        loss = 0
        for i in range(train_scaled.shape[0]):
            X, y = train_scaled[i, :-1].reshape(-1, 1), train_scaled[i, -1]
            y_pred, h_prev, c_prev, concat, cct, it, ft, ot = lstm.forward(X, h_prev, c_prev)

            dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby, dh_prev, dc_prev = \
                lstm.backward(X, y, y_pred, h_prev, c_prev, h_prev, c_prev, concat, cct, it, ft, ot,
                              np.zeros_like(h_prev), np.zeros_like(c_prev))

            lstm.update(dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby, lr)

            loss += np.mean((y - y_pred) ** 2)

        if epoch % 2 == 0:
            print(f'Epoch {epoch}, Loss: {loss / train_scaled.shape[0]}')

    return lstm

def make_predictions(lstm, test_scaled, raw_values, min_val, max_val, train_length):
    predictions = []
    h_prev = np.zeros((lstm.n_units, 1))
    c_prev = np.zeros((lstm.n_units, 1))
    lstm.training = False

    for i in range(test_scaled.shape[0]):
        X, y = test_scaled[i, :-1].reshape(-1, 1), test_scaled[i, -1]
        yhat, h_prev, c_prev, _, _, _, _, _ = lstm.forward(X, h_prev, c_prev)
        yhat = invert_scale(min_val, max_val, yhat.item())
        yhat = yhat + raw_values[train_length + i].item()  # Ensure adding scalar value
        if yhat < 0:
            yhat = 0
        predictions.append(yhat)

    return np.array(predictions)

def evaluate_model(file_path):
    raw_values, min_val, max_val, train_scaled, test_scaled = preprocess_data(file_path)
    lstm = train_lstm(train_scaled)
    predictions = make_predictions(lstm, test_scaled, raw_values, min_val, max_val, len(train_scaled))

    actual = raw_values[-168:].flatten()
    mae = mean_absolute_error(actual, predictions)

    return mae, actual, predictions
