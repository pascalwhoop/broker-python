import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis


def make_sequences(timeseries, seq_len, normalise_window):
    sequence_length = seq_len + 1
    result = []
    for index in range(len(timeseries) - sequence_length):
        result.append(timeseries[index: index + sequence_length])


    if normalise_window:
        result = normalise_windows(result)

    sequences = np.array(result)
    return sequences

def make_train_data(sequences):
    """taking 90% of the sequences and using them for training, then 10% for verification"""
    row = round(0.9 * sequences.shape[0])
    train = sequences[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1] #up to last entry in sequence
    y_train = train[:, -1]  #last entry --> what to forecast
    x_test = sequences[int(row):, :-1]
    y_test = sequences[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def find_first_non_zero(window) -> float:
    for v in window:
        if v is not 0 or 0.0:
            return v
    return 0.0


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        base = find_first_non_zero(window)
        if base is 0:
            return window_data
        normalised_window = [(p / base - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for _ in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
