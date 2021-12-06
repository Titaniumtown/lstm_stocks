from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import gc, time
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.client import device_lib
from collections import deque
from statistics import mean
from pandas import DataFrame
from numpy import split,array,expand_dims,asarray
from sklearn import preprocessing
from tensorflow.keras.callbacks import ModelCheckpoint
from libs.funcs import acc_func, percent, find_next_bday, BDay, direction_acc, geo_mean_overflow
from libs.misc_vars import *
from libs.train_vars import *
from datetime import date,datetime
from pytz import timezone
import numpy as np


def load_data(data, n_steps=50, scale=True, shuffle=False, lookup_step=1, feature_columns=FEATURE_COLUMNS[0], output_steps=1):
    df = DataFrame(data)
    result = {'df': df.copy()}
    for col in feature_columns:
        assert col in df.columns

    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        result["column_scaler"] = column_scaler

    df['future'] = df[FEATURE_COLUMNS[0]].shift(-lookup_step)

    last_sequence = array(df[feature_columns].tail(lookup_step))

    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([array(sequences), target])

    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = array(DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result['last_sequence'] = last_sequence

    X, y, y_orig = [], [], []

    for i in range(len(sequence_data) - output_steps):
        X.append(sequence_data[i][0])
        target = sequence_data[i][1]
        y_orig.append(target)

        if output_steps > 1 and i != 0:
            target_new = []
            i_1 = 0
            while output_steps > i_1:              
                target_new.append(float(df[feature_columns].values[i+i_1]))

                i_1 += 1

            target = target_new

        y.append(asarray(target, dtype=np.float32))

    X = array(X)
    y = array(y)
    y_orig = array(y_orig)

    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    result["X_train"], result["y_train"], result["date"], result["y_train_orig"] = X, y, df.index, y_orig
    return result

def forecast_loop(futuredays, feature, data, model, N_STEPS, verbose=0):
    forecast = []
    for i in range(futuredays):
        if i == 0:
            last_sequence = data["last_sequence"][:N_STEPS]
        else:
            last_sequence = data["last_sequence"][:N_STEPS] + forecast

        last_sequence = last_sequence[:N_STEPS]

        column_scaler = data["column_scaler"]

        last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))

        last_sequence = expand_dims(last_sequence, axis=0)

        predicted_price = float(column_scaler[feature].inverse_transform(model.predict(last_sequence))[0][0])
        # predicted_price = (int(predicted_price*100)/float(100))
        forecast.append(predicted_price)
    return forecast

def fcast(feature, data, model, N_STEPS):
    last_sequence = data["last_sequence"][:N_STEPS]
    column_scaler = data["column_scaler"]
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    last_sequence = expand_dims(last_sequence, axis=0)
    return float(column_scaler[feature].inverse_transform(model.predict(last_sequence))[0][0])

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
    
def tf_device_test():
    tf.config.optimizer.set_jit(True)
    options = get_available_gpus()
    if "/device:XLA_GPU:0" in options:
        return "/XLA_GPU:0"
    elif "/device:XLA_CPU:0" in options:
        return "/XLA_CPU:0"
    else: 
        return "/CPU:0"

def multi_model(INPUT_STEPS, OUT_LOOKUP_STEP, input_data, compile_info=False, layers=4, verbose=1, fail_wo_file=0, output_steps=1, UNITS=False, NOTE="def"):
    if UNITS == False:
        UNITS = INPUT_STEPS
    
    data = load_data(input_data, INPUT_STEPS, lookup_step=OUT_LOOKUP_STEP,
                    feature_columns=FEATURE_COLUMNS, shuffle=False, output_steps=output_steps)

    model = create_model(data=data, loss=LOSS, units=UNITS, cell=CELL, optimizer=OPTIMIZER, compile=compile_info, STEPS=INPUT_STEPS, opt=layers, output_steps=output_steps)

    model_name_sel = f"{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{INPUT_STEPS}-step-{OUT_LOOKUP_STEP}-layers-{layers}-units-{UNITS}-{feature}-{Ticker_Name}-outnum-{output_steps}"

    if NOTE == "def":
        model_name_sel = model_name_sel
    elif NOTE == "max":
       model_name_sel = f"{model_name_sel}-note-maxperiod"
    else:
        model_name_sel = f"{model_name_sel}-note-{str(NOTE)}"

    saved_model_path = str(os.path.join("results", model_name_sel) + ".h5")
    if verbose != 0:
        print(saved_model_path)
    if os.path.exists(saved_model_path):
        if verbose != 0:
            print("Model exists, loading now!")
        model.load_weights(saved_model_path)
    else:
        if verbose != 0:
            print("No model found")
        if fail_wo_file == 1:
            print("ERROR: Function multi_model was run with argument 'fail_wo_file=1' which fails w/o finding a file")
            exit()
    
    return data, model, saved_model_path

def create_model(data, units=256, cell=LSTM,
                loss="mean_absolute_error", optimizer="rmsprop", compile=True, STEPS=60, opt=4, output_steps=1):
    model = Sequential()
    model.add(cell(units, return_sequences=True, input_shape=(None, STEPS))) #input layer

    for _ in range(opt): model.add(cell(units, return_sequences=True)) #hidden layers

    model.add(cell(units, return_sequences=False)) #Last layer before output layer

    model.add(Dense(output_steps, activation="linear")) #Output Layer

    if compile:
        model.compile(loss=loss, metrics=["mean_absolute_error", tf.keras.metrics.Accuracy()], optimizer=optimizer)
    return model


def info_print(model, data, feature, N_STEPS, LOOKUP_STEP, exclude_var):
    stat_range = 365*20

    if int(stat_range) != stat_range:
        ValueError("stat_range is not an integer")
    
    if stat_range >= 365:
        print(f"Range in which statistics are generated from: {round(stat_range/365, 5)} years")
    else:
        print(f"Range in which statistics are generated from: {stat_range} days")

    close_range = 2
    len_data = int(len((data["y_train"])[:stat_range]))
    last_bday = (datetime.now(timezone('EST'))-BDay(LOOKUP_STEP)).strftime("%m/%d/%y")
    forecast = round(fcast(feature, data, model, N_STEPS), 3)

    X_train = data["X_train"]
    y_train = data["y_train_orig"]

    y_train = np.squeeze(data["column_scaler"][feature].inverse_transform(np.expand_dims(y_train, axis=0)))
    y_pred_train = model.predict(X_train)
    y_pred_train = np.squeeze(data["column_scaler"][feature].inverse_transform(y_pred_train))

    accuracy, mean_diff, geo_mean, int_matches, close_matches = acc_func(y_train, y_pred_train, close_range=close_range, stat_range=stat_range)
    next_bday = find_next_bday(LOOKUP_STEP-exclude_var)

    dir_acc = direction_acc(y_train, y_pred_train, LOOKUP_STEP, stat_range=stat_range)

    print(f'Avg diff from ground truth: {mean_diff}\nDirectional accuracy: {dir_acc}%\nGeometric mean diff from ground truth: {geo_mean}')

    print(f'Number of matches (rounded to int):, {int_matches}/{len_data} \n\tPercent matched: {round(float(float(int_matches/len_data)*100), 5)}%')

    print(f'Number of matches (within {close_range} points): {close_matches}/{len_data}\n\tPercent matched: {round(float(float(close_matches/len_data)*100), 5)}%')

    print(f'Average Accuracy: {round(accuracy, 5)}%')
    print(f"last input day: {last_bday}")
    print(f'Prediction for {next_bday}: {forecast}')

