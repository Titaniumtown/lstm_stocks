from libs.all import *
import matplotlib.pyplot as plt

print(curr_date)

data_ticker = data_tick_make(Ticker, opt=timeper, exclude=exclude_var)
data, model, model_name = multi_model(N_STEPS, LOOKUP_STEP, data_ticker, layers=LAYERS, NOTE=timeper, UNITS=UNITS)

forecast = fcast(feature, data, model, N_STEPS)

print("Processing data...")
X_train = data["X_train"]
y_train_ttp = data["y_train"]
y_train = np.squeeze(data["column_scaler"][feature].inverse_transform(np.expand_dims(y_train_ttp, axis=0)))
print(y_train)
print("Predicting")
y_pred_train_ttp = model.predict(X_train)
y_pred_train = np.squeeze(data["column_scaler"][feature].inverse_transform(y_pred_train_ttp))
len_data = int(len(data["y_train"]))

info_print(model, data, feature, N_STEPS, LOOKUP_STEP, (exclude_var+1))

forecasted = list(y_pred_train)
forecasted.append(forecast)

print(forecast)
