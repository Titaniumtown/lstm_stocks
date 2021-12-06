from libs.all import *
import matplotlib.pyplot as plt

def forecast_graph(futdays, data, model, N_STEPS, LOOKUP_STEP, opts=[10]):
    forecast = fcast(feature, data, model, N_STEPS)

    print("Processing data...")
    X_train = data["X_train"]
    y_train_ttp = data["y_train"]
    y_train = np.squeeze(data["column_scaler"][feature].inverse_transform(np.expand_dims(y_train_ttp, axis=0)))
    print("Predicting")
    y_pred_train_ttp = model.predict(X_train)
    y_pred_train = np.squeeze(data["column_scaler"][feature].inverse_transform(y_pred_train_ttp))
    len_data = int(len(data["y_train"]))

    info_print(model, data, feature, N_STEPS, LOOKUP_STEP, exclude_var)

    forecasted = list(y_pred_train)
    forecasted.append(forecast)


    #Graph of all Ground Truth and the Prediction
    print("Creating plot...")
    plt.figure(1)
    plt.ticklabel_format(style='plain')
    plt.plot(y_pred_train[-len(y_train):], color='r', label="Predicted")
    plt.plot(y_train[-len(y_train):], color='b', label="Ground Truth")
    plt.legend()

    lengthOffset = 0
    if not LOOKUP_STEP in [0, False]:
        lengthOffset = LOOKUP_STEP

    lengthOffset = 1 - lengthOffset

    length = int(opts[0]+lengthOffset+1)
    forcasted_done = forecasted[-length:]
    
    #Graph of the past 9 predictions and ground truth, with the future prediction
    print("Creating plot...")
    plt.figure(2)
    plt.ticklabel_format(style="plain")
    plt.plot(forcasted_done, color='r', label="Predicted")
    plt.plot(y_train[-(opts[0]):], color='b', label="Ground Truth")
    print(f"Predicted: {str(forcasted_done[:-1])}")
    print(f"Ground Truth: {str(y_train[-opts[0]:])}")
    print(f"Future prediction: {str(forcasted_done[-1])}")
    plt.legend()

    #true to pred graph
    plt.figure(3)
    plt.scatter(y_train, y_pred_train_ttp, s=0.5)

    #show all of the graphs
    plt.show()

data_ticker = data_tick_make(Ticker, opt=timeper, exclude=exclude_var)
data, model, model_name = multi_model(N_STEPS, LOOKUP_STEP, data_ticker, layers=LAYERS, NOTE=timeper, UNITS=UNITS)

try:
    forecast_graph(10, data, model, N_STEPS, LOOKUP_STEP, opts=[10])
except KeyboardInterrupt:
    print('Interrupted!')
    tf.keras.backend.clear_session()
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
