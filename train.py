from libs.all import *
import numpy as np

if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")

data_ticker = data_tick_make(Ticker, opt=timeper, exclude=exclude_var)
# if (data_ticker['Close'][-1] == curr_value(Ticker)) and not (time_compare(datetime.today(), (16*60))):
#     print("UH OH")
#     exit(1)

    
data, model, model_name = multi_model(N_STEPS, LOOKUP_STEP, data_ticker, compile_info=True, layers=LAYERS, output_steps=output_steps, UNITS=UNITS, NOTE=timeper)
checkpointer = ModelCheckpoint(filepath=model_name, save_weights_only=False, save_best_only=True, verbose=1, monitor="loss", mode="min")

def train(model, input_epochs):
    y_train = tf.ragged.constant(data["y_train"])
    history = model.fit(data["X_train"], y_train,
                        batch_size=BATCH_SIZE,
                        epochs=input_epochs,
                        callbacks=[checkpointer],
                        verbose=0)

def initial_info():
    print("Info:")
    print("Input steps:", str(N_STEPS))
    print("Days ahead:", str(LOOKUP_STEP))
    print("Hidden Layers:", str(LAYERS))
    print("LOSS:", str(LOSS))
    print("OPTIMIZER:", str(OPTIMIZER))
    print("# of output steps:", str(output_steps))
    print("Next Pred date:", find_next_bday(LOOKUP_STEP-exclude_var-1))
    print('\n')

def train_loop(model, data, data_ticker, N_STEPS, LOOKUP_STEP, LAYERS):
    global tmp
    loop_num_times = 0
    info_print(model, data, feature, N_STEPS, LOOKUP_STEP, (exclude_var+1))
    if LOOP_NUM != False:
        for _ in range(LOOP_NUM):
            start = time.time()
            print(f'\nTraining for {EPOCHS} epochs...')
            train(model, EPOCHS)
            end = time.time()
            elapsed = end - start
            print("took", str(elapsed))
            model.save(model_name)
            loop_num_times += 1
    else:
        while True:
            start = time.time()
            print(f'\nTraining for {EPOCHS} epochs...')
            train(model, EPOCHS)
            end = time.time()
            elapsed = end - start
            print(f'took {elapsed}'')
            model.save(model_name)
            loop_num_times += 1


try:
    initial_info()
    train_loop(model, data, data_ticker, N_STEPS, LOOKUP_STEP, LAYERS)
    model.save(model_name)
except KeyboardInterrupt:
    print('Interrupted!')
    print(f'Looped {loop_num_times} times')
    model.save(model_name)
    # print("Calculating prediction before exiting...")
    # info_print(model, data, feature, N_STEPS, LOOKUP_STEP, (exclude_var+1))
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)