from statistics import mean
from datetime import date,datetime,timedelta
from pytz import timezone
import pandas as pd
import yfinance as yf
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from statistics import harmonic_mean


def geo_mean_overflow(x):
    if sum(x) == 0:
        return 0
    return np.exp(np.log(x).mean())

usfedcal = USFederalHolidayCalendar()

def BDay(num):
    return CustomBusinessDay(num, calendar=usfedcal)

def percent(a, b):
    a = float(a)
    b = float(b)
    diff = float(b - a)
    result = float((diff * 100) / a)
    return float(result)

def direction(num1, num2):
    #-1 = down
    #0 = same (I assume this'll never happen, but it's good to have that implemented)
    #1 = up
    if num2 > num1:
        return 1
    elif num1 > num2:
        return -1
    else:
        return 0

def direction_acc(y_train, y_pred_train, LOOKUP_STEP, stat_range=3650):
    matches = 0
    not_matches = 0

    if stat_range > len(y_train):
        ValueError("stat_range must be smaller than the length of the input data")

    for i_1 in range(stat_range):
        i = len(y_train)-1-i_1
        orig_direction = direction(y_train[i-LOOKUP_STEP], y_train[i])
        pred_direction = direction(y_pred_train[i-LOOKUP_STEP], y_pred_train[i])
        if orig_direction == pred_direction:
            matches += 1
        else:
            not_matches += 1
    
    return (matches/(matches+not_matches))*100
    
    

def acc_func(y_train, y_pred_train, close_range=2, stat_range=3650):
    diff_list = []
    acc_list = []
    close_matches = 0
    int_matches = 0

    if stat_range > len(y_train):
        ValueError("stat_range must be smaller than the length of the input data")

    for i_1 in range(stat_range):
        i = len(y_train)-1-i_1
        x = float(y_train[i])
        y_pred_train_ele = y_pred_train[i]
        if type(y_pred_train_ele) == np.ndarray:
            y_pred_train_ele = y_pred_train_ele[0]
        
        y = float(y_pred_train_ele)

        if int(y_train[i]) == int(y_pred_train_ele):
           int_matches += 1

        acc_data = abs(percent(float(x), float(y)))
        diff_data = abs(x-y)
        if close_range >= diff_data:
            close_matches += 1

        acc_list.append(acc_data)
        diff_list.append(diff_data)

    accuracy = float(100-mean(acc_list))
    mean_diff = mean(diff_list)
    geo_mean = geo_mean_overflow(diff_list)
    return accuracy, mean_diff, geo_mean, int_matches, close_matches

def time_compare(in_time1, in_close_time):
    time1 = int(in_time1.hour)*60 + int(in_time1.minute)
    close_time = in_close_time

    return time1 >= close_time

def find_next_bday(in_STEP):
    tz = timezone('EST')
    est_now = datetime.now(tz)

    today = datetime.today()
    # next_bday = est_now + BDay(in_STEP)
    # next_bday = next_bday.strftime("%m/%d/%y")
    close_time = 16*60
    if time_compare(today, close_time):
        next_bday = est_now + BDay(in_STEP)
    else:
        next_bday = est_now + BDay(in_STEP-1)
    next_bday = next_bday.strftime("%m/%d/%y")
    return next_bday

def current_date():
    today = datetime.today()
    close_time = 16*60
    if time_compare(today, close_time):
        return str((datetime.now(timezone('EST'))+timedelta(days=1)).strftime("%Y-%m-%d"))
    else:
        return str(datetime.now(timezone('EST')).strftime("%Y-%m-%d"))
    return

curr_date = current_date()
# curr_date = "2021-02-01"
print(curr_date)

#function to get data_ticker data
def data_tick_make(Ticker, opt="2000", exclude=False):
    data_ticker = yf.Ticker(Ticker)
    if exclude not in [False, 0]:
        if isinstance(exclude, int):
            data_ticker = yf.Ticker(Ticker)
            tz = timezone('EST')
            est_now = datetime.now(tz)
            today = datetime.today()
            max_date = est_now-BDay(exclude)
            max_date = max_date.strftime("%Y-%m-%d")
        else:
            print("[ERROR]: In function 'data_tick_make' has been specified an 'exclude' option, but isn't a int!")
            exit()
    else:
        max_date = curr_date

    if opt == "2000":
        data_ticker = data_ticker.history(start="2000-01-01", period="1d", end=max_date)
    elif opt == "max":
        data_ticker = data_ticker.history(period="max", end=max_date)
    elif opt == "1980":
        data_ticker = data_ticker.history(start="1980-01-01", period="1d", end=max_date)
    elif opt == "2010":
        data_ticker = data_ticker.history(start="2010-01-01", period="1d", end=max_date)
    elif opt == "2016":
        data_ticker = data_ticker.history(start="2016-01-01", period="1d", end=max_date)
    elif opt == "2020":
        data_ticker = data_ticker.history(start="2020-01-01", period="1d", end=max_date)
    elif opt == "2019":
        data_ticker = data_ticker.history(start="2019-01-01", period="1d", end=max_date)
    else:
        print("[ERROR]: In function 'data_tick_make': variable 'opt' is set to an invalid value!")
        exit()
    return data_ticker

#used in test.py to smooth data
def smooth_line(input, sigma=2):
    return gaussian_filter1d(input, sigma=sigma)

def curr_value(Ticker):
    # today = datetime.today()
    # close_time = 16*60
    # if not (time_compare(today, close_time)):
    #     print("[INFO]: stock market should be open right now! (or this program is broken)")
    # else:
    #     print("[INFO]: The stock market is closed!")
    curr_date = str((datetime.now(timezone('EST'))+timedelta(days=1)).strftime("%Y-%m-%d"))
    prev_date = str((datetime.now(timezone('EST'))-timedelta(days=1)).strftime("%Y-%m-%d"))
    data_ticker = yf.Ticker(Ticker)
    data_ticker = data_ticker.history(start=prev_date, period="1d", end=curr_date)
    return data_ticker['Close'][-1]
