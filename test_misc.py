from libs.misc_vars import *
from libs.funcs import curr_value
import yfinance as yf
from datetime import timedelta
from pytz import timezone
from datetime import datetime
from pandas.tseries.offsets import BDay

data_ticker = yf.Ticker(Ticker)
tz = timezone('EST')
est_now = datetime.now(tz)
today = datetime.today()
# max_date = est_now-BDay(1)
max_date = est_now
max_date = max_date.strftime("%Y-%m-%d")

data_ticker = data_ticker.history(start="2020-01-01", period="1d", end=max_date)

print(data_ticker['Close'])