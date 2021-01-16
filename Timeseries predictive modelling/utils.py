import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

def adf_test(series, title='', print_results=True):
    """
    Pass in a time series and an optional title, returns the p-value of the
    Augmented Dickey-Fuller Test (for stationarity)
    """
    result = adfuller(series.dropna(), autolag='AIC') # .dropna() since differencing produces NaNs
    if print_results:
        print(f'Augmented Dickey-Fuller Test: {title}')
        print("Test the null hypothesis that the data is non-stationary")
        labels = ['ADF test statistic','p-value','lags', '# observations']
        out = pd.Series(result[0: 4],index=labels)

        for key,val in result[4].items():
            out[f'critical value ({key})']=val

        print(out.to_string())          # .to_string() removes the line "dtype: float64"

        if result[1] <= 0.05:
            print("Reject the null hypothesis, data is stationary")
        else:
            print("Fail to reject the null hypothesis, data is  non-stationary")
    return result[1]


def print_errors(y_true, y_pred, msg):
    print('ERRORS in ' + msg)
    print('MAE: {0}'.format(mean_absolute_error(y_true, y_pred)))
    print('RMSE: {0}'.format(np.sqrt(mean_squared_error(y_true, y_pred))))
    print('MAPE: {0}'.format(np.mean(np.absolute(y_true - y_pred)/(y_true+1)*100)))
    print('RMSPE: {0}'.format(np.sqrt(np.mean(np.square(((y_true - y_pred) /(y_true+1))), axis=0))))