# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import os
from pandas import read_csv, to_datetime, DataFrame
from matplotlib import pyplot as plt
from fbprophet import Prophet
'''
# For connection with HANA DB

from hdbcli import dbapi
from hana_ml import dataframe

ADDRESS = '172.0.1.211'
PORT = '30015'
USER = str(os.getenv('HANADB_USER'))
PASS = str(os.getenv('HANADB_PASS'))

with dataframe.ConnectionContext(ADDRESS, PORT, USER, PASS) as hana_conn:
'''


def get_sales_forecast():
    # load data
    path = os.getcwd() + '\\sales_2018-20.csv'
    df = read_csv(path, header=0)
    # prepare expected column names
    df.columns = ['ds', 'y']
    # define the model
    model = Prophet()
    # fit the model
    model.fit(df)
    # define the period for which we want a prediction
    future = list()
    for i in range(1, 13):
        date = '2021-%02d' % i
        future.append([date])
    future = DataFrame(future)
    future.columns = ['ds']
    future['ds'] = to_datetime(future['ds'])
    # use the model to make a forecast
    forecast = model.predict(future)
    # summarize the forecast
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    # plot forecast
    plt.style.use('dark_background')

    fig = model.plot(forecast, xlabel='', ylabel='Millones USD')
    ax = fig.gca()
    ax.set_title('Proyección Ventas 2021')
    ax.plot(to_datetime(df['ds']), df['y'], 'o-', color='gold', linewidth=2)
    ax.plot(to_datetime(forecast['ds']), forecast['yhat'], 'o-', color='blue', linewidth=2)
    # format for dark background
    ax.spines['bottom'].set_color('#dddddd')
    ax.spines['top'].set_color('#dddddd')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.yaxis.label.set_color('black')
    # ax.xaxis.label.set_color('black')
    ax.title.set_color('black')

    # plt.legend()
    plt.grid(True)
    plt.minorticks_on()
    plt.show()


if __name__ == '__main__':
    get_sales_forecast()

