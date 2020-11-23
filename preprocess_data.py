import pandas as pd
import numpy as np
import pickle
import h5py
from sklearn.preprocessing import MinMaxScaler
import pdb

def prepair_data(path,window_x,window_y,tickers=[],period=[]):
    df = pd.read_csv(path)
    df['date'] = df.date.apply(pd.Timestamp)

    df['dow'] = df.date.apply(lambda x: x.dayofweek)
    # ## max 500 tickers
    # num_tickers = df['ticker'].unique()
    # num_tickers = num_tickers[:500]
    # df = df[df['ticker'].isin(num_tickers)]
    #---------
    if len(tickers)>0:
        df = df[df['ticker'].isin(tickers)]
    ## just select working days
    df = df[(df.dow<=4)&(df.dow>=0)]
    df = df.drop(['dow'],axis=1)

    df = df.pivot_table(index='date', columns='ticker')
    ## select date
    if len(period)!=0:
        df = df[(df.index> period[0]) & (df.index < period[1])]

    ## select tickers not nan in final day
    columns = df.close.columns[~df.close.iloc[-1].isna()]
    df = df.iloc[:, df.columns.get_level_values(1).isin(columns)]
    


    df.volume = df.volume.interpolate(method='linear',limit_area='inside',limit_direction='both', axis=0)
    df.close = df.close.interpolate(method='linear',limit_area='inside',limit_direction='both', axis=0)


    close = df.close
    daily_return = ((close.shift(-1) - close)/close).shift(1)

    daily_return = daily_return.interpolate(method='linear',limit_area="inside",limit_direction='both', axis=0)
    # daily_return = daily_return.fillna(daily_return.min(axis=0),axis=0)
    # daily_return = daily_return.fillna(daily_return.min(axis=0),inplace=True)

    # daily_return.fillna(daily_return.min(axis=0), inplace=True)

    tickers = df.close.columns
    # date = df.index
    ## change datetime64 to string
    date = df.index.strftime('%Y-%m-%d')
    date = [n.encode("ascii", "ignore") for n in date]
    # pdb.set_trace()

    X = df.values.reshape(df.shape[0],2,-1)
    ## Using for max scaling data
    # X_max = X.max(axis=0)[np.newaxis,np.newaxis,:,:]
    y = daily_return.values

    ## fill X
    ##fill nan by 0.0
    X[np.isnan(X)] = 0.0

    ## fill y
    y[np.isnan(y)] = -1e2
    # y[np.isnan(y)] = 0

    # X1 = rolling_array(X[window_x:],stepsize=1,window=window_y)

    #### using h5 file for big data
    h5file = h5py.File(path.replace('csv','h5'), 'w')
    dsetX = h5file.create_dataset('X', (X.shape[0] - window_x - window_y + 1, X.shape[-1],window_x, X.shape[1]), dtype=np.float32)
    dset_dateX = h5file.create_dataset('date_X', (X.shape[0] - window_x - window_y + 1, window_x), dtype='S10')
    dsety = h5file.create_dataset('y', (X.shape[0] - window_x - window_y + 1, y.shape[1],window_y), dtype=np.float32)
    dset_datey = h5file.create_dataset('date_y', (X.shape[0] - window_x - window_y + 1, window_y),  dtype='S10')
    stepsize = 1
    for i in range(0,X.shape[0] - window_x - window_y + 1):
        dsetX[i:i+1] = np.moveaxis(X[:-window_y][i:i + window_x:stepsize],-1,0)
        dset_dateX[i:i+1] = date[:-window_y][i:i + window_x:stepsize]
        dsety[i:i+1] = np.swapaxes(y[window_x:][i:i + window_y:stepsize],1,0)
        dset_datey[i:i+1] = date[window_x:][i:i + window_y:stepsize]
    tickers = [n.encode("ascii", "ignore") for n in tickers]
    h5file.create_dataset('ticker', (len(tickers),1),'S10', tickers)
    h5file.close()
    return path.replace('csv','h5')

    

    # X = rolling_array(X[:-window_y],stepsize=1,window=window_x)
    # y = rolling_array(y[window_x:],stepsize=1,window=window_y)
    # dates_X = rolling_array(date[:-window_y],stepsize=1,window=window_x)
    # dates_y = rolling_array(date[window_x:],stepsize=1,window=window_y)
    # X = np.moveaxis(X,-1,1)
    # y = np.swapaxes(y,1,2)

    # return X,y,tickers,dates_X,dates_y
    

# def rolling_array(a, stepsize=1, window=60):
#     n = a.shape[0]
#     return np.stack((a[i:i + window:stepsize] for i in range(0,n - window + 1)),axis=0)
