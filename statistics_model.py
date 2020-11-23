import scipy.optimize as sco
from sklearn.model_selection import TimeSeriesSplit
from multiprocessing import Pool
from preprocess_data import *
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pdb
def aggregate_data(list_X):
    data = []
    data.append(list_X[0])
    for i in range(1,len(list_X)):
        data.append(list_X[i][:,-1:,:])
    data = np.concatenate(data,axis=1)
    return data

def negative_sharpe(weight,data):
    """data is price have shape (tickers,days) """
    epsilon = 1e-6
    constraint_value = 3e-3
    sum_w = np.clip(weight.sum(),epsilon,data.shape[0])

    norm_weight = weight/sum_w
    norm_weight = norm_weight[:,np.newaxis]
    port_return = norm_weight.T.dot(data).squeeze()
    mean = np.mean(port_return)
    std = np.maximum(np.std(port_return),epsilon)
    constraint =  np.sum((1.6 - weight) * weight)
    return -mean/std + constraint_value * constraint

def cal_sharpe_ratio(weight,y):
    """Here y is the daily return have the shape (tickers,days) 
    weight have the shape (tickers,)"""
    epsilon = 1e-6
    weights = np.greater(np.clip(weight,0,1),0.5).astype(int)
    sum_w = np.clip(weights.sum(),epsilon,y.shape[0])
    norm_weight = weights/sum_w
    port_return = norm_weight.dot(y).squeeze()
    mean = np.mean(port_return)
    std = np.maximum(np.std(port_return),epsilon)
    return mean/std


def max_sharpe(data,y):
  
    initial_guess = np.ones(shape=data.shape[0])
    ## bound values for weights
    bounds = ((0.,1.),) * data.shape[0]
    args = (data)
    result = sco.minimize(
        negative_sharpe,
        x0=initial_guess,
        args=args,
        method="SLSQP",
        bounds=bounds,
        options={'maxiter':50}
    )
    weights = result["x"]
    print(result["message"])
    # print(weights.shape)
    sharpe = cal_sharpe_ratio(weights,y)
    sharpe_old = cal_sharpe_ratio(weights,data)
    return sharpe,weights,sharpe_old


path_data = '/vinai/hieuck/Porfolio/DELAFO-DeEp-Learning-Approach-for-portFolio-Optimization/data/data.csv'
timesteps_input = 64
timesteps_output = 19
n_fold = 10
X,y,tickers,date_X,date_y = prepair_data(path_data,window_x=timesteps_input,window_y=timesteps_output)
# shape x (N_ticker,M,2), shape y(N'_ticker,M'_days)
# pdb.set_trace()
score = []
tscv = TimeSeriesSplit(n_splits=n_fold)
dates = []
take_tickers = ['VNINDEX','VN30']
take_tickers_data = {i:[] for i in take_tickers}
index_tickers = [np.where(tickers==i)[0][0] for i in take_tickers]
# pdb.set_trace()
for train_index, test_index in tscv.split(X):
    
    X_tr, X_val = X[train_index], X[test_index[range(timesteps_output-1,len(test_index),timesteps_output)]]
    date_X_val = date_X[test_index[range(timesteps_output-1,len(test_index),timesteps_output)]]
    y_tr, y_val = y[train_index], y[test_index[range(timesteps_output-1,len(test_index),timesteps_output)]]
    date_y_val = date_y[test_index[range(timesteps_output-1,len(test_index),timesteps_output)]]
    results = []
    d = []
    for i in range(len(take_tickers)):
        take_tickers_data[take_tickers[i]].append(np.mean([cal_sharpe_ratio(np.array([1,]),y_i[index_tickers[i]:index_tickers[i]+1,:]) for y_i in y_val]))
    X_app = [x for x in X_tr]
    for i in range(len(X_val)):
        X_app += [x for x in X[test_index[range(0,i*timesteps_output+1)]]]
        dd = aggregate_data(X_app)
        d.append((X_val[i][:,1:,0] - X_val[i][:,:-1,0])/(X_val[i][:,:-1,0] + 1e-6))
        # d.append((dd[:,1:,0] - dd[:,:-1,0])/(dd[:,:-1,0] + 1e-6))
        dates.append((date_X_val[i],date_y_val[i]))
    #     results.append(max_sharpe(d,y_val[i]))
    # score.append(np.mean(results))

    with Pool(processes=6) as pool:
        results = pool.starmap(max_sharpe, list(zip(d,y_val)))
    score += results
# pdb.set_trace()
# print('Mean of Sharpe ratio %0.4f'%(np.mean(score)))
# print('STD of Sharpe ratio %0.4f'%(np.std(score)))
dateX = [t[0][-1] for t in dates]
datey = [t[1][-1] for t in dates]
sharpey = [x[0] * np.sqrt(19) for x in score ]
sharpex = [x[-1] * np.sqrt(19) for x in score ]
plt.plot(dateX[28:],sharpex[28:],'o--',color='red',label='Past')
plt.plot(datey[28:],[t/6000 for t in sharpey[28:]],'v-.',color='blue',label='Future')
# plt.plot(date,sharpe,'d:',color='green',label='Model')
# plt.plot(range(len(sharpe)),sharpe_FIR,color='purple',label='FIR')
plt.ylabel("Sharpe ratio")
plt.xticks(rotation=60)
plt.legend()
plt.tight_layout()
plt.savefig('destination_path.eps', format='eps')
pdb.set_trace()
print('Complete')
# plt.show()
