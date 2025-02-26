import pandas as pd
import numpy as np
from Ensemble import Ensemble
from inputTools import generateInstances,walk_instances


col_name = ['Date','Open','High','Low','Close']
dataset = pd.read_csv('T9-RL/SP500st00.csv', names=col_name)
train_len = 1000
# dataset = pd.read_csv('T9-RL/US_XNAS_MSFT.csv', names=col_name)
# train_len = 1600
# dataset = pd.read_csv('T9-RL/TAIEX.csv', names=col_name)
# train_len = 2000 
inputData = dataset[['Open','High','Low','Close']].to_numpy()
inputData = inputData[0:2000]
# parameters
window_size = 8
num_classifier=1
num_epochs = 3
num_next_day = 1
num_sig = len(inputData[0])

instances = generateInstances(inputData,window_size,num_sig,num_next_day)
acc_list = []
profit_list = []
BH_profit = []
wlk_step = 100
for i in range(10):
    train_data,test_data = walk_instances(instances,(i*wlk_step),train_len,(i*wlk_step)+train_len,test_len = wlk_step)
    traderEnsemble = Ensemble(window_size,num_classifier,num_sig)
    traderEnsemble.train(train_data,num_epochs)
    profit = traderEnsemble.test(test_data)
    profit_list.append(profit)
    firstIndex = window_size*8 + i*wlk_step + train_len - 1
    BH_profit.append(inputData[firstIndex+wlk_step - 1, num_sig-1]-inputData[firstIndex, num_sig-1])
    print("result step#",i,"Profit:",profit)
    print("result step#",i,"ProfitBH:",inputData[firstIndex+wlk_step-1, num_sig-1]-inputData[firstIndex, num_sig-1])
print(profit_list)
print(np.sum(profit_list))
print(BH_profit)
print(np.sum(BH_profit))