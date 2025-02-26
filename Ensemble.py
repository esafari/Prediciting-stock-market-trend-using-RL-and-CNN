import matplotlib.pyplot as plt
import numpy as np
import math
from TraderAgent import TraderAgent


class Ensemble():
    def __init__(self,window_size,num_classifier,num_sig):
        self.num_classifier = num_classifier
        self.classifierList = self.ensemble_builder(window_size,num_sig)
        
    def ensemble_builder(self,window_size,num_sig):
        classifierList = []
        for i in range(self.num_classifier):
            classifierList.append(TraderAgent(window_size,num_sig))
        return classifierList

    def train(self,train_data,num_epochs):
        data_size = len(train_data)-5
        print("Training:")
        for ep in range(num_epochs):
            print("epoch #:",ep,"...")
            for i in range(self.num_classifier):
                for j in range(10):
                    samplesState = []
                    samplesNextState = []
                    samplesQ = []
                    for j2 in range(math.floor(data_size/10)):
                        index = j*math.floor(data_size/10) + j2
                        samplesState.append(train_data[index].state)
                        samplesNextState.append(train_data[index+4].state)
                    v_next = self.classifierList[i].test(np.array(samplesNextState))
                    for j2 in range(math.floor(data_size/10)):
                        index = j*math.floor(data_size/10) + j2
                        futVal = np.max(v_next[j2])
                        samplesQ.append([train_data[index].profit + 
                                        (.9)*train_data[index+1].profit + 
                                        (.9**2)*train_data[index+2].profit + 
                                        (.9**3)*train_data[index+3].profit + 
                                        (.9**4)*futVal, 0])
                    self.classifierList[i].train(np.array(samplesState),np.array(samplesQ))
    
    def test(self, test_data):
        data_size = len(test_data)
        testSamplesState = []
        for i in range(data_size):
            testSamplesState.append(test_data[i].state)
            # plt.figure()
            # plt.imshow(test_data[i].state)
        output = []
        print("Testing ...")
        for i in range(self.num_classifier):
            output.append(self.classifierList[i].test(np.array(testSamplesState)))
        
        sums = np.sum(output,axis=0)
        strategy = np.argmax(sums,axis=1)

        netProfit = 0
        buyValue = 0
        state = 0               # state == hasNot
        for j in range(data_size):
            if(strategy[j] == 0 and state == 0):
                state = 1       # state = has
                buyValue = test_data[j].closeValue
            elif(strategy[j] == 1 and state == 1):
                state = 0       # state == hasNot
                netProfit += test_data[j].closeValue - buyValue

        if state == 1:
            netProfit += test_data[data_size-1].closeValue - buyValue

        return netProfit