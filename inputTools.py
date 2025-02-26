import numpy as np

def window2image(list):
#normalize
    max_value = max(list)
    min_value = min(list)
    window_size = len(list)
    norm_list = np.zeros((window_size))
    i=0
    for x in list:
        norm_list[i] = ((x-max_value)+(x-min_value))/(max_value-min_value)
        i+=1
#transform2polar
    teta = np.arccos(norm_list)
 #GDAF
    out = np.zeros((window_size,window_size))
    for i in range(window_size):
        for j in range(window_size):
            out[i][j] = np.sin(teta[i]+teta[j])
    return out.reshape(window_size,window_size)

def mergeImages(img1,img2,img4,img8):
    r1 = np.concatenate((np.array(img1),np.array(img2)),axis=1)
    r2 = np.concatenate((np.array(img4),np.array(img8)),axis=1)
    return np.concatenate((r1,r2),axis=0)

def generateInstances(data,window_size,num_sig,num_next_day):

    
    instances = []
    num_next_day -=1
    inst = Instance([],0,0)
    for i in range(window_size*8,len(data)-num_next_day):
        if(i%500==0):
            print("# of Sample: ",i)

        img3d = np.zeros([window_size*2,window_size*2,num_sig])
        for j in range(num_sig):
            img1 = genSubImage(data[:,j],i,window_size,1)
            img2 = genSubImage(data[:,j],i,window_size,2)
            img4 = genSubImage(data[:,j],i,window_size,4)
            img8 = genSubImage(data[:,j],i,window_size,8)
            img3d[:,:,j] = mergeImages(img1,img2,img4,img8)

        # inst.set_next_state(img3d)
        inst = Instance(img3d,data[i][num_sig-1]-data[i-1][num_sig-1],data[i-1][num_sig-1])
        instances.append(inst)
    return instances

def genSubImage(data,data_index,window_size,step):
    list = np.zeros(window_size)
    ind = data_index-step
    for i in range(window_size):
        list[window_size-i-1] = data[ind]
        ind -= step
    return window2image(list)

class Instance():
    def __init__(self,state,profit,closeValue):
        # self.day_index = day_index
        self.closeValue = closeValue
        self.state = state
        self.profit = profit
        # self.nextState = []
    # def set_next_state(self,img):
    #     self.nextState = img
    
def walk_instances(data,train_index, train_len,test_index, test_len):
    train_data = []
    test_data = []
    for i in range(train_index,train_index+train_len):
        train_data.append(data[i])
    for i in range(test_index,test_index+test_len):
        test_data.append(data[i])
    return train_data,test_data
