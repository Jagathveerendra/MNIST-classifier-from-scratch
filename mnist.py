import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def padding(data,mini_batch_size):
    row,col = data.shape
    k = col//mini_batch_size
    data = data[:,0:k*mini_batch_size]
    return data


def data_standardization(data):
    # row,col = data.shape
    # mean = np.mean(data,axis = 1)
    # data_1 = data - mean.reshape(row,1)
    # var = np.mean(data_1**2,axis = 1) + 0.00000001
    # data_2 = data_1/var.reshape(row,1)
    data_2 = data/255
    return data_2


def label_encoding(labels, clases_count):
    labels_encoded = np.zeros((len(labels),clases_count))
    for i in range(len(labels)):
        labels_encoded[i,labels[i]] = 1
    return labels_encoded.T


def initialization_parameters(layer_dims):

    parameters = {}
    for i in range(1,len(layer_dims)):
        parameters["W"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*np.sqrt(2/layer_dims[i-1])
        parameters["b"+str(i)] = np.zeros((layer_dims[i],1))
        # print( parameters["W"+str(i)].shape,"w"+str(i))
        # print(parameters["b" + str(i)].shape, "b" + str(i))
    return parameters


def sigmoid(x):
    return 1/(np.exp(-x)+1)

# derivative of sigmoid
def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)


def accuracy(y_preds, labels):
    z = np.argmax(y_preds,axis = 0)
    l = np.argmax(labels,axis = 0)
    mask = z==l
    accuracy = mask.mean()
    return accuracy*100

# derivative of softmax
def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))

def ReLU(x):

    return abs(x) * (x > 0)


def forward_propagation(data, parameters, no_of_layers):
    cache = {}
    for i in range(1,no_of_layers+1):
        if i==1:
            cache["Z"+ str(i)] = np.dot(parameters["W"+ str(i)],data) + parameters["b"+str(i)]
        else:
            cache["Z" + str(i)] = np.dot(parameters["W" + str(i)],cache["A"+str(i-1)]) + parameters["b" + str(i)]
        if i <= no_of_layers-1:
            cache["A" + str(i)] = sigmoid(cache["Z"+str(i)])
        else:
            cache["A" + str(i)] = softmax(cache["Z"+str(i)])
        # print(cache["Z"+str(i)].shape,"Z"+str(i))
        # print(cache["A" + str(i)].shape,"A"+str(i))

    y_preds = cache["A" + str(no_of_layers)]
    return cache, y_preds


def cost_function(y_preds,labels):
    cost__ = -np.sum(labels * np.log(y_preds),axis = 0)
    cost = np.mean(cost__)
    #print("cost = ",cost)
    return cost


def relu_backward(z):

    return z>0




def backward_propagation(data, cache, parameters, no_of_layers,y_preds,labels, mini_batch_size):
    bk_cache = {}
    derv_param = {}
    l = no_of_layers
    m = mini_batch_size
    bk_cache["dz" +str(l)] = y_preds - labels
    while(l>0):
        if l==1:
            derv_param["dw"+str(l)] = np.dot(bk_cache["dz" + str(l)],data.T)/m
        else:
            derv_param["dw" + str(l)] = np.dot(bk_cache["dz" + str(l)], cache["A" + str(l-1)].T) / m

        derv_param["db" + str(l)] = np.sum(bk_cache["dz" + str(l)],axis=1,keepdims=True)/m
        if l>1:
            bk_cache["dz" + str(l-1)] = np.dot(parameters["W"+str(l)].T,bk_cache["dz" + str(l)]) * d_sigmoid(cache["Z"+str(l-1)])
        l = l-1
    return derv_param


def update_parameters(parameters, derv_param, learning_rate, no_of_layers):

    for l in range(1,no_of_layers+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * derv_param["dw" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * derv_param["db" + str(l)]

    return parameters


def model(data, mini_batch_size, layer_dims, no_of_layers, learning_rate, labels, no_of_iterations):
    l = mini_batch_size
    parameters = initialization_parameters(layer_dims)
    costs = []
    acuracy = []
    row,col = data.shape
    j = 1
    for i in range(no_of_iterations):
        k = 0
        while(k<col):
            batch = data[:,k:k+l]
            batch_labels = labels[:,k:k+l]

            cache,y_preds = forward_propagation(batch,parameters,no_of_layers)
            if j%100 == 0:
                cost = cost_function(y_preds,batch_labels)
                acuracy.append(accuracy(y_preds,batch_labels))
                costs.append(cost)
            derv_param = backward_propagation(batch,cache,parameters,no_of_layers,y_preds,batch_labels,mini_batch_size)
            parameters = update_parameters(parameters, derv_param, learning_rate, no_of_layers)
            k += l
            j += 1

        #learning_rate = (0.95) * learning_rate
    return costs,acuracy,parameters

directory = "C:\\Users\\hp\\Downloads\\digit-recognizer\\train.csv"

data_mnist = np.genfromtxt(directory,delimiter = ",",skip_header=1,dtype= int)
data = data_mnist[:40000,1:].T
# print(data[:,0])
# print(data.shape)
learning_rate = 0.01
mini_batch_size = 128
data = padding(data,mini_batch_size)

labels_before_enc = data_mnist[:40000,0]
# print("labels_before_enc",labels_before_enc)
# print(labels_before_enc.shape)

k = len(labels_before_enc)//mini_batch_size
labels_before_enc = labels_before_enc[0:k*mini_batch_size]
# print(labels_before_enc)
# print(labels_before_enc.shape)

labels = label_encoding(labels_before_enc,10)
# print("labels",labels)
# print(labels.shape)
no_of_layers = 2
no_of_iterations = 200

layer_dims = [784,256,10]

data = data_standardization(data)
# print("data",data[:,0])
# print(data.shape)
# print(np.mean(data))
costs,acuracy,parameters = model(data,mini_batch_size,layer_dims,no_of_layers,learning_rate,labels,no_of_iterations)
# print(costs)
x_axis = np.arange(0,((data.shape[1]//mini_batch_size)*no_of_iterations)//100,1)
plt.plot(x_axis,costs)
plt.title("costs")
plt.show()

plt.plot(x_axis,acuracy)
plt.title("acuracy")
plt.show()
def testing(parameters,test_data,test_labels):
    no_of_layers = len(parameters)//2
    cache,y_preds = forward_propagation(test_data,parameters,no_of_layers)
    cost = cost_function(y_preds,test_labels)
    acc = accuracy(y_preds,test_labels)
    return cost,acc

# directory1 = "C:\\Users\\hp\\Downloads\\digit-recognizer\\test.csv"
# data_mnist1 = np.genfromtxt(directory1,delimiter = ",",skip_header = 1,dtype= int)
# test_data = data_mnist1.T
# test_data = data_standardization(test_data)
# print(test_data.shape)
# test_labels_before_enc = data_mnist1[:,0]
# test_labels = label_encoding(test_labels_before_enc,10)
#
# cost = testing(parameters,test_data,test_labels)
# print(cost)


test_data = data_mnist[30000:,1:].T
test_data = data_standardization(test_data)
# print(test_data.shape)
test_labels_before_enc = data_mnist[30000:,0]
test_labels = label_encoding(test_labels_before_enc,10)

cost,acc = testing(parameters,test_data,test_labels)
print("accuracy in test= ", acuracy[-1])
print("accuracy in test= ", acc)