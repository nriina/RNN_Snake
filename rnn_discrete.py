#code from https://snaildove.github.io/2018/06/02/Building+a+Recurrent+Neural+Network+-+Step+by+Step+-+v3/

import numpy as np
import matplotlib.pyplot as plt
import math
from CTRNN_supplies import mga



#theyre code looks a bit exausting i'm just gonna interpret it into my own syntax

#activation functions all work :)
def tanh(x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def relu(x):
    return max(0.0,x)
def sigmoid(x):
    return (1/(1+np.exp(-x)))

def softmax(x): #not exactly sure how this works but I think it chooses an index/probability of an neuron rather than a value (1 vaue == 100%)
	return np.exp(x) / np.exp(x).sum()

##single cell network first
class rnn_unit():
    def __init__(self,input_size=1,initial_Act = 0):
        self.Input = np.zeros(input_size)
        # self.Wi = np.ones(input_size)
        self.Wi = np.random.random(input_size)
        self.Act = [initial_Act]
        self.Wa = [np.random.uniform(low=-1,high=1)]
        self.Bias = [np.random.uniform(low=-1,high=1)]
        self.time = 0

    def set_parameters(self,genes): #genes go input weight, self weight, bias
        i = 0
        # while i < len(genes):
        for j in range(0,len(self.Input)):
            self.Input[j] = genes[i]
            i += 1
        for k in range(0,len(self.Act)):
            self.Wa = genes[i]
            i +=1
        for l in range(0,len(self.Bias)):
            self.Bias = genes[i]
            i += 1

    def make_genes(self):
        genes = []
        for i in range(0,len(self.Input)):
            genes.append(self.Wi[i])
        for j in range(0,len(self.Act)):
            genes.append(self.Wa[j])
        for k in range(0,len(self.Bias)):
            genes.append(self.Bias[k])
        return genes

    def step(self): #calculate a new activation
        self.time += 1
        self.Act.append(tanh( np.dot(self.Wi,self.Input) + np.dot(self.Wa, self.Act[self.time - 1]) + self.Bias) )

    def compute_output(self,output):
        return softmax(output)
    
    
    


class rnn():
    def __init__(self,initial_Act = 0.5,size=2):
        self.Input = np.zeros(size)
        self.Wi = np.ones(size)
        self.Act = list(np.ones(size))
        for act in range(0,len(self.Act)):
            self.Act[act] = [self.Act[act] * initial_Act] #act[0] = act of first, act[1] = act of second
        self.old_act = list(np.ones(size))
        # self.Wa = np.ones(input_size) not needed cuz self weight in weight matrix
        # self.Bias = -0.2
        self.Bias = list(np.ones(size))
        self.time = 0
        # self.Weight = np.zeros((size,size))
        # self.Weight = np.random.rand(size,size)
        self.Weight = np.random.uniform(low=-1.0,high=1.0,size=(size,size))
        self.size = size

    def step(self):
        self.time +=1
        for i in range(0,len(self.Act)):
            self.old_act[i] = self.Act[i]
            self.Act[i] =  tanh(np.dot(self.Wi[i],self.Input[i]) + self.Bias[i] + np.dot(self.Weight[i],self.Act))
    
    def make_genes(self): #weight, bias, wi
        genes = []
        for i in range(self.size): #weight
            for j in range(self.size):
                genes.append(self.Weight[i][j])
        for k in range(self.size):
            genes.append(self.Bias[k])
        for l in range(self.size):
            genes.append(self.Wi[l])
        return genes
    
    def set_parameters(self,genes,WeightRange,BiasRange):
        k = 0
        for i in range(self.size):
            for j in range(self.size):
                self.Weight[i][j] = genes[k]*WeightRange
                k +=1
        for l in range(self.size):
            self.Bias[l] = genes[k]*BiasRange
            k +=1
        for m in range(self.size):
            self.Wi[m] = genes[k]*WeightRange
            k +=1

        

if __name__ == "__main__":
#    print("File one executed when ran directly")
    group = rnn(size=5)
    group_genes = group.make_genes()
    print(group_genes)

    # newb = rnn(size=5)
    # # newb.set_parameters(group_genes)
    # print(group.Weight)
    # # print(group.Act)
    # # group.step()
    # # print(group.Act)
    outputs = []
    # newb_outputs = []
    # # append_values = group.Act
    outputs.append(group.Act.copy())
    # newb_outputs.append(newb.Act.copy())
    # # print('output length',outputs)
    for i in range(10):
        group.step()
    #     newb.step()
        outputs.append(group.Act.copy())
    #     newb_outputs.append(newb.Act.copy())
    #     # print('output length',outputs)
    #     # print('group act',group.Act)
        
    #     # print('output length',outputs)
    # # print('group act',group.Act)   
    # # print('outputs',outputs[0][0])
    neuron0 = []
    neuron1 = []
    neuron2 = []
    # newb_0 = []
    # newb_1 = []
    # newb_2 = []
    for plot in range(0,len(outputs)):
        neuron0.append(outputs[plot][0])
        neuron1.append(outputs[plot][1])
        neuron2.append(outputs[plot][2])
    #     newb_0.append(newb_outputs[plot][0])
    #     newb_1.append(newb_outputs[plot][1])
    #     newb_2.append(newb_outputs[plot][2])
    #     # print(neuron0)
    #     # print(outputs[plot][0])
    plt.plot(neuron0)
    plt.plot(neuron1)
    plt.plot(neuron2)
    plt.show()

    # plt.plot(newb_0)
    # plt.plot(newb_1)
    # plt.plot(newb_2)
    # plt.show()
    #     for act in outputs[plot]:
    #         plt.plot(act,label='plot'+str(plot))
    #         # print(act)
    #     # print(outputs)
    #     plt.show()
    # plt.show()
# class rnn():
#     def __init__(self, size=1,initial_Act = 0.5):
#         self.Input = np.zeros(size)
#         self.Wi = np.ones(size)
#         self.Act = list(np.ones(size))
#         for act in range(0,len(self.Act)):
#             self.Act[act] = [self.Act[act] * initial_Act] #act[0] = act of first, act[1] = act of second
#         self.old_act = list(np.ones(size))
#         self.Wa = np.ones(size)
#         self.weights = np.random.random((size,size)) #weights from neuron to neuron
#         self.Bias = -2
#         self.time = 0

#     def step(self): #all fucked up
#         self.time +=1
#         for i in range(0,len(self.Act)):
#             current_acts = []
#             for k in range(0,len(self.Act)):
#                 current_acts.append(self.Act[k][0])
#             # self.old_act[i] = self.Act[i]
#             # print('act',self.Act)
#             # print('indiv act',self.Act[i])
#             other_act = np.sum(np.dot(self.weights[i],current_acts))
#             # print('other act',other_act)
#             self.Act[i].append((tanh( np.dot(self.Wi[i],self.Input[i]) + other_act + self.Bias) ))
            # self.Act[i].append((sigmoid( np.dot(self.Wi[i],self.Input[i]) + other_act + self.Bias) ))


    
        # pass
 
# rnns = rnn(size = 5)
# rnns.Input[0] = 0
# rnns.Input[1] = 1
# rnns.Input[2] = -1
# rnns.Input[3] = -1
# rnns.step()
# # print(rnns.Act)
# rnns.step()
# print(rnns.Act)
# rnns.step()
# rnns.step()
# rnns.step()
# rnns.step()

# print('len',len(rnns.Act[0]))
# plt.plot(rnns.Act[4]) #very non smooth activation
# plt.show()
# print(rnns.Act)
#trying some simulating

# rnn = rnn_unit(initial_Act=0.5,input_size=1)
# # rnn.Wa = -0.3
# new_genes = rnn.make_genes()
# print('new genes',new_genes)
# # # rnn.Wi = 1
# # # rnn.Input[0] = 0.2
# # for inp in range(0,len(rnn.Input)):
# #     rnn.Input[inp] = np.random.random()
# # # rnn.Bias = 0
# # outputs = []
# # time = range(0,10,1)
# # for i in time:
# #     outputs.append(rnn.Act[i])
# #     # print('simulate time',i)
# #     # print('net time',rnn.time)
# #     rnn.step()

# # plt.plot(outputs)
# # plt.show()

# rnn2 = rnn_unit(initial_Act=0.5,input_size=1)
# sep_genes = rnn2.make_genes()
# print('sep genes',sep_genes)

# rnn3 = rnn_unit(initial_Act=0.5, input_size=1)
# rnn3.set_parameters(new_genes)
# rnn4 = rnn_unit(initial_Act=0.5, input_size=1)
# rnn4.set_parameters(sep_genes)
# newoutputs = []
# outputs4 = []
# time = range(0,10,1)
# for i in time:
#     newoutputs.append(rnn3.Act[i])
#     outputs4.append(rnn4.Act[i])
#     # print('simulate time',i)
#     # print('net time',rnn.time)
#     rnn3.step()
#     rnn4.step()

# plt.plot(newoutputs,label='3')
# plt.plot(outputs4,label='4')
# plt.legend()
# plt.show()
# # print(rnn.Input)

# # # rnn.
# # # define input data
# # # inputs = [1.0, 3.0, 2.0]
# # # # calculate outputs
# # # outputs = softmax(inputs)
# # # report the probabilities
# # print(outputs)
# # report the sum of the probabilities
# print(outputs.sum())
	# e = math.exp(vector)


# input = range(-10,10,1)
# realinput = []
# for j in range(0,len(input)):
#     realinput.append(input[j] / 10)
# output = []
# for i in input:
#     output.append(tanh(i))
#     # output.append(relu(i))
#     # output.append(sigmoid(i))
# # print('output',output)
# plt.plot(realinput,output)
# plt.show()

# data = [1, 3, 2]
# # convert list of numbers to a list of probabilities
# result = softmax(data)
# # report the probabilities
# print(result)
# # report the sum of the probabilities
# print(sum(result))
