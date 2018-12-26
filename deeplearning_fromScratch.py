# -*- coding: utf-8 -*-

# This code produces are neural network based on the example given by Joel Grus
# in his book "Data Science from Scratch".

import auxiliary
import math
import random

def step_function(x):
    return 1 if x>=0 else 0

def perceptron_output(weights, bias,x):
    calculation = auxiliary.dotdot(weights, x) + bias
    return step_function(calculation)

def neuron_output(weights, x):
   return sigmoid(auxiliary.dot(weights, x))
   

def sigmoid(t):
    return 1/(1+math.exp(-t))

def feed_forward(neural_network, input_vector):
    outputs=[]
    
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector = output
    
    return outputs


def backpropagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network,input_vector)
    output_deltas = [output * (1-output)*(output-target) for output, target in zip(outputs, targets)]
    
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i]*hidden_output
            hidden_deltas = [hidden_output * (1-hidden_output) * auxiliary.dot(output_deltas,
                             [n[i] for n in output_layer]) for i, hidden_output in enumerate(hidden_outputs)]
    
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector +[1]):
            hidden_neuron[j] -= hidden_deltas[i]*input
            
            
### Application 1

init_digit  = [0,0,0,0,0,
               0,0,0,0,0,
               1,0,0,0,0,
               0,0,0,0,0,
               0,0,0,0,0]

zero_digit  = [1,1,1,1,1,
               1,0,0,0,1,
               1,0,0,0,1,
               1,0,0,0,1,
               1,1,1,1,1]

one_digit  =  [0,0,1,0,0,
               0,1,1,0,0,
               1,0,1,0,0,
               0,0,1,0,0,
               0,0,1,0,0]

two_digit  = [0,1,1,1,0,
               1,0,0,0,1,
               0,0,0,1,0,
               0,0,1,0,0,
               0,1,1,1,1]

three_digit  = [1,1,1,1,1,
               0,0,0,0,1,
               1,1,1,1,1,
               0,0,0,0,1,
               1,1,1,1,1]

four_digit  = [1,0,1,0,0,
               1,0,1,0,0,
               1,1,1,1,1,
               0,0,1,0,0,
               0,0,1,0,0]

five_digit  = [1,1,1,1,1,
               1,0,0,0,0,
               1,1,1,1,1,
               0,0,0,0,1,
               1,1,1,1,1]

six_digit  =  [1,1,0,0,0,
               1,0,0,0,0,
               1,1,1,1,1,
               1,0,0,0,1,
               1,1,1,1,1]

seven_digit  = [1,1,1,1,1,
               0,0,0,1,0,
               0,1,1,1,0,
               0,1,0,0,0,
               1,0,0,0,0]

eight_digit  = [1,1,1,1,1,
               1,0,0,0,1,
               1,1,1,1,1,
               1,0,0,0,1,
               1,1,1,1,1]

nine_digit  = [1,1,1,1,1,
               1,0,0,0,1,
               1,1,1,1,1,
               0,0,0,0,1,
               0,0,0,0,1]

feature_set = [zero_digit,one_digit,two_digit,three_digit,four_digit,five_digit,six_digit,seven_digit,eight_digit,nine_digit]

targets = [[1 if i==j else 0 for i in range(10)] for j in range(10)]

random.seed(0)
input_size = 25
num_hidden = 5
output_size = 10

hidden_layer = [[random.random() for _ in range(input_size+1)] for _ in range(num_hidden)]
output_layer = [[random.random() for _ in range(num_hidden+1)] for _ in range(output_size)]

network = [hidden_layer, output_layer]

def predict(input_v):
    return feed_forward(network, input_v)[-1]

for _ in range(10000):
    for input_vector, target_vector in zip(feature_set, targets):
        backpropagate(network, input_vector,target_vector)
        



### Application 2

xor_network = [
        [[20,20,-30],[20,20,-10]],
        [[-60,60,-40]]
        ]
        
for x in [0,1]:
    for y in [0,1]:
        print(x,y, feed_forward(xor_network,[x,y])[-1])