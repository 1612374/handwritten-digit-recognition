import numpy as np
import math
import scipy.special



class NeuralNetWork:
    def __init__(self, input_nodes, hid_nodes, output_nodes, learn_rate):
        self.num_input_nodes = input_nodes
        self.num_hid_nodes = hid_nodes
        self.num_output_nodes = output_nodes

        self.learn_rate = learn_rate
        
        # set weight input_hidden and weight hidden_ouput, by normal distribution
        
        self.weight_ih = np.random.normal(0.0, math.pow(self.num_hid_nodes, -0.5), (self.num_hid_nodes, self.num_input_nodes))
        self.weight_ho = np.random.normal(0.0, math.pow(self.num_output_nodes,-0.5), (self.num_output_nodes, self.num_hid_nodes))

        
        # acticvation function sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)



    def train(self, inputs_list, targets_list):
        
        inputs = np.array(inputs_list, ndmin = 2).T # transpose to vector
        targets = np.array(targets_list, ndmin = 2).T
        
        hidden_inputs = np.dot(self.weight_ih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weight_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        final_errors = targets - final_outputs
        hidden_errors = np.dot(self.weight_ho.T, final_errors)

    
        self.weight_ho += self.learn_rate*np.dot((final_errors*final_outputs*(1.0-final_outputs)), np.transpose(hidden_outputs))
        self.weight_ih += self.learn_rate*np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)), np.transpose(inputs))

        pass

    def query(self, inputs_list):
        
        inputs = np.array(inputs_list, ndmin = 2).T # transpose to vector

        hidden_inputs = np.dot(self.weight_ih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weight_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
    
        return final_outputs





