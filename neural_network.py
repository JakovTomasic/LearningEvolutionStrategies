import numpy as np

def sigmoid(x):
    sig = 1.0 / (1.0 + np.exp(-x))
    sig = np.minimum(sig, 0.9999)
    sig = np.maximum(sig, 0.0001)
    return sig

class NeuralNetwork:

    def __init__(self, layers: list[int], process_out_action):
        initiali_w_list = []
        num_weights: list[int] = []

        previous_layer = layers[0]
        for layer in layers[1:]:
            intial_layer_w = np.random.randn(layer, previous_layer) / np.sqrt(previous_layer)
            initiali_w_list.append(intial_layer_w)
            num_weights.append(len(intial_layer_w.flatten()))
            previous_layer = layer

        self.initiali_w_list = initiali_w_list
        self.num_weights = num_weights
        self.layers_count = len(layers)
        self.weights_count = sum(num_weights)
        self.process_out = process_out_action

    def reshape_parameters(self, w):
        w_list = []
        next_index_to_take = 0
        for i in range(self.layers_count - 1):
            right_margin = next_index_to_take + self.num_weights[i]
            as_list = w[next_index_to_take:right_margin]
            w_list.append(as_list.reshape(self.initiali_w_list[i].shape))
            next_index_to_take = right_margin
        return w_list

    #forward propagation
    def predict(self, s, w_list):
        out = np.dot(w_list[0], s) #input to hidden layer
        for w in w_list[1:]:
            out[out<0]=0 #relu
            out = np.dot(w, out) #hidden layer to output
        return self.process_out(out)