import tflearn

def getModel(input_length, output_length):
    # Build neural network
    net = tflearn.input_data(shape=[None, input_length])
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, output_length, activation='softmax')
    net = tflearn.regression(net)

    return net
