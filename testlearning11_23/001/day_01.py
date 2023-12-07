#start in 11.23 20:08
import numpy
import scipy.special
import matplotlib.pyplot


class neuralNetwork:

    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.indoes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        #link weight matrices
        self.wih = numpy.random.normal(0.0,pow(self.indoes,-0.5),(self.hnodes,self.indoes))
        self.who = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))
        #learning rate
        self.lr = learningrate
        #activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin = 2).T
        targets = numpy.array(targets_list,ndmin=2).T
        hidden_inputs = numpy.dot(self_wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)
        self.who += self.lr*numpy.dot((output_errors * final_outputs * (1.0- final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)),numpy.transpose(inputs))

        pass
    def query(self,inputs_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list,ndmin = 2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

input_nodes = 784
hidden_nodes = 300
output_nodes = 10
learning_rate = 0.15
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

data_file = open(r"C:\Users\13066\Downloads\makeyourownneuralnetwork-master\makeyourownneuralnetwork-master\mnist_dataset\mnist_train_100.csv",'r')
data_list = data_file.readlines()
data_file.close()

scorecard = []
for record in data_list:
    all_values =record.split(',')
    correct_label = int(all_values[0])
    print(correct_label,"correct label")
    inputs = (numpy.asarray(all_values[1:],dtype=float) /255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print(label,"network is answer")
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass


    pass
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)