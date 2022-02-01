import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data.head()

data = np.array(data)

row, cols = data.shape
np.random.shuffle(data)

dev_data = data[:1000].T

y_dev = dev_data[0]
x_dev = dev_data[1:]
x_dev = x_dev / 255

train_data = data[1000:].T

y_train = train_data[0]
x_train = train_data[1:]
x_train = x_train / 255

label_range = set(list(y_dev) + list(y_train))

x_train.shape

class numpy_nn:
    def __init__(self, X=None, Y=None, alpha=None):
        # 1st hidden layer has 10 nodes,
        # 784 pixels as input to the layer.
        # (W.XT)
        self.W1 = np.random.rand(10, 784) - 0.5
        self.B1 = np.random.rand(10, 1) - 0.5

        # 2nd layer has 10 nodes, 1st layer output nodes are 10
        self.W2 = np.random.rand(10, 10) - 0.5
        self.B2 = np.random.rand(10, 1) - 0.5

        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        self.alpha = alpha

        self.labels = Y
        self.X = X

    def forward_prop(self):
        # 1st hidden layer
        self.Z1 = self.W1.dot(self.X) + self.B1
        self.A1 = self.relu(self.Z1)

        # 2nd hidden layer
        self.Z2 = self.W2.dot(self.A1) + self.B2
        self.A2 = self.softmax(self.Z2)

    def relu(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, z):
        a = np.exp(z) / sum(np.exp(z))
        return a

    def backward_prop(self):
        m = self.labels.size
        one_hot_y = self.one_hot_encod()
        dz2 = self.A2 - one_hot_y
        dw2 = (1 / m) * dz2.dot(self.A1.T)
        db2 = (1 / m) * np.sum(dz2)

        dz1 = self.W2.T.dot(dz2) * self.deriv_relu(self.Z1)
        dw1 = (1 / m) * dz1.dot(self.X.T)
        db1 = (1 / m) * np.sum(dz1)

        # after backpropagation, update the parameters
        self.__update_params(dw1, db1, dw2, db2)

    def one_hot_encod(self):
        # each row = label encoded
        # col size = max classes available
        one_hot_shape = (self.labels.size, len(label_range))
        one_hot_y = np.zeros(one_hot_shape)
        one_hot_y[np.arange(self.labels.size), self.labels] = 1
        return one_hot_y.T

    def deriv_relu(self, Z):
        return Z > 0

    def __update_params(self, dw1, db1, dw2, db2):
        self.W1 = self.W1 - self.alpha * dw1
        self.B1 = self.B1 - self.alpha * db1

        self.W2 = self.W2 - self.alpha * dw2
        self.B2 = self.B2 - self.alpha * db2

    def get_accuracy(self):
        predictions = self._get_predictions()
        return np.sum(predictions == self.labels) / self.labels.size

    def _get_predictions(self, ):
        return np.argmax(self.A2, 0)

def gradient_descent(X, Y, iterations):
    for i in range(iterations):
        neural_net_init.forward_prop()
        neural_net_init.backward_prop()
        if i%2==0:
            print('Iterations: {}'.format(i))
            print("Accuracy: {}".format(neural_net_init.get_accuracy()))

iterations = 100
alpha = 0.1
#gloabally call the model since it will be used later for testing
neural_net_init = numpy_nn(x_train, y_train, alpha)
gradient_descent(x_train, y_train, iterations)

def trial():
    _, m = x_dev.shape
    index = np.random.randint(m +1, size=1)
    current_image = x_dev[:, int(index), None]
    label = y_dev[index]
    #to make prediction use the model which has the weights and biases saved
    neural_net_init.X = current_image
    neural_net_init.labels = label

    #only feed forward is required to check the prediction
    neural_net_init.forward_prop()
    prediction = neural_net_init._get_predictions()
    print("Predicted label: {}".format(prediction))
    print("Actual label: {}".format(label))
    if prediction == label:
        print("Our algorithm correctly predicted the value.")
    else:
        print("Our algorithm does'nt correctly predicted the value. We have to update our weights.")

    #display the image
    current_image = current_image.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

trial()
# calculate accuracy, round off to value 4