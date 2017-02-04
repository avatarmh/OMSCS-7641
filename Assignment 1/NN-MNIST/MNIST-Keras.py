"""
    NN Model powered by Keras - applied to MNIST database
"""

import theano
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
#from sklearn.model_selection import GridSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import mnist
import time

#n_samples =  5000

"""
Create Keras model
    input_dim = X_train[0].shape[0], accepts integer (number of input vectors)
    input_shape = X_train[0].shape
    input_shape accepts a tuple
    y = W*x+b, number of nodes per hidden layer, x
    E.g., for network with 784 input nodes and 32 nodes in 1 hidden layer
    784 -> 32 -> 10 ; W = 32*784 = 25088 + b = 25088 + 32 = 25120
    784 -> 784 -> 10 ; W = 784^2 + b = 615440
"""

def graph(history, param='loss'):
    plt.plot(history.history['val_'+param])
    plt.plot(history.history[param])
    plt.title('model '+param)
    plt.ylabel(param)
    plt.xlabel('epoch')
    plt.legend(['test', 'train'], loc='lower right')
    plt.show()

def plot_data_save(df,  title="Daily Portfolio Values", xlabel="Date", ylabel="Price", filename = "comparison_optimal"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12, grid=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # plt.show(block=False)
    # plt.interactive(False)
    # plt.show()
    # plt.savefig(filename +'.png')

    # To remember other options
    # red dashes, blue squares and green triangles
    # plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')"""def plot_data_save(df, title="Daily Portfolio Values", xlabel="Date", ylabel="Price", filename = "comparison_optimal"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12, grid=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # plt.show(block=False)
    # plt.interactive(False)
    # plt.show()
    # plt.savefig(filename +'.png')

    # To remember other options
    # red dashes, blue squares and green triangles
    # plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')"""


    # categorical_crossentropy (aka multiclass logloss) objective requires labels/outputs to be binary arrays of
    # shape(nb_samples, nb_classes).

def create_model(input_dim, nb_classes):
    """
    @ params
    input_dim = integer representing the number of features, here = X_train[0].shape[0] = 784 or use
    input_shape = X_train[0].shape, which accepts a tuple, e.g., (784,)
    # nodes in hidden layer 1 => # params = input_dim * nb_dense1 + input_dim = 784^2 + 784 = 615,440

    Define and compile model in two parts using Keras
    """
    # 1. Define the network model
    model = Sequential() # an object of Sequential (one of 2 model options in Keras)
    model.add(Dense(output_dim=784, init='normal', input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim=nb_classes, init='normal', activation='softmax')) # knows input = num. output = num out

    # 2. Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def plot_samples(X_train):
    # Plot ad hoc mnist instances for report - use once

    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
    # show/save the plot
    plt.savefig('digits.png')


# prepare data
def prep_data(get_plot=False, n_samples=None):
    # X_train = raw loaded features of training set (28x28 array of pixels) [0,255] with total sample 60,000
    # y_train = raw loaded target values of feature set with values = [0,9] with total sample 60,000
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print "n_samples = ", n_samples
    if get_plot:
        plot_samples(X_train, y_train)
    # print X_train.shape, y_train.shape
    max_X = X_train.max()
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(n_train, n_features)
    X_test = X_test.reshape(n_test, n_features)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= max_X
    X_test /= max_X

    # print(X_train.shape)
    # print(X_test.shape)
    # print type(X_train), type(y_train)

    if n_samples is None:
        n_samples = X_train.shape[0]
    X_train = X_train[:n_samples]
    y_train = y_train[:n_samples]

    # print X_train.shape # verify sample
    # print X_train[0].shape[0] # for input_dim
    # print X_train[0].shape # for input_shape

    # one-hot array (somewhat expensive operation after slice)
    Y_train = keras.utils.np_utils.to_categorical(y_train, np.unique(y_train).shape[0])
    Y_test = keras.utils.np_utils.to_categorical(y_test, np.unique(y_test).shape[0])

    # print(Y_train.shape)
    # print(Y_test.shape)

    # bar chart of frequencies of class values for testing
    # plt.bar(range(10), np.bincount(y_train[:])

    return X_train, Y_train, X_test, Y_test

def run_experiment(batch_size, nb_epoch, train_set_inc=5000, nb_train_set_incs=2):
    """

    Source for much of the Code: Brownlee, Jason. (2017). "How to Grid Search Hyperparameters for Deep Learning Models
    in Python With Keras, Deep Learning - Machine Learning Mastery, August 9, 2016, accessed on 1/22/2017 from
    http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

    """

###############################
    # 1. Load and prepare the dataset
    X_train, Y_train, X_test, Y_test = prep_data(get_plot=False)

    input_dim = X_train[0].shape[0]
    nb_classes = Y_train.shape[1]

    print "input_dim = ", input_dim, " nb_classes = ", nb_classes

    # 2. Define the network and compile model
    model = create_model(input_dim, nb_classes)

    # Gather performance metrics for each of the training set size runs as dictionary
    from collections import OrderedDict
    # train_sets = OrderedDict()
    # #train_sets = {}
    train_sets = []

    max_train_sets = train_set_inc * (nb_train_set_incs + 1)

    print "train_set_inc = {} nb_train_set_incs = {}", train_set_inc, nb_train_set_incs

    print "sizes = ", range(train_set_inc, max_train_sets, train_set_inc)
    #print " sizes = ", range(5000, 15000, 5000)

    for size in range(train_set_inc, max_train_sets, train_set_inc):
    #for size in range(5000, 15000, 5000):

        # Set up to calculate training time for each training set size
        start_time = time.time()
        # Create each test subset of the training and test sets

        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # 3. Fit the model while monitoring test data performance
        history = model.fit(x=X_train[:size], y=Y_train[:size],
                            nb_epoch=nb_epoch,
                            batch_size=batch_size,
                            validation_data=(X_test, Y_test),
                            #callbacks=[EarlyStopping()],
                            verbose=1)

        # 4. evaluate the network
        loss, accuracy = model.evaluate(X_train, Y_train)
        print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))
        #
        # 5. make predictions
        # probabilities = model.predict(X_test)
        # predictions = [float(round(x)) for x in probabilities]
        # accuracy = numpy.mean(predictions == Y)
        # print("Prediction Accuracy: %.2f%%" % (accuracy * 100))

        # 5. make predictions
        probabilities = model.predict(X_test)
        predictions = np.argmax(probabilities, axis=1)
        accuracy = float(np.sum(predictions == y_test))
        accuracy = accuracy / float(y_test.shape[0])
        print("Prediction Accuracy: %.2f%%" % (accuracy * 100))


        #scores = model.evaluate(X_test, Y_test, verbose=0)

        print scores
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

        # Pull accuracy for training and test runs and note they are at 2nd to last value due to early stop option
        acc = history.history['acc'][-2]
        val_acc = history.history['val_acc'][-2]

        train_sets.append([size, acc, val_acc, (time.time() - start_time) ])

        print acc
        print val_acc
    cols = "size acc val_acc elapsed".split()
    df = pd.DataFrame(train_sets,columns=cols)

    filepath = "MNIST-run-" + str(batch_size) + "-" + str(nb_epoch) + ".csv"

    df.to_csv(filepath, index=False)

#     for size, data in train_sets.items():
#         print(data) # (acc, val_acc, elapsedTime)
#
#     filepath = "MNIST-run128-20.csv"
#     with open(filepath, "w") as f:
#         for size, data in train_sets.items():
# #            linedata = [size] + list(data)
#             linedata = list(data)
#             f.write(",".join(str(item) for item in linedata) + "\n")

    #results = pd.read_csv(filepath)
    #print results


    # print 'training set size  training accuracy  test set accuracy   time complexity'
    # from ast import literal_eval
    # with open(filepath) as f:
    #     lines = f.read().splitlines()
    #     data = []
    #     for line in lines:
    #         data.append(list(literal_eval(item) for item in line.split(",")))
    #     print 'data_type = ', type(data), ' length data = ', len(data)
    #     print 'Batch_size = ', batch_size, ' Number epochs = ', nb_epoch
    #     print data


run_experiment(batch_size=128, nb_epoch=32, train_set_inc=5000, nb_train_set_incs=3)


##########################



# print(len(test_x), test_y)
#
# test_pd = model.predict(np.array([test_x]))
# print(test_pd.shape)
#
# plt.plot(test_pd[0]);
#
#
# max(0, x), tanh, Convolutional neural nets(cnn)
# np.array([X_train1[0]]).shape
# get_ipython().magic(u'pinfo model.predict_classes')
#
# model.predict_classes(np.array([X_train1[0]]), batch_size=1)
#
# y_train1[0]



