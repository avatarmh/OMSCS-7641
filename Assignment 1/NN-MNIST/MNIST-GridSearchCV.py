
# Keras Gridsearch to optimize batch_size and epochs


# Use scikit-learn to grid search the batch size and epochs
import time
import numpy as np
import keras
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import mnist
import matplotlib.pyplot as plt

#get_ipython().magic(u'matplotlib inline')

# import theano
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD, Adam
# from keras.callbacks import EarlyStopping

input_dim=None

# Function to create model, required for KerasClassifer
def create_model():
     """
     @ params
     input_dim = integer representing the number of features, here = X_train[0].shape[0] = 784 or use
     input_shape = X_train[0].shape, which accepts a tuple, e.g., (784,)
     # nodes in hidden layer 1 => # params = input_dim * nb_dense1 + input_dim = 784^2 + 784 = 615,44

     Define and compile model in two parts using Keras
     """
     print "In create_model "

     # 1. Define the network model
     model = Sequential()  # an object of Sequential (one of 2 model options in Keras)
     model.add(Dense(output_dim=784, init='normal', input_dim=input_dim, activation='relu'))
     model.add(Dense(output_dim=10, init='normal', activation='softmax'))  # knows input = num. output = num out

     # 2. Compile model
     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
     model.summary()
     print "model defined and compiled "

     return model


    # ##########
    # model = Sequential() # an object of Sequential with one hidden layer
    # # input_dim accepts integer (number of input vectors)
    # # input_shape accepts a tuple
    # # y = W*x+b - Hidden layer
    # # 784 -> 32 -> 10 ; W = 32*784 = 25088 + b = 25088 + 32 = 25120
    # # 784 -> 784 -> 10 ; W = 784^2 + b = 615440
    # model.add(Dense(output_dim=784, init='normal', input_dim=input_dim, activation='relu'))
    # model.add(Dense(output_dim=10, init='normal', activation='softmax')) # knows input = num. output = num out
    # #compile model
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # #model.summary()
    # return model

def plot_samples(X_train):
    """
    Plot ad hoc mnist instances
    Brownlee, J. (2016). Handwritten Digit Recognition Using Convolutional Neural Networks in Python with Keras,
    MachineLearningMaster, retreived 1/30/2017
    from http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
    """

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
    #print X_train.shape, y_train.shape
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

    #print(X_train.shape)
    #print(X_test.shape)
    #print type(X_train), type(y_train)
    
    if n_samples is None:
        n_samples = X_train.shape[0]
    X_train = X_train[:n_samples]
    y_train = y_train[:n_samples]
    
    #print X_train.shape # verify sample
    #print X_train[0].shape[0] # for input_dim
    #print X_train[0].shape # for input_shape
    
    # one-hot array (somewhat expensive operation after slice)
    Y_train = keras.utils.np_utils.to_categorical(y_train, np.unique(y_train).shape[0])
    Y_test = keras.utils.np_utils.to_categorical(y_test, np.unique(y_test).shape[0])

    #print(Y_train.shape)
    #print(Y_test.shape)

    return X_train, Y_train, X_test, Y_test

#plt.bar(range(10), np.bincount(y_train[:])

def optimize_hyperparameters():

    """

    Source for much of the Code: Brownlee, Jason. (2017). "How to Grid Search Hyperparameters for Deep Learning Models
    in Python With Keras, Deep Learning - Machine Learning Mastery, August 9, 2016, accessed on 1/22/2017 from
    http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

    """
    global input_dim
    start_time = time.time()

    print "After global - about to start prep_data"

    X_train,Y_train,X_test,Y_test = prep_data(get_plot=False, n_samples=5000)
    input_dim = X_train[0].shape[0]
   
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    print "After pre_data - before meta_model def "

    # create model
    meta_model = KerasClassifier(build_fn=create_model, verbose=False)
    # define the grid search parameters
    #batch_size = [32,64,128,256]
    batch_size = [64]
    epochs = [20]
    # batch_size = [64, 128, 256]
    # epochs = [10, 20]
    #print "About to do gridsearch"
    # cv = {None defaults to 3; int = k folds}
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)

    print "Before GridSearch "

    grid = GridSearchCV(cv=None, estimator=meta_model, param_grid=param_grid, n_jobs=-1, return_train_score=True,
                        verbose=True)

    print "After GridSearch"

    grid_result = grid.fit(X_train, Y_train)

    print "grid_result.cv_results_ = ", grid_result.cv_results_

    # summarize results
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    mean_fit_times = grid_result.cv_results_['mean_fit_time']
    std_fit_times = grid_result.cv_results_['std_fit_time']
    mean_score_times = grid_result.cv_results_['mean_score_time']
    std_score_times = grid_result.cv_results_['std_score_time']

    print "mean_test_scores = ", means
    print "std_test_scores = ", stds
    print "mean_fit_times = ", mean_fit_times
    print "std_fit_times = ", std_fit_times
    print "mean_score_times = ", mean_score_times
    print " std_score_times = ", std_score_times


    ##TODO Figure out whether and how to use measures for mean/std _fit_time and _score_time values
    params = grid_result.cv_results_['params']
    with open("MNIST-GSCV-run128.txt", "w") as f:
        f.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for mean, stdev, param in zip(means, stds, params):
            f.write("%f (%f) with: %r\n" % (mean, stdev, param))
            print("%f (%f) with: %r" % (mean, stdev, param))
        f.write("--- %s seconds ---" % (time.time() - start_time))
        print("--- %s seconds ---" % (time.time() - start_time))


# if __name__ == "__main__":
#     run_experiment()

optimize_hyperparameters()

