from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE      = 0.1   # The learning rate.
    MINIBATCH_SIZE     = 1000  # Minibatch size (only for minibatch gradient descent)
    RATIO              = 0.9   # The split ratio, i.e. what part of training data remains in the training set
    PATIENCE           = 5     # Max number of validation set epochs where loss can increase

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')


        if theta:
            # The model exists
            self.FEATURES = len(theta)
            self.theta = theta
        elif x and y:
            # The model should be trained. First split the data into a training set and
            # a validation (development) set.
            x_tr, y_tr, x_val, y_val = self.train_val_split(np.array(x), np.array(y), ratio=self.RATIO)

            # Number of training datapoints.
            self.TRAINING_DATAPOINTS = len(x_tr)

            # Number of validation datapoints.
            self.VALIDATION_DATAPOINTS = len(x_val)

            # Number of features.
            self.FEATURES = len(x_tr[0]) + 1

            # Encoding of the training data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.TRAINING_DATAPOINTS, 1)), x_tr), axis=1)

            # Correct labels for the training datapoints.
            self.y = y_tr

            # Encoding of the validation data points (as a DATAPOINTS x FEATURES size array).
            self.x_val = np.concatenate((np.ones((self.VALIDATION_DATAPOINTS, 1)), x_val), axis=1)

            # Correct labels for the validation datapoints.
            self.y_val = y_val

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)


    # ----------------------------------------------------------------------

    def train_val_split(self, x, y, ratio):
        """
        Performs a split of the given training data into a training set and a validation set

        :param      x:      The input features of the training data
        :param      y:      The correct labels of the training data
        :param      ratio:  The split ratio, i.e. what part of training data remains in the training set
                            e.g. ratio = 0.8 means that 80% of the training data will be used as training set
                                       and the remaining 20% will be used a validation set
        """

        # For explanation on use of train_test_split, see following links:
        # https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros
        # https://stackoverflow.com/questions/42191717/python-random-state-in-splitting-dataset/42197534
        #x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=1-ratio, random_state=42)
        ###
        ### THIS FUNCTION MUST BE RE-DONE, THIS ^ IS NOT ALLOWED
        ###

        ## BELOW IS ALLOWED, USE IT FOR PRESENTATION OF THE SMALLER DATA SETS
        train_size = ratio * len(x)

        x_tr = np.array([[0,0]]) # Extend by appropriate number of zeros depending on number of features we have.
        y_tr = np.array([])

        while len(x_tr) <= train_size:
            index = random.randrange(0, len(x))

            x_tr = np.vstack([x_tr, x[index]])
            x = np.delete(x, index, 0)

            y_tr = np.append(y_tr, y[index])
            y = np.delete(y, index)

        x_val = x
        y_val = y
        x_tr = np.delete(x_tr, 0, 0)

        return x_tr, y_tr, x_val, y_val


    def loss(self, x, y):
        """
        Computes the loss function given the input features x and labels y

        :param      x:    The input features
        :param      y:    The correct labels
        """

        # Dot product of THETA * X
        theta_mult_x = self.theta.dot(x.T)

        # Sigmoid each value in the result
        sigmoid_v = np.vectorize(self.sigmoid)
        theta_mult_x = sigmoid_v(theta_mult_x)

        # Get the loss function for case label = 1
        val_label_1 = np.log(theta_mult_x)
        val_label_1 *= -y

        # Get the loss function for case label = 0
        val_label_0 = np.log(1-theta_mult_x)
        val_label_0 *= (1-y)

        # Subtract
        sigma = val_label_1 - val_label_0

        # Sum all terms to get the total loss
        loss = np.sum(sigma) / len(y)

        # Return average loss
        return loss


    def compute_validation_loss(self, val_loss, inc_val_loss):
        """
        Calculates the validation loss, and tracks the number of consecutive iterations
        the validation loss increases, using the `inc_val_loss` variable.

        :param      val_loss:      The current value of the validation loss
        :param      inc_val_loss:  The number of iterations of constant increase of validation loss
        """

        new_val_loss = self.loss(self.x_val, self.y_val)

        if new_val_loss > val_loss:
            # If validation loss increases
            inc_val_loss += 1
        else:
            inc_val_loss = 0

        val_loss = new_val_loss

        return val_loss, inc_val_loss


    def sigmoid(self, z):
        """
        The logistic function at the point z.
        """
        return 1.0 / ( 1 + math.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)

        :param      label:      The label
        :param      datapoint:  The datapoint itself (NOT the ID)
        """

        return self.sigmoid(self.theta.dot(datapoint))


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """

        # Compute theta * x (for clarification, * is DOT PRODUCT)
        sigma = self.theta.dot(self.x.T)

        # Sigmoid the result of theta * x
        sigmoid_v = np.vectorize(self.sigmoid) # So we can use our sigmoid()-method on vectors
        sigma = sigmoid_v(sigma) # Sigmoid all values

        # Subtract all values by corresponding label in the labels vector (self.y)
        sigma -= self.y

        # Now we must multiply by Xk^i
        sigma = self.x.T.dot(sigma)

        for k in range(self.FEATURES):
            self.gradient[k] = sigma[k] / self.TRAINING_DATAPOINTS


    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for mini-batch gradient descent).

        :param      minibatch:  A list of IDs for the datapoints in the minibatch
        """

        # Compile the minibatch into a numpy matrix
        x_mb = np.array(self.x[minibatch[0]]) # Creating a vector out of minibatch
        y_mb = np.array(self.y[minibatch[0]]) # Getting first element in corresponding labels vector
        for i in range(1, len(minibatch)):
            new_row_x = np.array(self.x[minibatch[i]]) # Might not need this
            new_row_y = np.array(self.y[minibatch[i]]) # Might not need this
            x_mb = np.vstack([x_mb, new_row_x]) # Adding values to vector to form a matrix
            y_mb = np.append(y_mb, new_row_y) # Adding elements to labels vector

        # Code below is exactly the same as for batch gradient descent
        sigma = self.theta.dot(x_mb.T)

        sigmoid_v = np.vectorize(self.sigmoid)
        sigma = sigmoid_v(sigma)

        sigma -= y_mb

        sigma = x_mb.T.dot(sigma)

        for k in range(self.FEATURES):
            self.gradient[k] = sigma[k] / len(minibatch)


    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).

        :param      datapoint:  The ID of the datapoint in which the gradient is to be computed
        """

        ## In this code, sigma is not actually a sigma sum; it's just one term in the sigma sum at a random index i.

        # Dot product theta * x^i
        sigma = self.theta.dot(self.x[datapoint]) # theta * x^i

        # Sigmoid the resulting value
        sigma = self.sigmoid(sigma)

        # Subtract y^i
        sigma -= self.y[datapoint]

        # Multiply by datapoint (Xk^i)
        sigma *= self.x[datapoint]

        for k in range(self.FEATURES):
            self.gradient[k] = sigma[k]


    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        val_loss = 0
        val_inc_loss = 0
        compute_val_loss = 0
        it = 0 # Track iterations
        while val_inc_loss < 5:
            it += 1
            compute_val_loss += 1

            i = random.randrange(0, self.TRAINING_DATAPOINTS) # Get random i
            prev_gradient = np.array(self.gradient[:])
            self.compute_gradient(i)

            for k in range(self.FEATURES):
                self.theta[k] -= BinaryLogisticRegression.LEARNING_RATE * self.gradient[k]

            if compute_val_loss == 1000:
                compute_val_loss = 0
                val_loss, val_inc_loss = self.compute_validation_loss(val_loss, val_inc_loss)
                regular = self.loss(self.x, self.y)
                print(regular)
                print(val_loss)
                self.update_plot(regular, val_loss)

                print("total iterations")
                print(it)

            if np.sum(np.square(self.gradient)) < 0.000001:
                print("sum of squares of gradient smaller than tolerance, exiting at iteration", it)
                break
            #if np.allclose(self.gradient, prev_gradient, rtol=1e-04, atol=1e-07):
            #    print("gradient difference smaller than tolerance, exiting at iteration", it)
            #    break


    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        val_loss = 0
        val_inc_loss = 0
        plot = 0
        it = 0
        while val_inc_loss < 5:
            it += 1
            plot += 1

            # Generate 1000 datapoints, then send through method below
            datapoints = []
            for i in range(1000):
                random_datapoint = random.randrange(0, self.TRAINING_DATAPOINTS)
                datapoints.append(random_datapoint)

            # To calculate difference in gradient
            prev_gradient = np.array(self.gradient[:])
            self.compute_gradient_minibatch(datapoints)

            for k in range(self.FEATURES):
                self.theta[k] -= BinaryLogisticRegression.LEARNING_RATE * self.gradient[k]

            val_loss, val_inc_loss = self.compute_validation_loss(val_loss, val_inc_loss)

            if plot == 50:
                plot = 0
                regular = self.loss(self.x, self.y)
                print(regular)
                print(val_loss)
                self.update_plot(regular, val_loss)

                print("iterations")
                print(it)

            if np.sum(np.square(self.gradient)) < 0.000001:
                print("sum of squares of gradient smaller than tolerance, exiting at iteration", it)
                break
            # If gradient difference close enough by a given tolerance, break
            #if np.allclose(self.gradient, prev_gradient, rtol=1e-04, atol=1e-07):
            #    print("gradient difference smaller than tolerance, exiting at iteration", it)
            #    break


    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)

        val_loss = 0
        val_inc_loss = 0
        plot = 0
        it = 0 # Track iterations
        while val_inc_loss < 5:
            it += 1
            plot += 1

            prev_gradient = np.array(self.gradient[:])
            self.compute_gradient_for_all()

            for k in range(self.FEATURES):
                self.theta[k] -= BinaryLogisticRegression.LEARNING_RATE * self.gradient[k]

            val_loss, val_inc_loss = self.compute_validation_loss(val_loss, val_inc_loss) # Uncomment this

            if plot == 50:
                plot = 0
                regular = self.loss(self.x, self.y)
                print(regular)
                print(val_loss)
                self.update_plot(regular, val_loss)

                print("iterations")
                print(it)

            if np.sum(np.square(self.gradient)) < 0.000001:
                print("sum of squares of gradient smaller than tolerance, exiting at iteration", it)
                break
            #if np.allclose(self.gradient, prev_gradient, rtol=1e-04, atol=1e-07):
            #    print("gradient difference smaller than tolerance, exiting at iteration", it)
            #    break


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints

        :param      test_data:    The input features for the test set
        :param      test_labels:  The correct labels for the test set
        """
        print('Model parameters:')

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        num_test_datapoints = len(test_data)

        x_test = np.concatenate((np.ones((num_test_datapoints, 1)), np.array(test_data)), axis=1)
        y_test = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(num_test_datapoints):
            # Why is first parameter (label) always ONE?
            prob = self.conditional_prob(1, x_test[d])
            predicted = 1 if prob > .5 else 0
            confusion[predicted][y_test[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        Initializes the plot.

        :param      num_axes:  The number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        colors = [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=colors[i], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()
