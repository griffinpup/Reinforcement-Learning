import numpy as np

def non_lin(x, deriv = False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0], [1], [1], [0]])

np.random.seed(1)

syn0 = 2 * np.random.random((3, 32)) - 1
syn1 = 2 * np.random.random((32, 1)) - 1

for j in range(60000):

    l0 = x
    l1 = non_lin(np.dot(l0,syn0))
    l2 = non_lin(np.dot(l1,syn1))

    l2_error = y - l2

    l2_delta = l2_error * non_lin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * non_lin(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))
        #print(l1_error)