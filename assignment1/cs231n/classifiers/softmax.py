import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores-=np.max(scores)
    probabilities=np.exp(scores)/np.sum(np.exp(scores))
    p=probabilities[y[i]]/np.sum(probabilities)
    i_loss=-np.log(p)
    loss+=i_loss
    for j in xrange(num_classes):
        if j==y[i]:
            constant=(1/p)*(np.exp(scores[y[i]])*(np.sum(np.exp(scores))-np.exp(scores[y[i]])))/(np.sum(np.exp(scores))**2)
            dW[:,j]-=constant*X[i]
        else:
            constant=(1/p)*(-np.exp(scores[j])*np.exp(scores[y[i]]))/(np.sum(np.exp(scores))**2)
            dW[:,j]-=constant*X[i]
  loss+=0.5*reg*np.sum(W*W)
  dW+=reg*W
  loss/=num_train
  dW/=num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  allscores=X.dot(W)
  Loss=np.ones_like(allscores)
  allscores=(allscores.T-np.max(allscores, axis=1).T).T
  exp_allscores=np.exp(allscores)
  norm=np.sum(exp_allscores, axis=1)
  Loss[np.arange(num_train), y]=exp_allscores[np.arange(num_train),y]/norm
  Loss=np.log(Loss)
  loss-=np.sum(Loss)
  loss+=0.5*reg*np.sum(W*W)
  norm=np.expand_dims(norm, axis=1)
  grad=exp_allscores/norm
  grad[np.arange(num_train),y]-=1.0
  dW=(X.T).dot(grad)+ reg*W
  loss/=num_train
  dW/=num_train
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

