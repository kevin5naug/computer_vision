import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
        margin = scores[j] - correct_class_score + 1 # note delta = 1
        if j == y[i]:
            continue
        if margin > 0:
            loss += margin
            dW[:,j]+=X[i,:]
            dW[:,y[i]]-=X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW/=num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW+=reg*W
  

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  allscores=X.dot(W)
  correct_class_score=allscores[np.arange(num_train),y]
  correct_class_score=np.expand_dims(correct_class_score, axis=1)
  adjust=np.ones(correct_class_score.shape)
  raw_loss=allscores-correct_class_score
  raw_loss=raw_loss+adjust
  raw_loss[np.arange(num_train),y]-=1
  Bool=raw_loss>0
  loss=np.sum(raw_loss[Bool])
  loss/=num_train
  Regularization=0.5*reg*np.sum(W*W)
  loss+=Regularization
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  Bool=Bool*np.ones(raw_loss.shape)
  Bool[np.arange(num_train),y]=-(np.sum(Bool,axis=1))
  dW=(X.T).dot(Bool)/float(num_train)
  dW+=reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
