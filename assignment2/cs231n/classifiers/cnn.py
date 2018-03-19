import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C,H,W=input_dim
    self.params['W1']=weight_scale*np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1']=np.zeros((num_filters,))
    M=int(H/2*W/2*num_filters)
    self.params['W2']=weight_scale*np.random.randn(M, hidden_dim)
    self.params['b2']=np.zeros((hidden_dim,))
    self.params['W3']=weight_scale*np.random.randn(hidden_dim, num_classes)
    self.params['b3']=np.zeros((num_classes,))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out1, cache1=conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out2, cache2=affine_relu_forward(out1, W2, b2)
    scores, cache3=affine_forward(out2, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dx=softmax_loss(scores, y)
    loss+=0.5*self.reg*np.sum(W1**2)
    loss+=0.5*self.reg*np.sum(W2**2)
    loss+=0.5*self.reg*np.sum(W3**2)
    dx, grads['W3'], grads['b3']=affine_backward(dx, cache3)
    grads['W3']+=self.reg*W3
    dx, grads['W2'], grads['b2']=affine_relu_backward(dx, cache2)
    grads['W2']+=self.reg*W2
    dx, grads['W1'], grads['b1']=conv_relu_pool_backward(dx, cache1)
    grads['W1']+=self.reg*W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
class ConvNet(object):

  
  def __init__(self, input_dim=(3,32,32), num_filters_list=[80, 60],
             filter_size_list=[5, 3], hidden_dims=[400, 400, 300], num_classes=10,
              weight_scale=1e-3, use_batchnorm=False, reg=0.0, dtype=np.float32):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    
    self.use_batchnorm = use_batchnorm
    self.reg = reg
    self.num_convlayers = len(num_filters_list)
    self.num_affinelayers = len(hidden_dims)
    self.num_layers=len(num_filters_list)+len(hidden_dims)+1
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    i=1
    index=None
    weight=None
    bias=None
    gamma=None
    beta=None
    C,H,W=input_dim
    
    M=H*W*num_filters_list[-1]
    
    while i<= self.num_layers:
        index=str(i)
        weight="W"+index
        bias="b"+index
        gamma='gamma'+index
        beta='beta'+index
        if i==self.num_layers:
            self.params[weight]=weight_scale*np.random.randn(hidden_dims[-1], num_classes)
            self.params[bias]=np.zeros(num_classes)
        elif i==1:
            self.params[weight]=weight_scale*np.random.randn(num_filters_list[i-1], C, filter_size_list[i-1], filter_size_list[i-1])
            self.params[bias]=np.zeros((num_filters_list[i-1]))
            M=M/4
            if self.use_batchnorm:
                self.params[gamma]=np.ones((num_filters_list[0],))
                self.params[beta]=np.zeros((num_filters_list[0],))
        elif i<=self.num_convlayers:
            M=M/4
            self.params[weight]=weight_scale*np.random.randn(num_filters_list[i-1], num_filters_list[i-2], filter_size_list[i-1], filter_size_list[i-1])
            self.params[bias]=np.zeros((num_filters_list[i-1]))
            if self.use_batchnorm:
                self.params[gamma]=np.ones(num_filters_list[i-1])
                self.params[beta]=np.zeros(num_filters_list[i-1])
        elif i==self.num_convlayers+1:
            M=int(M)
            self.params[weight]=weight_scale*np.random.randn(M, hidden_dims[0])
            self.params[bias]=np.zeros(hidden_dims[0])
            if self.use_batchnorm:
                self.params[gamma]=np.ones(hidden_dims[0])
                self.params[beta]=np.zeros(hidden_dims[0])
        else:
            k=i-self.num_convlayers-1
            self.params[weight]=weight_scale*np.random.randn(hidden_dims[k-1], hidden_dims[k])
            self.params[bias]=np.zeros(hidden_dims[k])
            if self.use_batchnorm:
                self.params[gamma]=np.ones(hidden_dims[k])
                self.params[beta]=np.zeros(hidden_dims[k])
        i=i+1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for item in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
        self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing. 
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
                
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    i=1
    index=None
    weight=None
    bias=None
    gamma=None
    beta=None
    
    d_conv_cache={}
    d_bn_cache={}
    d_relu_cache={}
    d_pool_cache={}
    
    d_affine_cache={}
    
    while i<=self.num_layers:
        index=str(i)
        weight="W"+index
        bias="b"+index
        gamma='gamma'+index
        beta='beta'+index
        
        if i==self.num_layers:
            scores, d_affine_cache[index]=affine_forward(scores, self.params[weight], self.params[bias])
       
        elif i==1 :
            # pass conv_param to the forward pass for the convolutional layer
            filter_size = self.params[weight].shape[2]
            conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
            # pass pool_param to the forward pass for the max-pooling layer
            pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
            scores, d_conv_cache[index]=conv_forward_fast(X, self.params[weight], self.params[bias], conv_param)
            if self.use_batchnorm:
                scores, d_bn_cache[index]=spatial_batchnorm_forward(scores, self.params[gamma], self.params[beta], self.bn_params[i-1])
            scores, d_relu_cache[index]=relu_forward(scores)
            scores, d_pool_cache[index]=max_pool_forward_fast(scores, pool_param)
        
        elif i<=self.num_convlayers:
            # pass conv_param to the forward pass for the convolutional layer
            filter_size = self.params[weight].shape[2]
            conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
            # pass pool_param to the forward pass for the max-pooling layer
            pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
            scores, d_conv_cache[index]=conv_forward_fast(scores, self.params[weight], self.params[bias], conv_param)
            if self.use_batchnorm:
                scores, d_bn_cache[index]=spatial_batchnorm_forward(scores, self.params[gamma], self.params[beta], self.bn_params[i-1])
            scores, d_relu_cache[index]=relu_forward(scores)
            scores, d_pool_cache[index]=max_pool_forward_fast(scores, pool_param)
        
        else:
            scores, d_affine_cache[index]=affine_forward(scores, self.params[weight], self.params[bias])
            if self.use_batchnorm:
                scores, d_bn_cache[index]=batchnorm_forward(scores, self.params[gamma], self.params[beta], self.bn_params[i-1])
            scores, d_relu_cache[index]=relu_forward(scores)
        i=i+1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dx=softmax_loss(scores, y)
    j=self.num_layers
    while j>0:
        index=str(j)
        weight='W'+index
        bias='b'+index
        gamma='gamma'+index
        beta='beta'+index
        loss+=0.5*self.reg*np.sum(self.params[weight]*self.params[weight])
        if j==self.num_layers:
            dx, dw, db=affine_backward(dx, d_affine_cache[index])
            grads[weight]=dw+self.reg*self.params[weight]
            grads[bias]=db
        elif j>=self.num_convlayers+1:
            dx=relu_backward(dx, d_relu_cache[index])
            if self.use_batchnorm:
                dx, dgamma, dbeta=batchnorm_backward(dx, d_bn_cache[index])
            dx, dw, db=affine_backward(dx, d_affine_cache[index])
            grads[weight]=dw+self.reg*self.params[weight]
            grads[bias]=db
            if self.use_batchnorm:
                grads[gamma]=dgamma
                grads[beta]=dbeta
        else:
            dx=max_pool_backward_fast(dx, d_pool_cache[index])
            dx=relu_backward(dx, d_relu_cache[index])
            if self.use_batchnorm:
                dx, dgamma, dbeta=spatial_batchnorm_backward(dx, d_bn_cache[index])
            dx, dw, db = conv_backward_fast(dx, d_conv_cache[index])
            grads[weight]=dw+self.reg*self.params[weight]
            grads[bias]=db
            if self.use_batchnorm:
                grads[gamma]=dgamma
                grads[beta]=dbeta
        j=j-1
            
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
