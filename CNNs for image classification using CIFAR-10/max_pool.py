"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        S=self.stride
        K=self.kernel_size
        N,C,H,W= x.shape
        H_out= int((H-K)/S)+1
        W_out= int((H-K)/S)+1
        
        out = np.zeros((N, C, H_out, W_out))

        
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h1 = h * S
                        h2 = h1 + K
                        w1 = w * S
                        w2 = w1 + K
                        x_piece=x[n,c,h1:h2,w1:w2]
                        out[n,c,h,w]=np.max(x_piece)
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        
       
        x, H_out, W_out = self.cache
        N,C,H,W= x.shape
    
        #Defining the inputs
        S=self.stride
        K= self.kernel_size
        dx=np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h1 = h * S
                        h2 = h1 + K
                        w1 = w * S
                        w2 = w1 + K
                        x_slice = x[n, c, h1:h2, w1:w2]
                        max_value = np.max(x_slice)
                        max_index = np.unravel_index(np.argmax(x_slice), x_slice.shape)
                        mask_filter=np.zeros_like(x_slice)
                        mask_filter[max_index]=1
                        dx[n,c,h1:h2,w1:w2]+=dout[n,c,h,w]*mask_filter
        self.dx = dx
        
        
                        
        
        
       
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
