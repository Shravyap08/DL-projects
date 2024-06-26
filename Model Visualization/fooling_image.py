import torch
from torch.autograd import Variable

class FoolingImage:
    def make_fooling_image(self, X, target_y, model):
        """
        Generate a fooling image that is close to X, but that the model classifies
        as target_y.

        Inputs:
        - X: Input image; Tensor of shape (1, 3, 224, 224)
        - target_y: An integer in the range [0, 1000)
        - model: A pretrained CNN

        Returns:
        - X_fooling: An image that is close to X, but that is classifed as target_y
        by the model.
        """

        model.eval()

        # Initialize our fooling image to the input image, and wrap it in a Variable.
        X_fooling = X.clone()
        X_fooling_var = Variable(X_fooling, requires_grad=True)

        # We will fix these parameters for everyone so that there will be
        # comparable outputs

        learning_rate = 10
        max_iter = 100  # maximum number of iterations

        for it in range(max_iter):
            
            out= model(X_fooling_var)
            S = out[0,target_y]
            model.zero_grad()
            S.backward()
            g = X_fooling_var.grad.sign()
            g_n= torch.sqrt(torch.sum(g ** 2))
            g_norm= g/g_n
            dX = learning_rate*g_norm
            X_fooling_var.data += dX 
            #X_fooling_var.data = torch.clamp(X_fooling_var.data,0,1)
            X_fooling_var.grad.data.zero_()
            

            ##############################################################################
            # TODO: Generate a fooling image X_fooling that the model will classify as   #
            # the class target_y. You should perform gradient ascent on the score of the #
            # target class, stopping when the model is fooled.                           #
            # When computing an update step, first normalize the gradient:               #
            #   dX = learning_rate * g / ||g||_2                                         #
            #                                                                            #
            # Inside of this loop, write the update rule.                                #
            #                                                                            #
            # HINT:                                                                      #
            # You can print your progress (current prediction and its confidence score)  #
            # over iterations to check your gradient ascent progress.                    #
            ##############################################################################
            pass
            ##############################################################################
            #                             END OF YOUR CODE                               #
            ##############################################################################

        X_fooling = X_fooling_var.data

        return X_fooling
