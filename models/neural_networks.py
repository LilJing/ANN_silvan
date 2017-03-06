from chainer import Chain, ChainList, Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import custom_links as CL
import chainer.initializers as init
from chainer.functions.activation import sigmoid

#####
## Deep Neural Network

class DeepNeuralNetwork(ChainList):
    """
    Fully connected deep neural network consisting of a chain of layers (weight matrices)
    with a fixed number of nhidden units
    """

    def __init__(self, ninput, nhidden, noutput, nlayer=2, actfun=F.relu):
        """

        :param ninput: number of inputs
        :param nhidden: number of hidden units
        :param noutput: number of outputs
        :param nlayer: number of weight matrices (2; standard MLP)
        :param actfun: used activation function (ReLU)
        """

        links = ChainList()
        if nlayer == 1:
            links.add_link(L.Linear(ninput, noutput))
        else:
            links.add_link(L.Linear(ninput, nhidden))
            for i in range(nlayer - 2):
                links.add_link(L.Linear(nhidden, nhidden))
            links.add_link(L.Linear(nhidden, noutput))

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.nlayer = nlayer
        self.actfun = actfun

        self.h = {}

        super(DeepNeuralNetwork, self).__init__(links)

    def __call__(self, x):

        if self.nlayer == 1:
            y = self[0][0](x)
        else:
            self.h[0] = self.actfun(self[0][0](x))
            for i in range(1,self.nlayer-1):
                self.h[i] = self.actfun(self[0][i](self.h[i-1]))
            y = self[0][-1](self.h[self.nlayer-2])

        return y


    def reset_state(self):
        # allows generic handling of stateful and stateless networks
        pass

#####
## Convolutional Neural Network

class ConvNet(Chain):
    """
    Basic convolutional neural network
    """

    def __init__(self, ninput, nhidden, noutput):
        """

        :param ninput: nchannels x height x width
        :param nhidden: number of hidden units
        :param noutput: number of action outputs
        """
        super(ConvNet, self).__init__(
            # dependence between filter size and padding; here output still 20x20 due to padding
            l1=L.Convolution2D(ninput[0], nhidden, 3, 1, 1),
            l2=L.Linear(np.prod(ninput) * nhidden, noutput)
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput

        self.h = {}

    def __call__(self, x):
        """
        :param x: sensory input (ntrials x nchannels x ninput[0] x ninput[1])
        """

        self.h[0] = F.relu(self.l1(x))
        y = self.l2(self.h[0])

        return y

    def reset_state(self):
        pass


#####
## Recurrent Neural Network

class RecurrentNeuralNetwork(ChainList):
    """
    Recurrent neural network consisting of a chain of layers (weight matrices)
    with a fixed number of nhidden units

    nlayer determines number of layers. The last layer is always a linear layer. The other layers
    make use of an activation function actfun

    """

    def __init__(self, ninput, nhidden, noutput, nlayer=2, link=L.LSTM):
        """

        :param ninput: number of inputs
        :param nhidden: number of hidden units
        :param noutput: number of outputs
        :param nlayer: number of weight matrices (2 = standard RNN with one layer of hidden units)
        :param link: used recurrent link (LSTM)

        """

        links = ChainList()
        if nlayer == 1:
            links.add_link(L.Linear(ninput, noutput))
        else:
            links.add_link(link(ninput, nhidden))
            for i in range(nlayer - 2):
                links.add_link(link(nhidden, nhidden))
            links.add_link(L.Linear(nhidden, noutput))

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.nlayer = nlayer

        self.h = {}

        super(RecurrentNeuralNetwork, self).__init__(links)

    def __call__(self, x):

        if self.nlayer == 1:
            y = self[0][0](x)
        else:
            self.h[0] = self[0][0](x)
            for i in range(1,self.nlayer-1):
                self.h[i] = self[0][i](self.h[i-1])
            y = self[0][-1](self.h[self.nlayer-2])

        return y


    def reset_state(self):
        for i in range(self.nlayer - 1):
            self[0][i].reset_state()


class CRNN(Chain):
    """
    Recurrent neural network consisting of a chain of layers (weight matrices)
    with a fixed number of nhidden units

    nlayer determines number of layers. The last layer is always a linear layer. The other layers
    make use of an activation function actfun

    """

    def __init__(self, ninput, nhidden1, nhidden2, noutput, nlayer=2, link=L.LSTM):
        """

        :param ninput: number of inputs
        :param nhidden: number of hidden units
        :param noutput: number of outputs
        :param nlayer: number of weight matrices (2 = standard RNN with one layer of hidden units)
        :param link: used recurrent link (LSTM)

        """
        super(CRNN, self).__init__(
            l1 = L.Convolution2D(ninput[0], nhidden1, 3, 1, 1),
            l2 = link(8*8*nhidden1,nhidden2),
            l3 = L.Linear(nhidden2, noutput)
        )



        self.ninput = ninput
        self.nhidden1 = nhidden1
        self.nhidden2 = nhidden2
        self.noutput = noutput

        self.h = {}



    def __call__(self, x):

        self.h[0] = h = F.max_pooling_2d(F.relu(self.l1(x)), ksize=3, stride=3)
        self.h[1] = self.l2(self.h[0])
        y = self.l3(self.h[1])

        return y


    def reset_state(self):
        self.l2.reset_state()

class RNN_ElmanFB(Chain):
    """
    Implements an Elman network
    """

    def __init__(self, ninput, nhidden, nhidden2, noutput):
        super(RNN_ElmanFB, self).__init__(
            L1=CL.ElmanFB(ninput, nhidden, nhidden2, actfun=sigmoid.sigmoid),
            L2=CL.Elman(nhidden,nhidden2, actfun=sigmoid.sigmoid),
            L3=L.Linear(nhidden2, noutput, initialW=init.HeNormal()),
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.h={}



    def __call__(self, x):

        self.h[0] = self.L1(x,self.L2.h)
        self.h[1] = self.L2(self.h[0])
        y = self.L3(self.h[1])

        return y

    def reset_state(self):
        self.L1.reset_state()
        self.L2.reset_state()

class RNN_ElmanFB1(Chain):
    """
    Implements an Elman network
    """

    def __init__(self, ninput, nhidden, noutput):
        super(RNN_ElmanFB1, self).__init__(
            L1=CL.ElmanFB(ninput, nhidden, noutput, actfun=sigmoid.sigmoid),
            L2=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.h={}



    def __call__(self, x):

        self.h[0] = self.L1(x,self.y)
        self.y = self.L2(self.h[0])

        return self.y

    def reset_state(self):
        self.L1.reset_state()
        self.y=None

class RNN_Elman2(Chain):
    """
    Implements an Elman network
    """

    def __init__(self, ninput, nhidden, nhidden2, noutput):
        super(RNN_Elman2, self).__init__(
            L1=CL.Elman(ninput, nhidden, actfun=sigmoid.sigmoid),
            L2=CL.Elman(nhidden,nhidden2, actfun=sigmoid.sigmoid),
            L3=L.Linear(nhidden2, noutput, initialW=init.HeNormal()),
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.h={}
        self.h[1]=None


    def __call__(self, x):

        self.h[0] = self.L1(x)
        self.h[1] = self.L2(self.h[0])
        y = self.L3(self.h[1])

        return y

    def reset_state(self):
        self.L1.reset_state()
        self.L2.reset_state()

class RNN_Elman(Chain):
    """
    Implements an Elman network
    """

    def __init__(self, ninput, nhidden, noutput,link=CL.Elman):
        super(RNN_Elman, self).__init__(
            L1=link(ninput, nhidden, actfun=sigmoid.sigmoid),
            L2=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.h={}

    def __call__(self, x):

        self.h[0] = self.L1(x)
        y = self.L2(F.dropout(self.h[0],ratio=0))

        return y

    def reset_state(self):
        self.L1.reset_state()

class RNN_Miconi(Chain):
    """
    Implements an Elman network
    """

    def __init__(self, ninput, nhidden, noutput, link=CL.Miconi):
        super(RNN_Miconi, self).__init__(
            L1=link(ninput, nhidden,dtdivtau=15/30.),
            L2=L.Linear(nhidden, noutput, initialW=init.HeNormal()),
        )

        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.h = []

    def __call__(self, x):
        self.h = self.L1(x)
        y = self.L2(F.dropout(self.h, ratio=0))

        return y

    def reset_state(self):
        self.L1.reset_state()