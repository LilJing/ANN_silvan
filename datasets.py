import numpy as np
import random
import chainer.datasets as datasets
import matplotlib.pyplot as plt

#####
## Base classes

class Data(object):

    def __init__(self):

        self.nbatches = self.X.shape[0] // self.batch_size

        self.step = 0

        self.nexamples = self.X.shape[0]

        # print 'Constructing labelled dataset; batch size: {0}; n batches: {1}'.format(self.batch_size, self.nbatches)
        # print 'Input data: {0} data points x {1} inputs'.format(self.nexamples,self.X.shape[1])
        # print 'Output data: {0} data points x {1} outputs'.format(self.nexamples, self.T.shape[1])

    def __iter__(self):
        return self  # simplest iterator creation

    def next(self):
        pass

    def reset(self):
        self.step = 0

class StaticData(Data):
    """
    Data class for static data consisting of independent data points
    """

    def __init__(self, X, T, batch_size=32):
        """

        :param X: ndatapoints x ninputs input data
        :param T: ndatapoints [x noutputs] target data
        :param batch_size: number of trials per batch

        """

        self.X = X
        self.T = T

        self.batch_size = batch_size

        self.perm = np.random.permutation(np.arange(len(self.X)))

        super(StaticData, self).__init__()

    def next(self):
        """

        :return: x: list of 1D arrays representing examples in the current minibatch
        """

        if self.step == self.nbatches:
            self.step = 0
            raise StopIteration

        x = [self.X[self.perm[(seq * self.nbatches + self.step) % len(self.X)]] for seq in xrange(self.batch_size)]
        t = [self.T[self.perm[(seq * self.nbatches + self.step) % len(self.T)]] for seq in xrange(self.batch_size)]

        self.step += 1

        return x, t

class DynamicData(Data):
    """
       Data class for dynamic data consisting of temporally ordered data points
    """

    def __init__(self, X, T, batch_size=32,cutoff=10):
        """

        :param X: ntimepoints x ninputs or ntrials x ntimepoints x ninputs input data
        :param T: ntimepoints [x noutputs] or ntrials x ntimepoints [x noutputs] target data
        :param batch_size: number of trials per batch

        NOTE:
        3D data is converted to 2D data. In this case, each of the trials will be processed in batch mode
        The batch size then becomes equal to ntrials since in each batch, all trials are processed at a certain time point

        """

        self.X = X
        self.T = T
        self.batch_size=batch_size
        self.trial_length=cutoff
        self.batch_ind=np.zeros((batch_size,X.shape[0]//batch_size))


        super(DynamicData, self).__init__()

    def next(self):
        """

        :return: x: list of 1D arrays representing examples in the current minibatch
        """


        if self.step == self.nbatches:
            self.step = 0
            raise StopIteration


        x = np.asarray([self.X[(seq * self.nbatches + self.step) % len(self.X)] for seq in xrange(self.batch_size)])
        t = np.asarray([self.T[(seq * self.nbatches + self.step) % len(self.T)] for seq in xrange(self.batch_size)])

        self.step += 1

        return x, t


#####
## Supervised datasets

class StaticDataClassification(StaticData):
    """
    Toy dataset for static classification data
    """

    def __init__(self, batch_size=32):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [0 if sum(i) < 1.0 else 1 for i in X]
        X = np.array(X, 'float32')
        T = np.array(T, 'int32')

        self.nin = X.shape[1]
        self.nout = np.max(T) + 1

        super(StaticDataClassification, self).__init__(X, T, batch_size)


class StaticDataRegression(StaticData):
    """
    Toy dataset for static regression data
    """

    def __init__(self, batch_size=32):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [[np.sum(i), np.prod(i)] for i in X]
        X = np.array(X, 'float32')
        T = np.array(T, 'float32')

        self.nin = X.shape[1]
        self.nout = T.shape[1]

        super(StaticDataRegression, self).__init__(X, T, batch_size)


class DynamicDataClassification(DynamicData):
    """
    Toy dataset for dynamic classification data in continuous mode
    """

    def __init__(self, batch_size=32):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [0] + [0 if sum(i) < 1.0 else 1 for i in X][:-1]
        X = np.array(X, 'float32')
        T = np.array(T, 'int32')

        self.nin = X.shape[1]
        self.nout = np.max(T) + 1

        super(DynamicDataClassification, self).__init__(X, T, batch_size)


class DynamicDataRegression(DynamicData):
    """
    Toy dataset for dynamic regression data in continuous mode
    """

    def __init__(self, batch_size=32):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [[1, 0]] + [[np.sum(i), np.prod(i)] for i in X][:-1]
        X = np.array(X, 'float32')
        T = np.array(T, 'float32')

        self.nin = X.shape[1]
        self.nout = T.shape[1]

        super(DynamicDataRegression, self).__init__(X, T, batch_size)

class DynamicDataRegressionBatch(DynamicData):
    """
    Toy dataset for dynamic regression data in batch mode
    """

    def __init__(self):

        X = [[random.random(), random.random()] for _ in xrange(992)]
        T = [[1, 0]] + [[np.sum(i), np.prod(i)] for i in X][:-1]
        X = np.array(X, 'float32')
        T = np.array(T, 'float32')

        self.nin = X.shape[1]
        self.nout = T.shape[1]

        X = np.reshape(X,[32,31,2])
        T = np.reshape(T,[32,31,2])

        super(DynamicDataRegressionBatch, self).__init__(X, T)

class MNISTData(StaticData):
    """
    Handwritten character dataset; example of handling convolutional input
    """


    def __init__(self, validation=False, convolutional=True, batch_size=32):

        if validation:
            data = datasets.get_mnist()[1]
        else:
            data = datasets.get_mnist()[0]

        X = data._datasets[0].astype('float32')
        T = data._datasets[1].astype('int32')

        if convolutional:
            X = np.reshape(X,np.concatenate([[X.shape[0]], [1], [28, 28]]))
            self.nin = [1, 28, 28]
        else:
            self.nin = X.shape[1]

        self.nout = (np.max(T) + 1)

        super(MNISTData, self).__init__(X, T, batch_size)

class CIFARData(Data):
    """
    Natural images dataset
    """

    def __init__(self, validation,data_loc,trial_length,pnoise,batch_size=32,c_noise=True):
        self.batch_size = batch_size
        self.pnoise=pnoise
        self.trial_length=trial_length
        self.predict=0
        self.c_noise=c_noise

        def unpickle(file):
            import cPickle
            fo = open(file, 'rb')
            dict = cPickle.load(fo)
            fo.close()
            return dict

        if validation:
            data_test = unpickle(data_loc + '/test_batch')
            self.X = data_test['data']
            self.T = np.asarray(data_test['labels'])
        else:
            data1 = unpickle(data_loc+'/data_batch_1')
            data2 = unpickle(data_loc+'/data_batch_2')
            data3 = unpickle(data_loc+'/data_batch_3')
            data4 = unpickle(data_loc+'/data_batch_4')
            data5 = unpickle(data_loc+'/data_batch_5')
            
            self.X = np.concatenate((data1['data'], data2['data'], data3['data'], data4['data'], data5['data']))
            
            self.T = np.asarray(data1['labels'] + data2['labels'] + data3['labels'] + data4['labels'] + data5['labels'])

        # Convert to grayscale
        self.X = (0.299 * self.X[:, :1024] + 0.587 * self.X[:, 1024:2048] + 0.114 * self.X[:,2048:]) / (255.)

        self.X =  np.tile(np.expand_dims(self.X,1),(1,self.trial_length,1)).astype('float32')
        self.T =  np.tile(np.expand_dims(self.T,1),(1,self.trial_length)).astype('int32')

        self.batch_ind = np.reshape(np.random.permutation(self.X.shape[0]),(self.batch_size,-1))

        super(CIFARData, self).__init__()

    def next(self):
        """

        :return: x: list of 1D arrays representing examples in the current minibatch
        """
        x=[]
        t=[]

        if self.predict==0:
            if self.step == self.batch_ind.shape[1]*self.trial_length:
                self.step = 0
                self.batch_ind = np.reshape(np.random.permutation(self.X.shape[0]),(self.batch_size,-1))

                raise StopIteration
        else:
            if self.step == self.trial_length:
                self.step = 0

                raise StopIteration



        for n in xrange(self.batch_size):

            if self.predict==1:
                seq=n
            else:
                seq=self.batch_ind[n,self.step/self.trial_length]

            tx=self.X[seq,self.step % self.trial_length,:]
            #tx[self.mask1]=np.random.rand(tx[self.mask1].size)
            x.append(tx)
            t.append(self.T[seq , self.step % self.trial_length])
        x=np.asarray(x)
        t=np.asarray(t)

        if self.c_noise:
            if self.step % self.trial_length ==0:
                self.mask1 = np.random.choice(2, size=(self.batch_size,self.X.shape[2]), p=[1 - self.pnoise, self.pnoise]).astype(np.bool)
                self.noise = np.random.rand(x[self.mask1].size)
        else:
            self.mask1 = np.random.choice(2, size=(self.batch_size, self.X.shape[2]),p=[1 - self.pnoise, self.pnoise]).astype(np.bool)
            self.noise = np.random.rand(x[self.mask1].size)

        x[self.mask1] = self.noise

        self.step += 1

	return x, t

class MNISTDataSilvan(Data):
    """
    Handwritten digit dataset;
    """


    def __init__(self, validation,trial_length,pnoise,batch_size=32,c_noise = True):
        self.batch_size = batch_size
        self.pnoise=pnoise
        self.trial_length=trial_length
        self.predict=0
        self.c_noise=c_noise

        if validation:
            data = datasets.get_mnist()[1]
        else:
            data = datasets.get_mnist()[0]

        self.X = data._datasets[0].astype('float32')
        self.T = data._datasets[1].astype('int32')

        self.X =  np.tile(np.expand_dims(self.X,1),(1,self.trial_length,1)).astype('float32')
        self.T =  np.tile(np.expand_dims(self.T,1),(1,self.trial_length)).astype('int32')

        self.batch_ind = np.reshape(np.random.permutation(self.X.shape[0]),(self.batch_size,-1))

        super(MNISTDataSilvan, self).__init__()

    def next(self):
        """

        :return: x: list of 1D arrays representing examples in the current minibatch
        """
        x=[]
        t=[]
        if self.predict==0:
            if self.step == self.batch_ind.shape[1]*self.trial_length:
                self.step = 0
                self.batch_ind = np.reshape(np.random.permutation(self.X.shape[0]),(self.batch_size,-1))

                raise StopIteration
        else:
            if self.step == self.trial_length:
                self.step = 0

                raise StopIteration

        for n in xrange(self.batch_size):

            if self.predict==1:
                seq=n
            else:
                seq=self.batch_ind[n,self.step/self.trial_length]

            tx=self.X[seq,self.step % self.trial_length,:]
        #    mask1 = np.random.choice(2, size=tx.shape, p=[1 - self.pnoise, self.pnoise]).astype(np.bool)
        #    tx[mask1]=np.random.rand(tx[mask1].size)
            x.append(tx)
            t.append(self.T[seq , self.step % self.trial_length])

        x=np.asarray(x)
        t=np.asarray(t)

        if self.c_noise:
            if self.step % self.trial_length ==0:
                self.mask1 = np.random.choice(2, size=(self.batch_size,self.X.shape[2]), p=[1 - self.pnoise, self.pnoise]).astype(np.bool)
                self.noise = np.random.rand(x[self.mask1].size)
        else:
            self.mask1 = np.random.choice(2, size=(self.batch_size, self.X.shape[2]),p=[1 - self.pnoise, self.pnoise]).astype(np.bool)
            self.noise = np.random.rand(x[self.mask1].size)

        x[self.mask1]=self.noise

        self.step += 1

        return x, t

class BartData(Data):
    """
       Data class for dynamic data consisting of temporally ordered data points
    """

    def __init__(self, X, T, batch_size=32, conv=0):
        """

        :param X: ntimepoints x ninputs or ntrials x ntimepoints x ninputs input data
        :param T: ntimepoints [x noutputs] or ntrials x ntimepoints [x noutputs] target data
        :param batch_size: number of trials per batch

        NOTE:
        3D data is converted to 2D data. In this case, each of the trials will be processed in batch mode
        The batch size then becomes equal to ntrials since in each batch, all trials are processed at a certain time point

        """


        ntrials, ntimepoints, nvariables = X.shape

        if conv==1:
            self.X = np.reshape(X,(ntrials,18,1,22,22))
        else:
            self.X=X

        self.T = np.swapaxes(T,1,2)

        # number of batches must be equal to number of trials
        self.batch_size = 1#ntrials

        self.batch_ind = np.random.randint(0,high=810,size=(self.batch_size,810))
        self.trial_length = ntimepoints
        super(BartData, self).__init__()


    def next(self):
        """

        :return: x: list of 1D arrays representing examples in the current minibatch
        """

        if self.step == 810*18:
            self.step = 0
            self.batch_ind = np.random.randint(0,high=810,size=(self.batch_size,810))
            raise StopIteration


        x = [self.X[self.batch_ind[patNo,self.step / self.trial_length],self.step % self.trial_length] for patNo in xrange(self.batch_size)]
        t = [self.T[self.batch_ind[patNo,self.step / self.trial_length],self.step % self.trial_length] for patNo in xrange(self.batch_size)]

        self.step += 1

        return x, t

class ZebraData(DynamicData):
    """
    Toy dataset for dynamic regression data in batch mode
    """

    def __init__(self):

        X = [[random.random(), random.random()] for _ in xrange(992)]
        T = [[1, 0]] + [[np.sum(i), np.prod(i)] for i in X][:-1]
        X = np.array(X, 'float32')
        T = np.array(T, 'float32')

        self.nin = X.shape[1]
        self.nout = T.shape[1]

        X = np.reshape(X,[32,31,2])
        T = np.reshape(T,[32,31,2])

        super(ZebraData, self).__init__(X, T)