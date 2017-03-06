from chainer import Variable, cuda, serializers, optimizers
import numpy as np
import pickle
import time
import tqdm
import matplotlib.pyplot as plt


class SupervisedLearner(object):

    def __init__(self, optimizer, gpu=-1):

        self.model = optimizer.target
        self.optimizer = optimizer

        self.log = {}
        self.log[('training', 'loss')] = []
        self.log[('training', 'throughput')] = []
        self.log[('validation', 'loss')] = []
        self.log[('validation', 'throughput')] = []

        self.xp = np if gpu==-1 else cuda.cupy

    def optimize(self, training_data, validation_data=None, epochs=50):
        """

        :param training_data: Required training data set
        :param validation_data: Optional validation data set; optimize returns best model
                according to validation or last model it was trained on
        :param epochs: number of training epochs
        :return:
        """

        # keep track of minimal validation loss
        min_loss = float('nan')

        for epoch in tqdm.tqdm(xrange(self.optimizer.epoch, self.optimizer.epoch + epochs)):

            then = time.time()
            loss = self.train(training_data)
            print(loss)

            now = time.time()
            throughput = training_data.nexamples / (now - then)

            self.log[('training', 'loss')].append(loss)
            self.log[('training', 'throughput')].append(throughput)

            # testing in batch mode is much faster
            if validation_data:
                then = time.time()
                loss = self.test(validation_data)

                now = time.time()
                throughput = validation_data.nexamples / (now - then)
            else:
                loss = float('nan')
                throughput = float('nan')

            print(loss)
            self.log[('validation', 'loss')].append(loss)
            self.log[('validation', 'throughput')].append(throughput)

            # store optimal model
            if np.isnan(min_loss):
                optimal_model = self.optimizer.target.copy()
                min_loss = self.log[('validation', 'loss')][-1]
            else:
                if self.log[('validation', 'loss')][-1] < min_loss:
                    optimal_model = self.optimizer.target.copy()
                    min_loss = self.log[('validation', 'loss')][-1]

            self.optimizer.new_epoch()

            if isinstance(self.optimizer,optimizers.MomentumSGD):
                if self.optimizer.epoch==40:
                    self.optimizer.lr=0.01
                elif self.optimizer.epoch==50:
                    self.optimizer.lr=0.001

        # model is set to the optimal model according to validation loss
        # or to last model in case no validation set is used
        self.model = optimal_model

    def load(self, fname):

        # with open('{}_log'.format(fname), 'rb') as f:
        #     self.log = pickle.load(f)

        # serializers.load_npz('{}_optimizer'.format(fname), self.optimizer)
        serializers.load_npz('{}_model'.format(fname), self.model)


    def save(self, fname):

        #with open('{}_log'.format(fname), 'wb') as f:
        #    pickle.dump(self.log, f, -1)

        #serializers.save_npz('{}_optimizer'.format(fname), self.optimizer)
        serializers.save_npz('{}_model'.format(fname), self.model)


    def report(self, fname=None):

        plt.clf()
        plt.subplot(121)
        plt.plot(self.log[('training', 'loss')], 'r', label='training')
        plt.plot(self.log[('validation', 'loss')], 'g', label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.subplot(122)
        plt.plot(self.log[('training', 'throughput')], 'r', label='training')
        plt.plot(self.log[('validation', 'throughput')], 'g', label='validation')
        plt.xlabel('epoch')
        plt.ylabel('throughput')
        plt.legend()

        if fname:
            plt.savefig(fname)
        else:
            plt.show()


class FeedforwardLearner(SupervisedLearner):

    def train(self, data):
        self.model.predictor.reset_state()

        cumloss = self.xp.zeros((), 'float32')

        loss = Variable(self.xp.zeros((), 'float32'))

        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = False
        self.model.predictor.train = True

        for _x, _t in data:

            x = Variable(self.xp.asarray(_x))
            t = Variable(self.xp.asarray(_t))

            loss = self.model(x, t)
            cumloss += loss.data

            self.optimizer.zero_grads()
            loss.backward()
            self.optimizer.update()

        return float(cumloss / data.nbatches)


    def test(self, data):

        loss = Variable(self.xp.zeros((), 'float32'), True)

        model = self.model

        model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        model.predictor.test = True
        model.predictor.train = False

        for _x, _t in data:
            x = Variable(self.xp.asarray(_x), True)
            t = Variable(self.xp.asarray(_t), True)

            loss += model(x, t)

        return float(loss.data / data.nbatches)


class RecurrentLearnerm2m(SupervisedLearner):
    """many to many RNN"""
    def __init__(self, optimizer, gpu=-1, cutoff=None):
        """

        :param optimizer: Optimizer to run
        :param gpu: Run on GPU or not (-1)
        :param cutoff: cutoff length for truncated backpropagation (None=no cutoff)
        """

        super(RecurrentLearnerm2m, self).__init__(optimizer, gpu)

        self.cutoff = cutoff


    def train(self, data):

        if not self.cutoff:
            cutoff = data.nbatches
        else:
            cutoff = self.cutoff
	
        self.model.predictor.reset_state()

        cumloss = self.xp.zeros((), 'float32')

        loss = Variable(self.xp.zeros((), 'float32'))

        # check if we are in train or test mode (used e.g. for dropout)
        self.model.predictor.test = False
        self.model.predictor.train = True

        for _x, _t in data:

            x = Variable(self.xp.asarray(_x))
            t = Variable(self.xp.asarray(_t))

            loss += self.model(x, t)
#            self.model.predict(x)
            # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
            if data.step % cutoff == 0 or data.step == data.nbatches:
		
 #               loss +=self.model(x, t)
                self.optimizer.zero_grads()
                
                loss.backward()
                loss.unchain_backward()
                self.optimizer.update()

                cumloss += loss.data
                loss = Variable(self.xp.zeros((), 'float32'))
                self.model.predictor.reset_state()
        return float(cumloss/data.nbatches)


    def test(self, data):

        loss = Variable(self.xp.zeros((), 'float32'), True)
        cumloss = self.xp.zeros((), 'float32')

        model = self.model.copy()

        model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        model.predictor.test = True
        model.predictor.train = False

        for _x, _t in data:
            x = Variable(self.xp.asarray(_x), True)
            t = Variable(self.xp.asarray(_t), True)
            loss += model(x,t)

            if data.step % self.cutoff == 0 or data.step == data.nbatches:

                cumloss += loss.data
                model.predictor.reset_state()
                loss = Variable(self.xp.zeros((), 'float32'), True)

        return float(cumloss / data.nbatches)

    def test1(self, data):
        model = self.model.copy()

        model.predictor.reset_state()

        model.predictor.test = True
        model.predictor.train = False
        Y = []
        R2 = []
        ntime = 18
        import scipy.stats as sst
        from chainer import Variable
        for seq in xrange(data.X.shape[0]):
            for step in xrange(data.X.shape[1]):
                x = Variable(np.asarray([data.X[seq,step]]), False)
                Y.append(model.predictor(x).data)
            model.predictor.reset_state()

        x2 = np.zeros(data.X.shape[0])
        Y = np.squeeze(np.asarray(Y))
        for n in range(0, 20):
            for pat in xrange(data.X.shape[0]):
                x2[pat], y2 = sst.pearsonr(Y[(18 * pat):(18 * pat + 18), n].T,
                                           data.T[pat,:, n].T)
            R2.append(np.mean(x2**2))


        return np.mean(R2)


class RecurrentLearnerm21(SupervisedLearner):
    """many to one RNN"""
    def __init__(self, optimizer, gpu=-1, cutoff=None):
        """

        :param optimizer: Optimizer to run
        :param gpu: Run on GPU or not (-1)
        :param cutoff: cutoff length for truncated backpropagation (None=no cutoff)
        """

        super(RecurrentLearnerm21, self).__init__(optimizer, gpu)

        self.cutoff = cutoff

    def train(self, data):

        if not self.cutoff:
            cutoff = data.nbatches
        else:
            cutoff = self.cutoff

        self.model.predictor.reset_state()

        cumloss = self.xp.zeros((), 'float32')

        loss = Variable(self.xp.zeros((), 'float32'))

        # check if we are in train or test mode (used e.g. for dropout)
        self.model.predictor.test = False
        self.model.predictor.train = True

        for _x, _t in data:

            x = Variable(_x)
            t = Variable(_t)
            self.model.predictor(x)
            # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
            if data.step % cutoff == 0 or data.step == data.nbatches:

                loss += self.model(x, t)
                self.optimizer.zero_grads()

                loss.backward()
                loss.unchain_backward()
                self.optimizer.update()
                #self.model.predictor[0][0].U.W.data[10:,:]=0

                cumloss += loss.data
                loss = Variable(self.xp.zeros((), 'float32'))
                self.model.predictor.reset_state()


        return float(cumloss / (data.batch_ind.shape[1]))

    def test(self, data):

        loss = Variable(self.xp.zeros((), 'float32'), True)

        model = self.model.copy()

        model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        model.predictor.test = True
        model.predictor.train = False
        data.predict = 1

        for _x, _t in data:
            x = Variable(_x, True)
            t = Variable(_t, True)
            model.predictor(x)

            if data.step % data.trial_length == 0 or data.step == data.nbatches:
                loss += model(x, t)
                model.predictor.reset_state()

        return float(loss.data)
