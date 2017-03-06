import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from chainer import Variable, cuda
import scipy.stats as ss
from chainer import Variable

class Analysis(object):

    def __init__(self, model, fname=None, gpu=-1):

        self.fname = fname
        self.model = model

        self.xp = np if gpu == -1 else cuda.cupy

    def regression_analysis(self, X, T):

        self.model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = True
        self.model.predictor.train = False

        Y = []
        for step in xrange(X.shape[0]):

            x = Variable(self.xp.asarray(X[step][None]), True)
            Y.append(self.model.predict(x))

            if step == 0:
                H = [[self.model.predictor.h[i].data[0]] for i in xrange(len(self.model.predictor.h))]
            else:
                _ = [H[i].append(self.model.predictor.h[i].data[0]) for i in xrange(len(self.model.predictor.h))]

        H = [self.xp.asarray(H[i]) for i in xrange(len(H))]
        Y = np.squeeze(self.xp.asarray(Y))

        [nexamples, nregressors] = Y.shape
        nregressors=1000
        plt.clf()

        plt.subplot(121)
        colors = cm.rainbow(np.linspace(0, 1, nregressors))
        for i in range(nregressors):
            plt.scatter(T[:, i], Y[:, i], c=colors[i,:])
            plt.hold('on')
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('Observed value')
        plt.ylabel('Predicted value')
        plt.title('Scatterplot')

        plt.subplot(122)
        R = np.zeros([nregressors,1])
        for i in range(nregressors):
            R[i] = ss.pearsonr(np.squeeze(T[:,i]),np.squeeze(Y[:,i]))[0]
        # plt.hist(R, np.min([nregressors, 50]), normed=1, facecolor='black')
        # plt.grid(True)
        # plt.xlabel('Pearson correlation')
        # plt.title('Histogram of Pearson correlations')

        print 'Correlation between predicted and observed outputs: {0}'.format(np.mean(R))

        if self.fname:
            plt.savefig(self.fname + '_regression_analysis.png')
        else:
            plt.show()

    def classification_analysis(self, X, T,trial_length=None):

        self.model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = True
        self.model.predictor.train = False

        Y = []
        for step in xrange(X.shape[0]):

            x = Variable(self.xp.asarray(X[step][None]), True)
            Y.append(self.model.predict(x))

            if step == 0:
                H = [[self.model.predictor.h[i].data[0]] for i in xrange(len(self.model.predictor.h))]
            else:
                _ = [H[i].append(self.model.predictor.h[i].data[0]) for i in xrange(len(self.model.predictor.h))]

            if step % trial_length == 0:
                self.model.predictor.reset_state()

        H = [self.xp.asarray(H[i]) for i in xrange(len(H))]
        Y = np.squeeze(self.xp.asarray(Y))

        [nexamples, nregressors] = Y.shape


        # compute count matrix
        count_mat = np.zeros([nregressors, nregressors])
        conf_mat = np.zeros([nregressors, nregressors])
        for i in range(nregressors):

            # get predictions for trials with real class equal to i
            clf = np.argmax(Y[T==i],axis=1)
            for j in range(nregressors):
                count_mat[i,j] = np.sum(clf == j)
            conf_mat[i] = count_mat[i]/np.sum(count_mat[i])

        # print accuracy
        clf = np.argmax(Y, axis=1)
        print 'Classification accuracy: {0}'.format(np.mean(clf==T))

	plt.clf()

        plt.subplot(121)
        plt.imshow(count_mat,interpolation=None)
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        plt.xticks(np.arange(nregressors))
        plt.gca().set_xticklabels([str(item) for item in 1+np.arange(nregressors)])
        plt.yticks(np.arange(nregressors))
        plt.gca().set_yticklabels([str(item) for item in 1+np.arange(nregressors)])
        plt.colorbar()
        plt.title('Count matrix')

        plt.subplot(122)
        plt.imshow(conf_mat,interpolation=None)
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        plt.xticks(np.arange(nregressors))
        plt.gca().set_xticklabels([str(item) for item in 1 + np.arange(nregressors)])
        plt.yticks(np.arange(nregressors))
        plt.gca().set_yticklabels([str(item) for item in 1 + np.arange(nregressors)])
        plt.colorbar()
        plt.title('Confusion matrix')

        if self.fname:
            plt.savefig(self.fname + '_classification_analysis.png')
        else:
            plt.show()

    def accuracy(self, supervised_data):
        """
        Return overall accuracy, calculated in batches (much faster).
        
        :param supervised_data: SupervisedData object
        """
        self.model.predictor.reset_state()
        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = True
        self.model.predictor.train = False
        Y = []
        T = []
        for data in supervised_data:
            x = Variable(self.xp.asarray(data[0]), True)
            Y.append(np.argmax(self.model.predict(x),axis=1))
            T.append(data[1])
        Y = np.squeeze(np.asarray(Y))
        T = np.squeeze(np.asarray(T))
        acc = np.mean(Y==T)      
        return acc
    
    def weight_matrix(self, W):
        """
        Plot weight matrix

        :param fname: file name
        :param W: N x M weight matrix
        """

        plt.clf()
        plt.pcolor(W)
        plt.title('Weight matrix')

        if self.fname:
            plt.savefig(self.fname + '_weight_matrix.png')
        else:
            plt.show()


    def functional_connectivity(self,data):
        """
        Plot functional connectivity matrix (full correlation)


        # perform an analysis on the optimal model
        z = [validation_data.X]
        [z.append(H[i]) for i in range(len(H))]
        z.append(Y)
        ana.functional_connectivity(z)


        :param data: list containing T x Mi timeseries data
        """

        x = np.hstack(data)
        M = np.corrcoef(x.transpose())

        plt.clf()
        plt.pcolor(M)
        plt.title('Functional connectivity')

        if self.fname:
            plt.savefig(self.fname + '_functional_connectivity.png')
        else:
            plt.show()


    def classification_analysis_Cifar(self, data, trial_length=None):
        self.model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = True
        self.model.predictor.train = False
        data.batch_size=data.X.shape[0]
        data.predict=1
        data.step=0


        #X=data.X
        #T=data.T
        Y = np.zeros((data.T.shape[0],data.T.shape[1],10))
        #for step in xrange(X.shape[0]):
        for _x, _t in data:
            x = Variable(_x, True)
            t = Variable(_t, True)
            #x = Variable(self.xp.asarray(X[step][None]), True)
            Y[:,data.step-1,:]=self.model.predict(x)

            if data.step == 1:
                H = [[self.model.predictor.h[i].data[0]] for i in xrange(len(self.model.predictor.h))]
            else:
                _ = [H[i].append(self.model.predictor.h[i].data[0]) for i in xrange(len(self.model.predictor.h))]

        H = [self.xp.asarray(H[i]) for i in xrange(len(H))]
        Y = np.reshape(np.squeeze(Y),(-1,10))

        [nexamples, nregressors] = Y.shape

        plt.clf()

        # compute count matrix
        count_mat = np.zeros([nregressors, nregressors])
        conf_mat = np.zeros([nregressors, nregressors])

        for i in range(nregressors):

            # get predictions for trials with real class equal to i
            clf = np.argmax(Y[data.T.reshape(-1) == i], axis=1)
            for j in range(nregressors):
                count_mat[i, j] = np.sum(clf == j)
            conf_mat[i] = count_mat[i] / np.sum(count_mat[i])

        # print accuracy
        clf = np.argmax(Y, axis=1)
        
        clft=np.mean(np.reshape(clf,(-1,trial_length)) == data.T,axis=0)

        if self.fname:
                np.savetxt(self.fname + '_class_accuracy',clft,fmt='%.4f', delimiter=',')
        else:
            print 'Classification accuracy: {0}'.format(clft)

        plt.subplot(121)
        plt.imshow(count_mat, interpolation=None)
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        plt.xticks(np.arange(nregressors))
        plt.gca().set_xticklabels([str(item) for item in 1 + np.arange(nregressors)])
        plt.yticks(np.arange(nregressors))
        plt.gca().set_yticklabels([str(item) for item in 1 + np.arange(nregressors)])
        plt.colorbar()
        plt.title('Count matrix')

        plt.subplot(122)
        plt.imshow(conf_mat, interpolation=None)
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        plt.xticks(np.arange(nregressors))
        plt.gca().set_xticklabels([str(item) for item in 1 + np.arange(nregressors)])
        plt.yticks(np.arange(nregressors))
        plt.gca().set_yticklabels([str(item) for item in 1 + np.arange(nregressors)])
        plt.colorbar()
        plt.title('Confusion matrix')

        # if self.fname:
        #     plt.savefig(self.fname + '_classification_analysis.png')
        # else:
        #     plt.show()

    def feedback_measure(self, threshold):
        """"Measure feedback vs feedforward connectivity from weight matrix"""

        W = self.model.predictor[0][0].W.W.data
        nhidden = W.shape[0]
        W[W < threshold] = 0
        I = self.model.predictor[0][0].U.W.data
        W1 = np.copy(W)
        D = np.copy(W)

        # Determine step distance between neurons
        for ii in xrange(nhidden):
            W1 = np.dot(W1, W)
            ind = np.nonzero(D == 0)
            print(sum(sum(ind)))
            D[ind] = (W1[ind] > 0) * (ii + 2)  # +2 because python numbering is different

        print(np.sum(D))
        np.fill_diagonal(D, 0)

        sinr = np.sum(I, 1)  #
        indr = sinr > 0
        h = np.zeros(nhidden)

        # Determine hierarchy (distance from input)
        for ii in xrange(nhidden):
            DI = D[ii, :] * indr
            DI = DI[np.nonzero(DI)]

            if DI.size > 0:
                h[ii] = np.min(DI) + 1

            else:
                h[ii] = None

        h[indr] = 1

        Fb = np.zeros(nhidden)
        Ff = np.zeros(nhidden)

        # Determine ratio between FF and FB
        for ii in xrange(nhidden):
            rnk = h[W[ii, :].astype('bool')]

            Fb[ii] = np.sum(rnk > h[ii])
            Ff[ii] = np.sum(rnk < h[ii]) + sinr[ii]

        rat = Fb / Ff.astype('float32')
        measure = np.mean(rat[np.isfinite(rat)])
        print(np.sum(h))
        return measure

    def predict(self, X):

        self.model.predictor.reset_state()

        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = True
        self.model.predictor.train = False

        Y = []
        for step in xrange(X.shape[0]):

            x = Variable(self.xp.asarray(X[step][None]), True)
            Y.append(self.model.predict(x))

            if step == 0:
                H = [[self.model.predictor.h[i].data[0]] for i in xrange(len(self.model.predictor.h))]
            else:
                _ = [H[i].append(self.model.predictor.h[i].data[0]) for i in xrange(len(self.model.predictor.h))]

        H = [self.xp.asarray(H[i]) for i in xrange(len(H))]
        Y = np.squeeze(self.xp.asarray(Y))
        return Y, H

    def reverse_correlation(self, P):

        self.model.predictor.reset_state()

        self.model.predictor.test = True
        self.model.predictor.train = False
        Y = []

        # generate activations for noise patterns
        for seq in xrange(P.shape[0]):
            for step in xrange(P.shape[1]):
                x = Variable(np.asarray([P[seq, step]]), False)
                Y.append(self.model.predict(x))
                if (seq == 0 and step == 0):
                    H = [[self.model.predictor.h[i].data[0]] for i in xrange(len(self.model.predictor.h))]
                else:
                    _ = [H[i].append(self.model.predictor.h[i].data[0]) for i in xrange(len(self.model.predictor.h))]
            if seq%100==0:
                print(seq)
            self.model.predictor.reset_state()

        H=np.squeeze(np.asarray(H))
        Y = np.squeeze(np.asarray(Y))
        A=np.zeros((P.shape[1],H[0].shape[1],P.shape[2]))
        B=np.zeros((P.shape[1],H[1].shape[1],P.shape[2]))
        C=np.zeros((P.shape[1],Y.shape[1],P.shape[2]))

        for step in xrange(P.shape[1]):
            A[step] = (np.dot(np.squeeze(H[0,step::18,:]).T,np.squeeze(P[:,step,:])).T/np.sum(np.squeeze(H[0,step::18,:]),axis=0)).T
            B[step] = (np.dot(np.squeeze(H[1,step::18, :]).T, np.squeeze(P[:, step, :])).T/np.sum(np.squeeze(H[1,step::18,:]),axis=0)).T
            C[step] = (np.dot(Y[step::18, :].T, np.squeeze(P[:, step, :])).T/np.sum(np.squeeze(Y[step::18,:]),axis=0)).T

        fig, axes = plt.subplots(8, 18,figsize=(15,7))
        for iii in xrange(8):
            for ii in xrange(18):
                axes[iii,ii].imshow(np.reshape(B[ii,iii,:],(22,22)))
    #    fig.tight_layout()
        plt.show()

        return A, B, C



