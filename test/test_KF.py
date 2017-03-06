import chainer
import numpy as np
import scipy.io

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('/vol/ccnlab-scratch1/silqua/ANN')

import datasets
from chainer.functions.activation import sigmoid
import models.custom_links as CL
import supervised_learning
from analysis import Analysis
from models import neural_networks as models
from models.utilities import Regressor
import scipy.io
import scipy.stats as sst

seqs = 192 # trials number.
epochs = 100 # epoch number.
ntime = 25 # number of time points
ninput = 1 # number of inputs
nhid = 10 # number of hidden units
nhid2 = 10


tpoints=5000
valpoints=1000
n=10 # number of states and observations of model data

S = np.zeros([tpoints, n])
h = np.zeros([tpoints, n])
h2 = np.zeros([tpoints, n])
z = np.zeros([tpoints, n])

x=np.zeros([2])
A=np.zeros([10,2])

W1 = np.random.rand(n, n)
Ws = np.sum(W1, axis=1)
W1 = W1 / Ws[:, None]
W2 = np.random.rand(n, n)
Ws = np.sum(W2, axis=1)
W2 = W2 / Ws[:, None]
W3 = np.random.rand(n, n)
Ws = np.sum(W3, axis=1)
W3 = W3 / Ws[:, None]
W4 = np.random.rand(n, n)
Ws = np.sum(W4, axis=1)
W4 = W4 / Ws[:, None]

for t in xrange(0, tpoints - 1):
	# Model
	S[t, :] = np.random.rand(n)
	temp = np.dot(S[t, :], W1)
	if t % 25 != 0:
		temp += np.dot(h[t, :], W2)

	h[t, :] = np.tanh(temp)
	h2[t, :] = np.tanh(np.dot(h[t, :], W3))
	z[t, :] = np.dot(h2[t, :], W4)

for ii in xrange(10):

	in_train=S[0:tpoints-valpoints,:].astype(np.float32)
	out_train=z[0:tpoints-valpoints].astype(np.float32)
	in_val = S[tpoints-valpoints:tpoints,:].astype(np.float32)
	out_val = z[tpoints-valpoints:tpoints].astype(np.float32)

	nin = in_train.shape[1] # number of pixels
	nout = out_train.shape[1] # number of electrodes

	# get data
	training_data = datasets.DynamicData(in_train, out_train,batch_size=100,cutoff=25)
	validation_data = datasets.DynamicData(in_val, out_val,batch_size=100,cutoff=25)

	# define model
	model = Regressor(models.RNN_ElmanFB(nin, nhidden=10, nhidden2=10, noutput=nout))
#	model = Regressor(models.RNN_Elman2(nin, nhidden=10, nhidden2=10, noutput=nout))

	# Set up an optimizer
	optimizer = chainer.optimizers.Adam(alpha=0.005)

	optimizer.setup(model)

	ann = supervised_learning.RecurrentLearnerm2m(optimizer, cutoff=25)

	# Finally we run the optimization
	ann.optimize(training_data, validation_data=validation_data, epochs=100)

	loc = '/home/squax/python_scripts/KF/results/tmp'

	# create analysis object
	ana = Analysis(ann.model, fname=loc)

	# analyse data
	Y,H=ana.predict(validation_data.X)
	if n>1:
		Y=np.squeeze(Y[:,0])
	# plot loss and throughput
	x[0] = ((Y-z[tpoints - valpoints:tpoints, 0])**2).mean()#sst.pearsonr(Y, np.squeeze(S[tpoints - valpoints:tpoints, 0]))
	x[1],y2 = sst.pearsonr(Y, np.squeeze(S[tpoints - valpoints:tpoints, 0]))
	A[ii, :] = x

print('MSE_all=%4.3f' %np.mean(A,axis=0))
print('MSE_avg=%4.3f' %np.mean(A,axis=0))

print('MSE_std=%4.3f' %np.std(A,axis=0))

w=np.zeros(3)
if isinstance(ann.model.predictor,models.RNN_ElmanFB):
	w[0]=np.mean(np.abs(ann.model.predictor.L1.U.W.data))
	w[1]=np.mean(np.abs(ann.model.predictor.L1.V.W.data))
	w[2]=np.mean(np.abs(ann.model.predictor.L1.W.W.data))
else:
	w[0] = np.mean(np.abs(ann.model.predictor.L1.U.W.data))
	w[1] = np.mean(np.abs(ann.model.predictor.L1.W.W.data))
print('W(FF,LA,FB)=%4.3f' %w)
# ann.report()
#
# # create analysis object
# ana = Analysis(ann.model, fname='/home/squax/python_scripts/Zebra/results/tmp')
#
# ###
# # analyse data
# ana.regression_analysis(validation_data.X, validation_data.T)

