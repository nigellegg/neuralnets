# This implementation of a standard feed forward network (FFN) is short and efficient,
# using numpy's array multiplications for fast forward and backward passes. The source
# code comes with a little example, where the network learns the XOR problem.
#
# Copyright 2008 - Thomas Rueckstiess
#
# Modifications for text classification using Gensim
# copyright 2014 Chibwe Ltd

from numpy import *
from gensim import corpora, models, similarities
import ntlutils
import nltk
import re
import logging


class PrepText:
    def __init__(self, TextName):
        self.DictName = ''

    def FileNamez(xfile):
        zfile = xfile[:-4]
        return zfile

    def dictCreate():
        stoplist = set('for a of the and to in http # @ : - A? he she it rt I you we twitpic com 1 2 3 4 5 6 7 8 9 0'.split())
        print 'clearing stopwords'
        dictionary = corpora.Dictionary(line.lower().split() for line in
                                        open(TextName))
        stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                    if stopword in dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq ==1]
        dictionary.filter_tokens(stop_ids + once_ids)
        dictionary.compactify()
        zfile = FileNamez(TextName)
        DictName = zfile+'.dict'
        dictionary.save(DictName)

    def dCorpusCreate():
        xdict = DictName
        dictionary = corpora.Dictionary.load(xdict)
        texts = open(TextName)
        corpus = [dictionary.doc2bow(line.lower().split()) for line in texts]
        zfile = FileNamez(TextName)
        print 'serialising corpus'
        corpora.MmCorpus.serialize(zfile+'.mm', corpus)


class FeedForwardNetwork:

    def __init__(self, nIn, nHidden, nOut):
        # learning rate
        self.alpha = 0.1

        # number of neurons in each layer
        self.nIn = nIn
        self.nHidden = nHidden
        self.nOut = nOut

        # initialize weights randomly (+1 for bias)
        self.hWeights = random.random((self.nHidden, self.nIn+1))
        self.oWeights = random.random((self.nOut, self.nHidden+1))

        # activations of neurons (sum of inputs)
        self.hActivation = zeros((self.nHidden, 1), dtype=float)
        self.oActivation = zeros((self.nOut, 1), dtype=float)

        # outputs of neurons (after sigmoid function)
        self.iOutput = zeros((self.nIn+1, 1), dtype=float)      # +1 for bias
        self.hOutput = zeros((self.nHidden+1, 1), dtype=float)  # +1 for bias
        self.oOutput = zeros((self.nOut), dtype=float)

        # deltas for hidden and output layer
        self.hDelta = zeros((self.nHidden), dtype=float)
        self.oDelta = zeros((self.nOut), dtype=float)

    def forward(self, input):
        # set input as output of first layer (bias neuron = 1.0)
        self.iOutput[:-1, 0] = input
        self.iOutput[-1:, 0] = 1.0

        # hidden layer
        self.hActivation = dot(self.hWeights, self.iOutput)
        self.hOutput[:-1, :] = tanh(self.hActivation)

        # set bias neuron in hidden layer to 1.0
        self.hOutput[-1:, :] = 1.0

        # output layer
        self.oActivation = dot(self.oWeights, self.hOutput)
        self.oOutput = tanh(self.oActivation)

    def backward(self, teach):
        error = self.oOutput - array(teach, dtype=float)

        # deltas of output neurons
        self.oDelta = (1 - tanh(self.oActivation)) * tanh(self.oActivation) * error

        # deltas of hidden neurons
        self.hDelta = (1 - tanh(self.hActivation)) * tanh(self.hActivation) * dot(self.oWeights[:,:-1].transpose(), self.oDelta)

        # apply weight changes
        self.hWeights = self.hWeights - self.alpha * dot(self.hDelta, self.iOutput.transpose())
        self.oWeights = self.oWeights - self.alpha * dot(self.oDelta, self.hOutput.transpose())

    def getOutput(self):
        return self.oOutput


if __name__ == '__main__':
    '''
    XOR test example for usage of ffn
    '''

    # define training set
    xorSet = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xorTeach = [[0], [1], [1], [0]]

    # create network
    ffn = FeedForwardNetwork(2, 2, 1)

    count = 0
    while(True):
        # choose one training sample at random
        rnd = random.randint(0,3)

        # forward and backward pass
        ffn.forward(xorSet[rnd])
        ffn.backward(xorTeach[rnd])

        # output for verification
        print count, xorSet[rnd], ffn.getOutput()[0],
        if ffn.getOutput()[0] > 0.8:
            print 'TRUE',
        elif ffn.getOutput()[0] < 0.2:
            print 'FALSE',
        print    
        count += 1