import numpy as np


class BackPropagationNetwork:

    # Class members
    layercount = 0
    shape = None
    weights = []

    def __init__(self, layersize):
        # initialise the network.
        self.layercount = len(layersize)-1
        self.shape = layersize

        self._layerInput = []
        self._layerOutput = []

        for (l1, l2) in zip(layersize[:-1], layersize[1:]):
            self.weights.append(np.random.normal(scale=0.1, size=(l2, l1+1)))

    def sgm(self, x, Derivative=False):
        if not Derivative:
            return 1/(1+np.exp(-x))
        else:
            out = 1/(1+np.exp(-x))
            return out*(1-out)

    def run(self, input):
        lnCases = input.shape[0]
        self._layerInput = []
        self._layerOutput = []

        for i in range(self.layercount):
            if i == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))
            else:
                layerInput = self.weights[i].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sgm(layerInput))

        return self._layerOutput[-1].T

    def TrainEpoch(self, input, target, trainingRate=0.2):
        delta = []
        lnCases = input.shape[0]

        self.run(input)
        for index in reversed(range(self.layercount)):
            if self.layercount-1:
                output_delta = self._layerOutput[i] - target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta*self.sgm(self._layerInput[index], True))
            else:
                delta_pullback = self.weights[index+1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :]*self.sgm(self._layerInput[index], True))

        for index in range(self.layercount):
            delta_index = self.layercount-1 - index
            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, lnCases])])
            else:
                layerOutput = np.vstack([self._layerOutput[index-1], np.ones([1, self._layerOutput[index-1].shape[1]])])

            weightDelta = np.sum(layerOutput[None, :, :].transpose(2, 0, 1)*delta[delta_index][None, :, :].transpose(2, 0, 1), axis=0)

            self.weights[index] = trainingRate * weightDelta

        return error


if __name__ == '__main__':
    bpn = BackPropagationNetwork((2, 2, 1))
    print(bpn.shape)
    print(bpn.weights)

    lvInput = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    lvTarget = np.array([[0.05], [0.05], [0.95], [0.95]])

    lnMax = 10000
    lnErr = 1e-5
    for i in range(lnMax-1):
        err = bpn.TrainEpoch(lvInput, lvTarget)
        if i%2500 == 0:
            print "Iteration {0}\t error: {1:0.6f}".format(i, err)
        if err <= minError:
            print "Minimum error reached at iteration {0}".format(i)
            
    lvOutput = bpn.run(lvInput)

    print("Input: {0}\nOutput: {1}".format(lvInput, lvOutput))