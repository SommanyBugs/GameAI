from __future__ import print_function

__author__ = 'Tom Schaul, Sun Yi, Tobias Glasmachers'

from pybrain.tools.rankingfunctions import HansenRanking
from pybrain.optimization.distributionbased.distributionbased import DistributionBasedOptimizer
from pybrain.auxiliary.importancemixing import importanceMixing
from scipy.linalg import expm
from scipy import dot, array, randn, eye, outer, exp, trace, floor, log, sqrt
from pybrain.optimization import XNES

import threading 
import gym
import gym_gvgai
import numpy as np
from random import randint

#global setting
omega = 32
epsilon = 1
delta = 10
gameStep = 150
gameRun = 2
updateDict = []
trainSet =[]
config = [gameRun, gameStep, updateDict]
mtx = threading.Lock()
outputDim = 6

class RNN():
    def __init__(self, w=None , u=None, b=None, input_dim=1, output_dim=1):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.state_t = np.zeros((output_dim,))
        self.successive_outputs = []
        self.maxes = []
        self.end_flag = 0
        self.score = 0
        self.w = np.random.random((output_dim,input_dim)) if w is None else w
        self.u = np.random.random((output_dim,output_dim)) if u is None else u
        self.b = np.random.random((output_dim,)) if b is None else b
    def sigmoid(self,x):
        s = 1 / (1 + np.exp(-x))
        return s
    def act(self, stateObs):
        if self.input_dim <= 0:
            return np.random.randint(0,self.output_dim)
        input_t = stateObs
        output_t = self.sigmoid(np.dot(self.w,input_t)+np.dot(self.u,self.state_t)+self.b)
        self.successive_outputs.append(output_t)
        self.state_t = output_t
        max_t = np.argmax(output_t, axis=0)
        self.maxes.append(max_t)
        action_id = max_t
        return action_id

#The main part of SixNeuron 
class DyXnes(XNES):
    """ NES with exponential parameter representation. """
    def __init__(self):
        #self.compressor = simulator()
        self.batchSize = 5
        self.maxEvaluations  = 200
        super().__init__(fitness, -ones(1))
    

    #Here is the main part of the modify of the
    def _learnStep(self):
        """ Main part of the algorithm. """    
        #initialize before learn
        global updateDict
        global trainSet

        #IDVQ trainning
        runTrain()

        #initialize the trainning set
        trainSet = []

        #print(64)
        input_dim = len(updateDict)
        lenWeights = input_dim*outputDim + outputDim*outputDim + 1 # Length of all weight parameters that need to be train
        #print(79)

        #XNES train
        # Dynamic Part, Upadate A and center with new demensions
        # modify the _A, _invA, numParameters, _center, _allEvaluated
        eta = 0.0001 # eta is an arbitrarily small real number 
        orgLenA = len(self._A) # original number of columns/rows of matrix A
        addNum = lenWeights - orgLenA
        #print(86)


        # Upadate matrix self._A
        #print(self._A.shape)
        biasColumn = self._A[:,-1] # save the last column
        biasRow = self._A[-1,:] # save the last row
        self._A = np.delete(self._A,-1,1) # delete the last column

        self._A = np.delete(self._A,-1,0) # delete the last row
        # extend the matrix size of A according to the code length
        #print(self._A.shape)
        self._A = np.column_stack((self._A, np.zeros((orgLenA-1,addNum+1))))
        self._A = np.row_stack((self._A, np.zeros((addNum+1,lenWeights))))
        # set the values of elements in A  
        #print(self._A.shape)
        self._A[-1,-1] = biasColumn[-1]
        for i in range(orgLenA-1):
            self._A[i,-1] = biasColumn[i]
            self._A[-1,i] = biasRow[i]
        for i in range(orgLenA-1, lenWeights-1):
            self._A[i,i] = eta

        # Update self.center
        #print(108)
        centerBias = self._center[-1]
        self._center = np.delete(self._center,-1,axis=0)
        for i in range(addNum):
            self._center = np.append(self._center,0)
        self._center = np.append(self._center,centerBias)

        # Upadate self.numParameters
        self.numParameters = lenWeights

        #print("116")
        # **Upadate self._invA **
        # Need Complete
        biasColumn = self._invA[:,-1] # save the last column
        biasRow = self._invA[-1,:] # save the last row
        self._invA = np.delete(self._invA,-1,1) # delete the last column
        self._invA = np.delete(self._invA,-1,0) # delete the last row
        # extend the matrix size of A according to the code length
        self._invA = np.column_stack((self._invA, np.zeros((orgLenA-1,addNum+1))))
        self._invA = np.row_stack((self._invA, np.zeros((addNum+1,lenWeights))))
        # set the values of elements in A  
        self._invA[-1,-1] = biasColumn[-1]
        for i in range(orgLenA-1):
            self._invA[i,-1] = biasColumn[i]
            self._invA[-1,i] = biasRow[i]
        for i in range(orgLenA-1, lenWeights-1):
            self._invA[i,i] = eta
        
        # Upadate self.numParameters
        self.numParameters = lenWeights
        I = eye(self.numParameters)
        #print(len(self._produceSample()))
        #print("134")
        '''
        self._produceSamples()
        '''
        """ Append batch size new samples and evaluate them. """
        reuseindices = []
        if self.numLearningSteps == 0 or not self.importanceMixing:
            #print(len(self._center))
            #print(149)
            [self._oneEvaluation(self._sample2base(self._produceSample())) for _ in range(self.batchSize)]
            #print(self.batchSize)
            self._pointers = list(range(len(self._allEvaluated)-self.batchSize, len(self._allEvaluated)))
        self._allGenSteps.append(self._allGenSteps[-1]+self.batchSize-len(reuseindices))
        self._allPointers.append(self._pointers)
        # Update self.center

        #print("144")
        #Update for All evaluated
        utilities = self.shapingFunction(self._currentEvaluations)
        utilities /= sum(utilities)  # make the utilities sum to 1
        if self.uniformBaseline:
            utilities -= 1./self.batchSize
        samples = array(list(map(self._base2sample, self._population)))

        dCenter = dot(samples.T, utilities)
        covGradient = dot(array([outer(s,s) - I for s in samples]).T, utilities)
        covTrace = trace(covGradient)
        covGradient -= covTrace/self.numParameters * I
        dA = 0.5 * (self.scaleLearningRate * covTrace/self.numParameters * I
                    +self.covLearningRate * covGradient)

        self._lastLogDetA = self._logDetA
        self._lastInvA = self._invA

        self._center += self.centerLearningRate * dot(self._A, dCenter)
        self._A = dot(self._A, expm(dA))
        self._invA = dot(expm(-dA), self._invA)
        self._logDetA += 0.5 * self.scaleLearningRate * covTrace
        if self.storeAllDistributions:
            self._allDistributions.append((self._center.copy(), self._A.copy()))
        #print("finish!s")


def fitness(dist):
    global outputDim
    times = config[0]
    step = config[1]
    curDict = config[2]
    lenCode = len(curDict)
    r = RNN(input_dim=lenCode,output_dim=outputDim)
    tlist,result = [],[]
    for x in range(times):
        t = threading.Thread(target=runGame, args=(r,step,result,curDict))
        tlist.append(t)
        t.start()
    for x in tlist:
        x.join()
    print(sum(result))
    return sum(result)

def downSample(stateObs):
    stateObs = stateObs[::2]
    obs = np.array([])
    for row in stateObs:
        row = row[::2]
        for col in row:
            rgbA = np.sum(np.array(col,dtype="int32")**2)
            obs = np.append(obs,rgbA)
    return obs

def runGame(agent,step,result,curDict):
    global trainSet
    env = gym_gvgai.make('gvgai-testgame1-lvl0-v0')
    #print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))
    actions = env.env.GVGAI.actions()
    stateObs = None
    current_score = 0
    localSet = []
    stateObs = env.reset()
    for t in range(step):
        #env.render()
        stateObs = downSample(stateObs)
        if t % 5 == 0:
            localSet.append(stateObs)
        code = DRSC(stateObs,curDict)
        action_id = agent.act(code)
        stateObs, increScore, done, debcug = env.step(action_id)
        current_score += increScore
        if done:
            ##print("Game over at game tick " + str(t+1) + " with player " + debug['winner'] + ", score is " + str(current_score))
            break
    result.append(current_score)
    if localSet != []:
        trainSet.append(localSet[np.random.randint(len(localSet))])
    return

def DRSC(x, curDict, epsilon=1, omega=10):
    p,w,code = x,0,[0]*len(curDict)
    if code == []:
        return code
    while np.sum(p) > epsilon and w < omega:
        delta = np.array([])
        for d in curDict:
            eq = p==d
            sim = eq.sum()
            delta = np.append(delta,sim)
        msc = np.argmax(delta)
        code[msc] = 1
        w = w+1
        p = p-curDict[msc]
        p = p.clip(min=0)
    return code

def IDVQ(trainSet, delta=10):
    global mtx
    global updateDict
    mtx.acquire()
    curDict = updateDict[:]
    for x in trainSet:
        p = x
        c = DRSC(x, curDict)
        code = np.array(c,dtype=bool)
        p_new = np.array(curDict)[code]
        p_new = np.sum(p_new, axis=0)
        R = p-p_new
        R = R.clip(min=0)
        if np.sum(R) > delta:
            curDict.append(R)
    updateDict = curDict
    mtx.release()
    return

def runTrain():
    global trainSet
    localSet = trainSet[:]
    trainSet = []
    t = threading.Thread(target=IDVQ, args=(localSet,))
    t.start()





if __name__ == '__main__':
    from pybrain.rl.environments.functions.unimodal import RosenbrockFunction
    from scipy import ones
    l = DyXnes()
    res = l.learn()
    print(res)
    print(len(res[0]))
    print(len(updateDict))

