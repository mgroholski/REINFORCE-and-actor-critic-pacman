from game import Directions, Agent, Actions
from pacman import GameState
import random,util,time,math
import sys
from featureExtractors import *

def softmaxPolicy(action, state, thetaVector, getLegalActions, featExtractor):
    # Implementation Help: https://towardsdatascience.com/policy-based-reinforcement-learning-the-easy-way-8de9a3356083
    numeratorFeatureVector = featExtractor.getFeatures(state, action)

    if thetaVector is None:
        thetaVector = [0] * len(numeratorFeatureVector)

    if len(numeratorFeatureVector) != len(thetaVector):
        print(f"Theta (Legnth {len(thetaVector)}) and Feature Vector (Length {len(numeratorFeatureVector)}) are different lengths.")
        exit()

    hValue = 0
    for i in range(len(thetaVector)):
        hValue += thetaVector[i] * numeratorFeatureVector[i]

    try:
        numerator = math.exp(hValue)
    except:
        if hValue < 0:
            numerator = 0
        else:
            numerator = sys.maxsize

    denominator = 0
    for legalAction in getLegalActions(state):
        featureVector = featExtractor.getFeatures(state, legalAction)
        hValue = 0
        for i in range(len(thetaVector)):
            hValue += thetaVector[i] * featureVector[i]

        try:
            denominator += math.exp(hValue)
        except:
            if hValue < 0:
                denominator += 0
            else:
                denominator += sys.maxsize

    return (numerator / denominator) if denominator != 0 else 0

# The Actor-Critic Agent class
class ActorCriticAgent(Agent):
    def __init__(self, actionFn=None, gamma= 0.8, alpha_theta=0.2, alpha_w=0.2, policy=softmaxPolicy, numTraining=100, extractor='SimpleExtractor'):
        if actionFn == None:
            actionFn = lambda state: state.getLegalActions()
        self.actionFn = actionFn

        self.featExtractor = util.lookup(extractor, globals())()

        self.theta = []
        self.w = []
        self.i = 1

        self.alpha_theta = float(alpha_theta)
        self.alpha_w = float(alpha_w)

        self.gamma = float(gamma)

        self.policy = policy

        self.numTraining = int(numTraining)
        self.episodesSoFar = 0

    def update(self):
        pass

    def getAction(self, state):
        actionProbabilities = []
        probabilitySum = 0
        for action in self.getLegalActions(state):
            probability = self.policy(action, state, self.theta, self.getLegalActions, self.featExtractor)
            actionProbabilities.append((probability, action))
            probabilitySum += probability

        randomNum = random.random()
        for action in actionProbabilities:
            randomNum -= action[0]
            if randomNum <= 0:
                self.doAction(state, action[1])
                return action[1]

        print("Did not find a legal action!")
        print("Action Probabilities: ", actionProbabilities)

        #Choose action uniformly at random
        return random.choice(self.getLegalActions(state))


    def getPolicy(self, state):
        util.raiseNotDefined()

    def getValue(self, state)->int:
        """
        Calculate the estimated value of a state based on a linear combination of features and their corresponding weights.
        """
        return -1


    def getLegalActions(self,state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.actionFn(state)

    def observeTransition(self, state, action, nextState, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        # Initialize w and theta if not already
        if len(self.w) == 0:
            self.w = [0] * len(self.featExtractor.getFeatures(state, action))
        if len(self.theta) == 0:
            self.theta = [0] * len(self.w)

        nabla_v = 0
        if nextState.isWin() or nextState.isLose():
            nabla_v = deltaReward
        else:
            nable_v = deltaReward + (self.gamma * self.getValue(nextState))


        delta = nabla_v - self.getValue(state)

        for t in range(len(self.episodeTriplets) - 1):
            gValue = self.episodeTriplets[t][2]

            # Calculates gradient vector
            featureVector = self.featExtractor.getFeatures(state, action)
            if self.theta is None:
                self.theta = [0] * len(featureVector)

            gradientVector = [0] * len(self.theta)
            actionProbabilties = []
            actionFeatureVectors = []

            for action in self.getLegalActions(self.episodeTriplets[t][0]):
                actionProbabilties.append(self.policy(action, state, self.theta, self.getLegalActions, self.featExtractor))
                actionFeatureVectors.append(self.featExtractor.getFeatures(state, action))

            # x(s,a) - Sum pi(s,*)x(s,*)
            for i in range(len(gradientVector)):
                actionSum = 0
                for j in range(len(actionProbabilties)):
                    actionSum += actionProbabilties[j] * actionFeatureVectors[j][i]

                gradientVector[i] = featureVector[i] - actionSum

            for j in range(len(self.theta)):
                self.theta[j] += self.alpha * pow(self.gamma, t) * gValue * gradientVector[j]

        self.i *= self.gamma


    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        print("Training started.")
        self.episodeTriplets = []
        self.lastState = None
        self.lastAction = None

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        self.update()

        print(f"Episode {self.episodesSoFar} finished")

        self.episodesSoFar += 1
        print("Increment post finish")
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha_theta = 0.0      # no learning
            self.alpha_w = 0.0      # no learning
            print("Training completed.")

    def isInTraining(self):
        print("In training")
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        print("In Test phase")
        return not self.isInTraining()

    ################################
    # Controls needed for Crawler  #
    ################################
    def setEpsilon(self, epsilon):
        self.epsilon = 0

    def setLearningRate(self, alpha):
        self.alpha = 0

    def setDiscount(self, discount):
        self.discount = 0

    def doAction(self,state,action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

    ###################
    # Pacman Specific #
    ###################
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        print("debug print in final")
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()


        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('Actor Critic Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print('\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (
                        trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                print('\tAverage Rewards over testing: %.2f' % testAvg)
            print('\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg))
            print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))
