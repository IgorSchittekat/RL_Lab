"""
Module containing the agent classes to solve a Bandit problem.

Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
An example can be seen on the Bandit_Agent and Random_Agent classes.
"""
# -*- coding: utf-8 -*-
import numpy as np
from utils import softmax, my_random_choice

class Bandit_Agent(object):
    """
    Abstract Agent to solve a Bandit problem.

    Contains the methods learn() and act() for the base life cycle of an agent.
    The reset() method reinitializes the agent.
    The minimum requirment to instantiate a child class of Bandit_Agent
    is that it implements the act() method (see Random_Agent).
    """
    def __init__(self, k:int, **kwargs):
        """
        Simply stores the number of arms of the Bandit problem.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        k: positive int
            Number of arms of the Bandit problem.
        kwargs: dictionary
            Additional parameters, ignored.
        """
        self.k = k

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        pass

    def learn(self, a:int, r:float):
        """
        Learning method. The agent learns that action a yielded reward r.
        Parameters
        ----------
        a: positive int < k
            Action that yielded the received reward r.
        r: float
            Reward for having performed action a.
        """
        pass

    def act(self) -> int:
        """
        Agent's method to select a lever (or Bandit) to pull.
        Returns
        -------
        a : positive int < k
            The action the agent chose to perform.
        """
        raise NotImplementedError("Calling method act() in Abstract class Bandit_Agent")


class Random_Agent(Bandit_Agent):
    """
    This agent doesn't learn, just acts purely randomly.
    Good baseline to compare to other agents.
    """
    def act(self):
        """
        Random action selection.
        Returns
        -------
        a : positive int < k
            A randomly selected action.
        """
        return np.random.randint(self.k)


class EpsGreedy(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # Non stationary agent with q estimating and eps-greedy action selection.
    def __init__(self, eps, lr, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.alpha = lr
        self.reset()

    def reset(self):
        self.means = [0] * self.k

    def act(self):
        rand = np.random.random_sample()
        if rand >= self.eps:
            action = np.argmax([mean for mean in self.means])
        else:
            action = np.random.randint(self.k)
        return action

    def learn(self, a, r):
        self.means[a] += self.alpha * (r - self.means[a])


class EpsGreedy_SampleAverage(EpsGreedy):
    # TODO: implement this class following the formalism above.
    # This class uses Sample Averages to estimate q; others are non stationary.
    def reset(self):
        super().reset()
        self.n = [0] * self.k

    def learn(self, a, r):
        self.n[a] += 1
        self.means[a] += 1 / self.n[a] * (r - self.means[a])


class OptimisticGreedy(EpsGreedy):
    # TODO: implement this class following the formalism above.
    # Same as above but with optimistic starting values.
    def __init__(self, q0, **kwargs):
        self.q0 = q0
        super().__init__(**kwargs)

    def reset(self):
        self.means = [self.q0] * self.k


class UCB(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    def __init__(self, c, lr, **kwargs):
        super().__init__(**kwargs)
        self.c = c
        self.alpha = lr

    def reset(self):
        self.means = [0] * self.k
        self.N = [0] * self.k
        self.time = 0

    def act(self):
        if self.time < self.k:
            action = self.time
        else:
            action = np.argmax([(mean + self.c * np.sqrt(np.log(self.time) / n)) for mean, n in zip(self.means, self.N)])
        self.N[action] += 1
        self.time += 1
        return action

    def learn(self, a, r):
        self.means[a] += self.alpha * (r - self.means[a])


class Gradient_Bandit(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # If you want this to run fast, use the my_random_choice function from
    # utils instead of np.random.choice to sample from the softmax
    # You can also find the softmax function in utils.
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.H = [0] * self.k
        self.time = 0
        self.mean_r = 0
        self.probs= [0] * self.k

    def act(self):
        self.probs = softmax(self.H)
        return my_random_choice(len(self.H), self.probs)

    def learn(self, a, r):
        self.time += 1
        self.mean_r += (r - self.mean_r) / self.time

        for i in range(len(self.H)):
            if i == a:
                self.H[i] += self.alpha * (r - self.mean_r) * (1 - self.probs[i])
            else:
                self.H[i] -= self.alpha * (r - self.mean_r) * self.probs[i]
