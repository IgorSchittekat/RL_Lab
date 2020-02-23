"""
Module containing the k-armed bandit problem
Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
We expect all classes to follow the Bandit abstract object formalism.
"""
# -*- coding: utf-8 -*-
import numpy as np
import math

class Bandit(object):
    """
    Abstract concept of a Bandit, i.e. Slot Machine, the Agent can pull.

    A Bandit is a distribution over reals.
    The pull() method samples from the distribution to give out a reward.
    """
    def __init__(self, **kwargs):
        """
        Empty for our simple one-armed bandits, without hyperparameters.
        Parameters
        ----------
        **kwargs: dictionary
            Ignored additional inputs.
        """
        pass

    def reset(self):
        """
        Reinitializes the distribution.
        """
        pass

    def pull(self):
        """
        Returns a sample from the distribution.
        """
        raise NotImplementedError("Calling method pull() in Abstract class Bandit")


class Gaussian_Bandit(Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: the Gaussian_Bandit's distribution is a fixed Gaussian.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def reset(self):
        self.mean = np.random.normal(loc=0, scale=1)

    def pull(self):
        return np.random.normal(loc=self.mean, scale=1)


class Gaussian_Bandit_NonStat(Gaussian_Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: the distribution mean changes each step over time,
    # with increments following N(m=0,std=0.01)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pull(self):
        dist = np.random.normal(loc=self.mean, scale=1)
        self.update_mean()
        return dist

    def update_mean(self):
        self.mean += np.random.normal(loc=0, scale=0.01)


class KBandit(Bandit):
    # TODO: implement this class following the formalism above.
    # Reminder: The k-armed Bandit is a set of k Bandits.
    # In this case we mean for it to be a set of Gaussian_Bandits.
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.set_bandits(k)

    def set_bandits(self, k):
        self.bandits = [Gaussian_Bandit() for _ in range(k)]

    def reset(self):
        for bandit in self.bandits:
            bandit.reset()
        self.best_action = np.argmax([bandit.mean for bandit in self.bandits])  # printing purposes

    def pull(self, action):
        return self.bandits[action].pull()


class KBandit_NonStat(KBandit):
    # TODO: implement this class following the formalism above.
    # Reminder: Same as KBandit, with non stationary Bandits.
    def set_bandits(self, k):
        self.bandits = [Gaussian_Bandit_NonStat() for _ in range(k)]

    def pull(self, action):
        for i, bandit in enumerate(self.bandits):
            if i != action:
                bandit.update_mean()
        dist = self.bandits[action].pull()
        self.best_action = np.argmax([bandit.mean for bandit in self.bandits])  # printing purposes
        return dist

