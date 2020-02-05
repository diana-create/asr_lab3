#!/usr/bin/env python3

import numpy as np

class ObservationModel():
    def __init__(self):
        self.timesteps = 0
        self.dummy = False

    def load_dummy_audio(self):
        self.timesteps = 30
        self.dummy = True
        self.hmm_labels = []
        with open('phonelist.txt', 'r') as f:
            for ph in f.readlines():
                for i in range(1, 4):
                    self.hmm_labels.append("{}_{}".format(ph.strip(),i))

    def observation_length(self):
        return self.timesteps

    def log_observation_probability(self, hmm_label, t):
        if t <= 0 or t > self.timesteps:
            raise IndexError("Timestep not in range [1,{}]".format(self.timesteps))
        if self.dummy:
            return self.dummy_observation_probability(hmm_label, t)
        else:
            raise NotImplementedError()

    def dummy_observation_probability(self, hmm_label, t):
        """ Computes b_j(t) where j is the current state

        This is just a dummy version!  In later labs we'll generate
        probabilities for real speech frames.

        You don't need to look at this function in detail.

        Args: hmm_label (str): the HMM state label, j.  We'll use string form: "p_1", "p_2", "eh_1" etc
              t (int) : current time step, starting at 1

        Returns:
              p (float): the observation probability p(x_t | q_t = hmm_label)
        """

        p = {}  # dictionary of probabilities

        assert(t > 0)

        # this is just a simulation!
        if t < 4:
            p = {'p_1': 1.0, 'p_2': 1.0, 'p_3': 1.0, 'eh_1': 0.2}
        elif t < 9:
            p = {'p_3': 0.5, 'eh_1': 1.0, 'eh_2': 1.0, 'eh_3': 1.0}
        elif t < 13:
            p = {'eh_3': 1.0, 'p_1': 1.0, 'p_2': 1.0, 'p_3': 1.0, 'er_1': 0.5}
        elif t < 18:
            p = {'p_3': 1.0, 'er_1': 1.0, 'er_2': 1.0, 'er_3': 0.7}
        elif t < 25:
            p = {'er_3': 1.0, 'z_1': 1.0, 'z_2': 1.0, 'z_3': 1.0}
        else:
            p = {'z_2': 0.5, 'z_3': 1.0}

        for label in self.hmm_labels:
            if label not in p:
                p[label] = 0.001  # give all other states a small probability to avoid zero probability

        # normalise the probabilities:
        scale = sum(p.values())
        for k in p:
            p[k] = p[k]/scale

        return np.log(p[hmm_label])
