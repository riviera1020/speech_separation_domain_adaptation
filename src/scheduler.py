import numpy as np

class RampScheduler(object):

    def __init__(self, start_step, end_step, start_value, end_value, now_step = 0):

        self.now_step = now_step
        self.ss = start_step
        self.es = end_step

        self.sv = start_value
        self.ev = end_value

        self.s = float(end_value - start_value) / (end_step - start_step)

    def value(self, step):

        self.now_step = step

        if step <= self.ss:
            return self.sv
        elif step >= self.es:
            return self.ev
        else:
            return self.s * (step - self.ss) + self.sv

class ConstantScheduler(object):
    def __init__(self, value):
        self.v = value

    def value(self, step):
        return self.v

class DANNScheduler(object):
    def __init__(self, gamma, scale, total_step):
        self.gamma = gamma
        self.scale = scale
        self.total_step = total_step

    def value(self, step):

        p = float(step) / self.total_step
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        return self.scale * alpha
