
import os

class Solver():
    def __init__(self, config):
        self.config = config

    @staticmethod
    def safe_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

