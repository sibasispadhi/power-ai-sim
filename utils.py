import numpy as np

class MarkovChainFaultPredictor:
    def __init__(self, fault_prob, recovery_prob):
        self.fault_prob = fault_prob
        self.recovery_prob = recovery_prob
        self.state = np.zeros(100)  # 100 nodes, 0=normal, 1=faulty

    def predict_faults(self):
        new_state = self.state.copy()
        for i in range(len(self.state)):
            if self.state[i] == 0:
                if np.random.rand() < self.fault_prob:
                    new_state[i] = 1
            else:
                if np.random.rand() < self.recovery_prob:
                    new_state[i] = 0
        self.state = new_state
        return self.state

def generate_loads(num_nodes, load_range):
    return np.random.uniform(load_range[0], load_range[1], num_nodes)
