import numpy as np

class CloudOrchestrator:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.mean_throughput = 0

    def update_loads(self, loads):
        # Ensure fairness: throughput within ±5% of mean
        self.mean_throughput = np.mean(loads)
        adjusted_loads = np.clip(loads, 0.95 * self.mean_throughput, 1.05 * self.mean_throughput)
        return adjusted_loads
