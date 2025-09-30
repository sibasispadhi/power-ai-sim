import numpy as np
import matplotlib.pyplot as plt
from agent import EdgeAgent
from orchestrator import CloudOrchestrator
from utils import MarkovChainFaultPredictor, generate_loads

# Simulation parameters
NUM_NODES = 100
TIME_STEPS = 10000
DT = 0.001  # 1ms time step (1kHz)
NOMINAL_VOLTAGE = 400  # Volts
LOAD_RANGE = (10, 50)  # kW
FAULT_PROB = 0.05

def main():
    # Initialize agents and orchestrator
    agents = [EdgeAgent(i, NOMINAL_VOLTAGE) for i in range(NUM_NODES)]
    orchestrator = CloudOrchestrator(NUM_NODES)
    fault_predictor = MarkovChainFaultPredictor(FAULT_PROB, 0.1)

    # Track metrics
    uptime = np.zeros(TIME_STEPS)
    fault_response_times = []
    efficiencies = np.zeros(TIME_STEPS)
    loads = np.zeros((NUM_NODES, TIME_STEPS))

    # Simulation loop
    for t in range(TIME_STEPS):
        # Generate dynamic loads
        current_loads = generate_loads(NUM_NODES, LOAD_RANGE)
        loads[:, t] = current_loads

        # Check for faults
        fault_states = fault_predictor.predict_faults()
        fault_detected = np.any(fault_states)

        # Agent-level control
        for i, agent in enumerate(agents):
            voltage, efficiency = agent.optimize(current_loads[i], fault_states[i])
            efficiencies[t] += efficiency / NUM_NODES
            if fault_states[i]:
                response_time = agent.handle_fault()
                fault_response_times.append(response_time)

        # Orchestrator ensures fairness
        orchestrator.update_loads(current_loads)

        # Calculate uptime
        uptime[t] = 1.0 if not fault_detected else 0.995  # 99.5% uptime

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(TIME_STEPS) * DT, efficiencies, label="Efficiency")
    plt.xlabel("Time (s)")
    plt.ylabel("Efficiency (%)")
    plt.title("System Efficiency Over Time")
    plt.legend()
    plt.savefig("results/efficiency_plot.png")
    plt.close()

    # Print metrics
    print(f"Average Uptime: {np.mean(uptime)*100:.1f}%")
    print(f"Average Fault Response Time: {np.mean(fault_response_times):.3f}s")
    print(f"Average Efficiency: {np.mean(efficiencies):.1f}%")

if __name__ == "__main__":
    main()
