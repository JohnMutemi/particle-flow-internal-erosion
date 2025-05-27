#!/usr/bin/env python3
"""
Script to run the water penetration test and visualize results.
"""

from src.particle_flow.validation.water_penetration_test import WaterPenetrationTest
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_test():
    # Get config file path
    config_path = os.path.join(
        project_root, "data/input/water_penetration_test/test_parameters.yaml")

    # Initialize test
    test = WaterPenetrationTest(config_path)

    # Run test
    print("Starting water penetration test...")
    results = test.run_test(duration=test.test_conditions['duration'],
                            time_step=test.test_conditions['time_step'])

    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_root, "results/water_penetration_test")
    os.makedirs(results_dir, exist_ok=True)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(results_dir,
            f"test_results_{timestamp}.npy"), results)

    # Generate and save plots
    plt.figure(figsize=(15, 10))

    # Pressure and flow rate
    plt.subplot(2, 2, 1)
    plt.plot(results['time'], results['pressure'], 'b-', label='Pressure')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (Pa)')
    plt.title('Pressure vs Time')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(results['time'], results['flow_rate'], 'r-', label='Flow Rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Flow Rate (m³/s)')
    plt.title('Flow Rate vs Time')
    plt.grid(True)

    # Erosion rate and porosity
    plt.subplot(2, 2, 3)
    plt.plot(results['time'], results['erosion_rate'],
             'g-', label='Erosion Rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Erosion Rate (kg/s)')
    plt.title('Erosion Rate vs Time')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(results['time'], results['porosity'], 'm-', label='Porosity')
    plt.xlabel('Time (s)')
    plt.ylabel('Porosity')
    plt.title('Porosity vs Time')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"test_results_{timestamp}.png"))
    plt.close()

    print(f"Test completed. Results saved to {results_dir}")
    print("\nTest Summary:")
    print(f"Final Pressure: {results['pressure'][-1]:.2f} Pa")
    print(f"Final Flow Rate: {results['flow_rate'][-1]:.2e} m³/s")
    print(f"Final Erosion Rate: {results['erosion_rate'][-1]:.2e} kg/s")
    print(f"Final Porosity: {results['porosity'][-1]:.3f}")


if __name__ == "__main__":
    run_test()
