"""
Demo script for running and demonstrating the main simulation and post-processing workflows.

- Demonstrates the new constitutive model, coarse-grained model, CFD-DEM coupling, and validation framework.
- Loads parameters from config files and uses personal geotechnical values.
- Generates and saves plots to the results directory.

Usage:
    python scripts/demo.py

References:
- data/input/triaxial_test/test_parameters.yaml
- config.yaml
- src/models/bond_model.py, src/coarse_grained/coarse_grained_model.py, src/coupling/coupling_manager.py, src/validation/validation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from src.models.bond_model import SeepageBondModel
from src.coarse_grained.coarse_grained_model import CoarseGrainedModel
from src.coupling.coupling_manager import CouplingManager
from src.validation.validation import ValidationManager
from src.case_studies.tunnel_water_inrush import TunnelWaterInrush
from src.visualization.visualizer import SimulationVisualizer
from scipy.interpolate import interp1d

def demonstrate_bond_model():
    """Demonstrate the new constitutive model for particle bonding"""
    print("\n1. Demonstrating New Constitutive Model...")
    
    # Create bond model
    config = {
        'dem': {'bond_strength': 1.0e6},
        'cfd': {
            'fluid_viscosity': 0.001,
            'fluid_density': 1000.0
        },
        'erosion': {
            'erosion_rate_coefficient': 1.0e-6,
            'critical_shear_stress': 1.0
        },
        'geotechnical': {
            'clay_content': 10.02,
            'water_content': 8.09,
            'Cu': 7.9,
            'Cc': 1.51,
            'cohesion': 17.53,
            'permeability': 1.1e-6,
            'density': 1.975,
            'specific_gravity': 2.65
        }
    }
    bond_model = SeepageBondModel(config)
    
    # Simulate bond degradation
    time_steps = np.linspace(0, 10, 100)
    dt = time_steps[1] - time_steps[0]  # Time step size
    
    # Create 3D velocity and pressure gradient arrays
    fluid_velocity = np.array([[0.5 * np.sin(t), 0.0, 0.0] for t in time_steps])
    pressure_gradient = np.array([[1.0e5, 0.0, 0.0] for _ in time_steps])
    
    bond_radius = 0.01
    current_strength = 1.0e6
    
    bond_strength = []
    for t, v, p in zip(time_steps, fluid_velocity, pressure_gradient):
        _, strength = bond_model.compute_bond_degradation(
            v,  # 3D velocity vector
            p,  # 3D pressure gradient vector
            bond_radius,
            current_strength,
            dt  # Use constant time step
        )
        bond_strength.append(strength)
        current_strength = strength
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, bond_strength, 'b-', label='Bond Strength')
    plt.plot(time_steps, fluid_velocity[:, 0], 'r--', label='Fluid Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Strength (Pa)')
    plt.title('Bond Degradation Under Fluid Flow')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/bond_degradation.png')
    plt.close()
    
    print("✓ Bond model demonstration complete")

def demonstrate_coarse_grained():
    """Demonstrate the coarse-grained model"""
    print("\n2. Demonstrating Coarse-Grained Model...")
    
    # Create configuration
    config = {
        'dem': {
            'particle_radius': 0.01,
            'bond_strength': 1.0e6
        },
        'cfd': {
            'fluid_viscosity': 0.001,
            'fluid_density': 1000.0
        },
        'coarse_grained': {
            'length_scale': 10.0,
            'time_scale': 5.0,
            'force_scale': 8.0,
            'calibration_file': 'data/calibrated_params.json'
        },
        'output': {
            'directory': 'results'
        },
        'geotechnical': {
            'clay_content': 10.02,
            'water_content': 8.09,
            'Cu': 7.9,
            'Cc': 1.51,
            'cohesion': 17.53,
            'permeability': 1.1e-6,
            'density': 1.975,
            'specific_gravity': 2.65,
            'porosity': 0.31,
            'void_ratio': 0.45
        }
    }
    
    # Create fine-scale data
    fine_scale_data = {
        'positions': np.random.rand(1000, 3),
        'velocities': np.random.rand(1000, 3),
        'forces': np.random.rand(1000, 3),
        'fluid_fields': {
            'velocity': np.random.rand(101, 51, 51, 3),
            'pressure': np.random.rand(101, 51, 51)
        }
    }
    
    # Create coarse-grained model
    coarse_model = CoarseGrainedModel(config)
    
    # Map to coarse scale
    coarse_data = coarse_model.map_fine_to_coarse(fine_scale_data)
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(fine_scale_data['positions'][:, 0], 
                fine_scale_data['positions'][:, 1], 
                c='blue', alpha=0.3, label='Fine Scale')
    plt.scatter(coarse_data['positions'][:, 0], 
                coarse_data['positions'][:, 1], 
                c='red', s=100, label='Coarse Scale')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Fine vs Coarse Scale Representation')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/coarse_grained.png')
    plt.close()
    
    print("✓ Coarse-grained model demonstration complete")

def demonstrate_coupling():
    """Demonstrate the CFD-DEM coupling"""
    print("\n3. Demonstrating CFD-DEM Coupling...")
    
    # Create configuration
    config = {
        'coupling': {
            'interval': 1,
            'interpolation_order': 1,
            'force_coupling': True,
            'heat_coupling': False,
            'mass_coupling': False
        },
        'output': {
            'directory': 'results'
        }
    }
    
    # Create coupling manager
    coupling_manager = CouplingManager(config)
    
    # Create sample data
    particles = {
        'positions': np.random.rand(100, 3),
        'velocities': np.random.rand(100, 3),
        'radii': np.random.rand(100) * 0.01
    }
    
    fluid_fields = {
        'velocity': np.random.rand(101, 51, 51, 3),
        'pressure': np.random.rand(101, 51, 51),
        'density': np.ones((101, 51, 51)) * 1000.0,
        'viscosity': np.ones((101, 51, 51)) * 0.001
    }
    
    # Compute coupling forces
    forces = coupling_manager.compute_coupling_forces(particles, fluid_fields)
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.quiver(particles['positions'][:, 0], particles['positions'][:, 1],
               forces['drag'][:, 0], forces['drag'][:, 1], scale=50)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Fluid-Particle Coupling Forces')
    plt.grid(True)
    plt.savefig('results/coupling_forces.png')
    plt.close()
    
    print("✓ CFD-DEM coupling demonstration complete")

def demonstrate_validation():
    """Demonstrate validation framework with geotechnical parameters."""
    print("\n4. Demonstrating Validation Framework...")
    
    # Create configuration
    config = {
        'validation': {
            'experimental_data_file': 'data/experimental/triaxial_test/test_data.csv',
            'data_type': 'triaxial',
            'variables_to_compare': ['erosion_rate', 'pressure', 'velocity'],
            'statistics': {
                'compute_basic': True,
                'compute_distribution': True,
                'compute_correlation': True
            }
        },
        'output': {
            'directory': 'results'
        }
    }
    
    # Create validation manager
    validation_manager = ValidationManager(config)
    
    # Load experimental data
    validation_manager.load_experimental_data(
        config['validation']['experimental_data_file'],
        config['validation']['data_type']
    )
    
    # Create sample simulation results (replace with actual simulation results)
    simulation_results = {
        'erosion_stats': np.random.normal(0.5, 0.1, 100),  # Replace with actual simulation data
        'fluid_data': {
            'velocity_field': np.random.rand(101, 51, 51, 3),
            'pressure_field': np.random.rand(101, 51, 51)
        },
        'particle_data': np.random.rand(1000, 3)
    }
    
    # Interpolate simulated data to match experimental time points
    exp_data = validation_manager.validation_data['experimental']
    sim_time_erosion = np.linspace(0, 0.5, 100)  # 100 points for erosion_stats
    sim_time_fields = np.linspace(0, 0.5, 101)   # 101 points for pressure/velocity fields
    
    # Interpolate erosion rates
    erosion_interp = interp1d(sim_time_erosion, simulation_results['erosion_stats'])
    simulation_results['erosion_stats'] = erosion_interp(exp_data['time'])
    
    # Interpolate pressure field
    pressure_interp = interp1d(sim_time_fields, np.mean(simulation_results['fluid_data']['pressure_field'], axis=(1,2)))
    simulation_results['fluid_data']['pressure_field'] = pressure_interp(exp_data['time'])
    
    # Interpolate velocity field
    velocity_interp = interp1d(sim_time_fields, np.mean(simulation_results['fluid_data']['velocity_field'], axis=(1,2,3)))
    simulation_results['fluid_data']['velocity_field'] = velocity_interp(exp_data['time'])
    
    # Compare with experimental data
    exp_data = validation_manager.validation_data['experimental']
    print("\nDebug Information:")
    print(f"Experimental data shape: {exp_data['erosion_rate'].shape}")
    print(f"Experimental time points: {exp_data['time']}")
    print(f"Experimental erosion rates: {exp_data['erosion_rate']}")
    print(f"Simulated data shape: {simulation_results['erosion_stats'].shape}")
    print(f"Simulated erosion rates (first 10): {simulation_results['erosion_stats'][:10]}")
    
    comparison = validation_manager.compare_with_experimental(simulation_results)
    
    # Perform statistical analysis
    validation_manager.perform_statistical_analysis(simulation_results)
    
    # Plot results
    plt.style.use('ggplot')  # Using ggplot style instead of seaborn
    validation_manager.plot_validation_results()
    
    print("✓ Validation framework demonstration complete")
    print(f"  - Detailed comparison plots saved to results/validation_comparison.png")
    print(f"  - Validation statistics saved to results/validation/")

def main():
    """Run all demonstrations"""
    print("Starting Project Demonstration...")
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run demonstrations (excluding tunnel water inrush)
    demonstrate_bond_model()
    demonstrate_coarse_grained()
    demonstrate_coupling()
    demonstrate_validation()
    
    print("\nDemonstration Complete!")
    print("Results have been saved to the 'results' directory")

if __name__ == "__main__":
    main() 