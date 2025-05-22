import numpy as np
import matplotlib.pyplot as plt
from src.models.bond_model import SeepageBondModel
from src.coarse_grained.coarse_grained_model import CoarseGrainedModel
from src.coupling.coupling_manager import CouplingManager
from src.validation.validation_manager import ValidationManager
from src.case_studies.tunnel_water_inrush import TunnelWaterInrush
from src.visualization.visualizer import SimulationVisualizer

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
        }
    }
    bond_model = SeepageBondModel(config)
    
    # Simulate bond degradation
    time_steps = np.linspace(0, 10, 100)
    fluid_velocity = np.array([0.5 * np.sin(t) for t in time_steps])
    pressure_gradient = np.array([1.0e5] * len(time_steps))
    bond_radius = 0.01
    current_strength = 1.0e6
    
    bond_strength = []
    for t, v, p in zip(time_steps, fluid_velocity, pressure_gradient):
        _, strength = bond_model.compute_bond_degradation(
            np.array([v, 0, 0]),
            np.array([p, 0, 0]),
            bond_radius,
            current_strength
        )
        bond_strength.append(strength)
        current_strength = strength
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, bond_strength, 'b-', label='Bond Strength')
    plt.plot(time_steps, fluid_velocity, 'r--', label='Fluid Velocity')
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
    """Demonstrate the validation framework"""
    print("\n4. Demonstrating Validation Framework...")
    
    # Create configuration
    config = {
        'validation': {
            'experimental_data_file': 'data/experimental_data.npz',
            'variables_to_compare': ['erosion_rate', 'pressure', 'velocity'],
            'statistics': {
                'compute_basic': True,
                'compute_distribution': True,
                'compute_correlation': True
            },
            'sensitivity_analysis': {
                'parameter_ranges': {
                    'bond_strength': [0.5e6, 2.0e6],
                    'critical_shear_stress': [0.5, 2.0],
                    'erosion_rate_coefficient': [0.5e-6, 2.0e-6]
                },
                'num_samples': 10,
                'compute_interactions': True
            }
        },
        'output': {
            'directory': 'results'
        }
    }
    
    # Create validation manager
    validation_manager = ValidationManager(config)
    
    # Create sample simulation results
    simulation_results = {
        'erosion_rate': np.random.rand(100),
        'pressure': np.random.rand(100),
        'velocity': np.random.rand(100),
        'particle_data': {
            'positions': np.random.rand(100, 3),
            'velocities': np.random.rand(100, 3),
            'forces': np.random.rand(100, 3)
        },
        'fluid_fields': {
            'velocity': np.random.rand(101, 51, 51, 3),
            'pressure': np.random.rand(101, 51, 51)
        }
    }
    
    # Load experimental data
    experimental_data = validation_manager.load_experimental_data()
    
    # Compare with experimental data
    comparison = validation_manager.compare_with_experimental(
        simulation_results, experimental_data)
    
    # Perform statistical analysis
    statistics = validation_manager.perform_statistical_analysis(simulation_results)
    
    # Perform sensitivity analysis
    sensitivity = validation_manager.perform_sensitivity_analysis(
        simulation_results,
        ['bond_strength', 'critical_shear_stress', 'erosion_rate_coefficient']
    )
    
    # Plot results
    validation_manager.plot_validation_results(comparison, statistics, sensitivity)
    
    print("✓ Validation framework demonstration complete")

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
    print("\nNote: Tunnel water inrush demonstration was skipped due to ongoing optimization.")

if __name__ == "__main__":
    main() 