"""
Tests for the main simulation module.
"""

import pytest
import yaml
from pathlib import Path
from particle_flow.main import load_config, initialize_simulation, run_simulation


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config file for testing."""
    config = {
        'simulation': {
            'time_steps': 10,
            'output_interval': 2,
            'domain_size': [1.0, 1.0, 1.0],
            'initial_particles': 5,
            'gravity': [0.0, -9.81, 0.0],
            'time_step': 0.001
        },
        'dem': {
            'particle_radius': 0.01,
            'particle_density': 2650.0,
            'youngs_modulus': 1e9,
            'poisson_ratio': 0.3,
            'friction_coefficient': 0.5,
            'restitution_coefficient': 0.5
        },
        'cfd': {
            'fluid_density': 1000.0,
            'fluid_viscosity': 1e-6,
            'pressure_gradient': [0.0, 0.0, 0.0],
            'boundary_conditions': {
                'type': 'no-slip',
                'velocity': [0.0, 0.0, 0.0]
            }
        },
        'erosion': {
            'critical_shear_stress': 0.1,
            'erosion_rate_coefficient': 1e-6,
            'transport_capacity': 1.0,
            'deposition_rate': 0.1
        },
        'output': {
            'directory': 'test_results',
            'save_particles': True,
            'save_fluid': True,
            'visualization_interval': 2
        }
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path


def test_config_loading(config_file):
    """Test configuration loading from file."""
    config = load_config(config_file)
    assert config['simulation']['time_steps'] == 10
    assert config['simulation']['initial_particles'] == 5
    assert config['dem']['particle_radius'] == 0.01
    assert config['cfd']['fluid_density'] == 1000.0
    assert config['erosion']['critical_shear_stress'] == 0.1


def test_simulation_initialization(config_file):
    """Test simulation initialization."""
    config = load_config(config_file)
    dem_solver, cfd_solver, erosion_model = initialize_simulation(config)

    # Check that all components are initialized
    assert dem_solver is not None
    assert cfd_solver is not None
    assert erosion_model is not None

    # Check particle initialization
    particle_data = dem_solver.get_particle_data()
    assert len(particle_data['positions']
               ) == config['simulation']['initial_particles']

    # Check fluid field initialization
    fluid_data = cfd_solver.get_fluid_data()
    assert fluid_data['velocity_field'] is not None
    assert fluid_data['pressure_field'] is not None


def test_simulation_run(config_file):
    """Test running the simulation for a few steps."""
    config = load_config(config_file)
    dem_solver, cfd_solver, erosion_model = initialize_simulation(config)

    # Run simulation for a few steps
    run_simulation(dem_solver, cfd_solver, erosion_model, config)

    # Check that simulation progressed
    particle_data = dem_solver.get_particle_data()
    fluid_data = cfd_solver.get_fluid_data()

    # Add more specific assertions once the simulation components are fully implemented
