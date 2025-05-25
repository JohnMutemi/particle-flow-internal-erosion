"""
Tests for the DEM solver component.
"""

import pytest
import numpy as np
from particle_flow.dem.solver import DEMSolver


@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'simulation': {
            'time_step': 0.001,
            'gravity': [0.0, -9.81, 0.0],
            'domain_size': [1.0, 1.0, 1.0]
        },
        'dem': {
            'particle_radius': 0.01,
            'particle_density': 2650.0,
            'youngs_modulus': 1e9,
            'poisson_ratio': 0.3,
            'friction_coefficient': 0.5,
            'restitution_coefficient': 0.5
        }
    }


def test_dem_solver_initialization(config):
    """Test DEM solver initialization."""
    solver = DEMSolver(config)
    assert solver.time_step == config['simulation']['time_step']
    assert np.array_equal(solver.gravity, np.array(
        config['simulation']['gravity']))
    assert solver.particle_radius == config['dem']['particle_radius']
    assert solver.particle_density == config['dem']['particle_density']


def test_particle_initialization(config):
    """Test particle initialization."""
    solver = DEMSolver(config)
    num_particles = 10
    solver.initialize_particles(num_particles)
    particle_data = solver.get_particle_data()
    assert len(particle_data['positions']) == num_particles
    assert len(particle_data['velocities']) == num_particles
    assert len(particle_data['forces']) == num_particles


def test_contact_force_computation(config):
    """Test contact force computation."""
    solver = DEMSolver(config)
    forces = solver.compute_contact_forces()
    assert isinstance(forces, list)
    # Add more specific assertions once contact force computation is implemented


def test_particle_state_update(config):
    """Test particle state update."""
    solver = DEMSolver(config)
    solver.initialize_particles(5)
    initial_positions = solver.get_particle_data()['positions'].copy()
    solver.update_particle_states()
    updated_positions = solver.get_particle_data()['positions']
    # Add more specific assertions once state update is implemented
