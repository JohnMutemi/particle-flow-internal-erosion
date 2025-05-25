"""
Tests for the CFD solver component.
"""

import pytest
import numpy as np
from particle_flow.cfd.solver import CFDSolver


@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'simulation': {
            'time_step': 0.001,
            'domain_size': [1.0, 1.0, 1.0]
        },
        'cfd': {
            'fluid_density': 1000.0,
            'fluid_viscosity': 1e-6,
            'pressure_gradient': [0.0, 0.0, 0.0],
            'boundary_conditions': {
                'type': 'no-slip',
                'velocity': [0.0, 0.0, 0.0]
            }
        }
    }


def test_cfd_solver_initialization(config):
    """Test CFD solver initialization."""
    solver = CFDSolver(config)
    assert solver.fluid_density == config['cfd']['fluid_density']
    assert solver.fluid_viscosity == config['cfd']['fluid_viscosity']
    assert np.array_equal(solver.pressure_gradient, np.array(
        config['cfd']['pressure_gradient']))
    assert solver.time_step == config['simulation']['time_step']


def test_field_initialization(config):
    """Test fluid field initialization."""
    solver = CFDSolver(config)
    solver.initialize_fields()
    fluid_data = solver.get_fluid_data()
    assert fluid_data['velocity_field'] is not None
    assert fluid_data['pressure_field'] is not None


def test_fluid_force_computation(config):
    """Test fluid force computation on particles."""
    solver = CFDSolver(config)
    solver.initialize_fields()
    particle_positions = [np.array([0.5, 0.5, 0.5])]
    forces = solver.compute_fluid_forces(particle_positions)
    assert isinstance(forces, list)
    assert len(forces) == len(particle_positions)


def test_fluid_state_update(config):
    """Test fluid state update."""
    solver = CFDSolver(config)
    solver.initialize_fields()
    initial_velocity = solver.get_fluid_data()['velocity_field'].copy()
    solver.update_fluid_state()
    updated_velocity = solver.get_fluid_data()['velocity_field']
    # Add more specific assertions once fluid state update is implemented
