"""
Test configuration and fixtures.
"""

import pytest
import numpy as np

@pytest.fixture
def config():
    """Test configuration."""
    return {
        'simulation': {
            'time_step': 0.1,
            'total_time': 1.0,
            'domain_size': [10.0, 5.0, 5.0],
            'grid_resolution': [50, 25, 25]
        },
        'cfd': {
            'fluid_density': 1000.0,
            'fluid_viscosity': 1e-3,
            'pressure_gradient': np.array([-1000.0, 0.0, 0.0]),
            'domain_size': [10.0, 5.0, 5.0]
        },
        'dem': {
            'particle_radius': 0.1,
            'particle_density': 2500.0,
            'bond_strength': 1e6,
            'gravity': np.array([0.0, 0.0, -9.81]),
            'youngs_modulus': 1e9,
            'poisson_ratio': 0.3
        },
        'erosion': {
            'critical_shear_stress': 1.0,
            'erosion_rate_coefficient': 1e-6
        },
        'case_study': {
            'water_pressure': 1e6,
            'mud_concentration': 0.3
        }
    }

@pytest.fixture
def sample_particle_data():
    """Sample particle data for testing."""
    return {
        'positions': np.array([
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6]
        ]),
        'velocities': np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0]
        ]),
        'forces': np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
    }

@pytest.fixture
def sample_fluid_data():
    """Sample fluid data for testing."""
    return {
        'velocity_field': np.ones((10, 10, 10, 3)) * 0.1,
        'pressure_field': np.ones((10, 10, 10)) * 1e5,
        'density_field': np.ones((10, 10, 10)) * 1000.0
    }

@pytest.fixture
def sample_erosion_data():
    """Sample erosion data for testing."""
    return {
        'total_eroded': 5,
        'erosion_rate': 1.0,
        'average_force': np.array([1.0, 0.0, 0.0]),
        'eroded_positions': np.array([
            [0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6]
        ])
    }

@pytest.fixture
def sample_bond_health_data():
    """Sample bond health data for testing."""
    return {
        'current_health': 0.8,
        'degradation_rate': 0.1,
        'bond_strength': 1e6
    } 