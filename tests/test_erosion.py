"""
Tests for the erosion model component.
"""

import pytest
import numpy as np
from src.erosion.model import ErosionModel

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'erosion': {
            'critical_shear_stress': 0.1,
            'erosion_rate_coefficient': 1e-6,
            'transport_capacity': 1.0,
            'deposition_rate': 0.1
        }
    }

def test_erosion_model_initialization(config):
    """Test erosion model initialization."""
    model = ErosionModel(config)
    assert model.critical_shear_stress == config['erosion']['critical_shear_stress']
    assert model.erosion_rate_coefficient == config['erosion']['erosion_rate_coefficient']
    assert model.transport_capacity == config['erosion']['transport_capacity']
    assert model.deposition_rate == config['erosion']['deposition_rate']

def test_shear_stress_computation(config):
    """Test shear stress computation."""
    model = ErosionModel(config)
    fluid_velocity = np.array([1.0, 0.0, 0.0])
    particle_radius = 0.01
    shear_stress = model.compute_shear_stress(fluid_velocity, particle_radius)
    assert isinstance(shear_stress, float)
    assert shear_stress >= 0.0

def test_erosion_rate_computation(config):
    """Test erosion rate computation."""
    model = ErosionModel(config)
    
    # Test below critical shear stress
    shear_stress_below = 0.05
    rate_below = model.compute_erosion_rate(shear_stress_below)
    assert rate_below == 0.0
    
    # Test above critical shear stress
    shear_stress_above = 0.15
    rate_above = model.compute_erosion_rate(shear_stress_above)
    expected_rate = model.erosion_rate_coefficient * (shear_stress_above - model.critical_shear_stress)
    assert rate_above == expected_rate

def test_transport_computation(config):
    """Test particle transport computation."""
    model = ErosionModel(config)
    particle_mass = 1.0
    fluid_velocity = np.array([1.0, 0.0, 0.0])
    transport_rate = model.compute_transport(particle_mass, fluid_velocity)
    assert isinstance(transport_rate, float)
    assert transport_rate >= 0.0

def test_deposition_computation(config):
    """Test particle deposition computation."""
    model = ErosionModel(config)
    particle_mass = 1.0
    fluid_velocity = np.array([0.1, 0.0, 0.0])
    deposition_rate = model.compute_deposition(particle_mass, fluid_velocity)
    assert isinstance(deposition_rate, float)
    assert deposition_rate >= 0.0

def test_particle_mass_update(config):
    """Test particle mass update."""
    model = ErosionModel(config)
    initial_mass = 1.0
    erosion_rate = 0.1
    transport_rate = 0.05
    deposition_rate = 0.02
    time_step = 0.001
    
    new_mass = model.update_particle_mass(
        initial_mass,
        erosion_rate,
        transport_rate,
        deposition_rate,
        time_step
    )
    
    expected_change = (erosion_rate - transport_rate - deposition_rate) * time_step
    assert abs(new_mass - (initial_mass + expected_change)) < 1e-10
