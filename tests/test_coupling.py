"""
Tests for the CFD-DEM coupling framework.
"""

import pytest
import numpy as np
from src.cfd.coupling import CFDDEMCoupling

@pytest.fixture
def config():
    """Test configuration."""
    return {
        'cfd': {
            'fluid_density': 1000.0,
            'fluid_viscosity': 1e-3,
            'pressure_gradient': [0.0, 0.0, -9.81],
            'grid_size': [32, 32, 32],
            'domain_size': [1.0, 1.0, 1.0]
        },
        'dem': {
            'particle_radius': 0.01,
            'particle_density': 2500.0,
            'youngs_modulus': 1e9,
            'poisson_ratio': 0.3,
            'restitution_coefficient': 0.5,
            'friction_coefficient': 0.3
        },
        'simulation': {
            'time_step': 1e-4,
            'gravity': [0.0, 0.0, -9.81],
            'initial_particles': 100
        },
        'coupling': {
            'interval': 1,
            'interpolation': 'linear'
        },
        'erosion': {
            'critical_shear_stress': 1.0,
            'erosion_rate_coefficient': 1e-6
        }
    }

@pytest.fixture
def coupling(config):
    """Test coupling instance."""
    return CFDDEMCoupling(config)

def test_initialization(coupling, config):
    """Test coupling initialization."""
    assert coupling.config == config
    assert coupling.coupling_interval == config['coupling']['interval']
    assert coupling.interpolation_method == config['coupling']['interpolation']
    assert coupling.coarse_model is None
    assert coupling.critical_shear_stress == config['erosion']['critical_shear_stress']
    assert coupling.erosion_rate_coefficient == config['erosion']['erosion_rate_coefficient']
    assert len(coupling.eroded_particles) == 0
    assert len(coupling.bond_health_history) == 0

def test_initialization_with_coarse_grained(config):
    """Test coupling initialization with coarse-grained model."""
    config['use_coarse_grained'] = True
    coupling = CFDDEMCoupling(config)
    assert coupling.coarse_model is not None

def test_compute_fluid_forces(coupling):
    """Test fluid force computation."""
    # Test positions and radii
    positions = [np.array([0.5, 0.5, 0.5])]
    radii = [0.01]
    
    # Compute forces
    forces = coupling.compute_fluid_forces(positions, radii)
    
    # Check results
    assert len(forces) == len(positions)
    assert isinstance(forces[0], np.ndarray)
    assert forces[0].shape == (3,)

def test_erosion_force(coupling):
    """Test erosion force computation."""
    # Test parameters
    fluid_velocity = np.array([10.0, 0.0, 0.0])  # High velocity to trigger erosion
    particle_radius = 0.01
    
    # Compute erosion force
    erosion_force = coupling._compute_erosion_force(fluid_velocity, particle_radius)
    
    # Check results
    assert isinstance(erosion_force, np.ndarray)
    assert erosion_force.shape == (3,)
    assert np.all(np.isfinite(erosion_force))
    
    # Test with low velocity (no erosion)
    low_velocity = np.array([0.1, 0.0, 0.0])
    no_erosion_force = coupling._compute_erosion_force(low_velocity, particle_radius)
    assert np.allclose(no_erosion_force, np.zeros(3))

def test_fluid_shear_stress(coupling):
    """Test fluid shear stress computation."""
    # Test parameters
    fluid_velocity = np.array([1.0, 0.0, 0.0])
    particle_radius = 0.01
    
    # Compute shear stress
    shear_stress = coupling._compute_fluid_shear_stress(fluid_velocity, particle_radius)
    
    # Check results
    assert isinstance(shear_stress, float)
    assert shear_stress > 0
    assert np.isfinite(shear_stress)

def test_erosion_tracking(coupling):
    """Test erosion tracking."""
    # Initialize simulation
    coupling.initialize_simulation()
    
    # Create test data
    particle_data = {
        'positions': [np.array([0.5, 0.5, 0.5])],
        'velocities': [np.array([0.0, 0.0, 0.0])]
    }
    fluid_forces = [np.array([2.0, 0.0, 0.0])]  # Force above critical shear stress
    
    # Track erosion
    coupling._track_erosion(particle_data, fluid_forces)
    
    # Check results
    assert len(coupling.eroded_particles) == 1
    assert 'position' in coupling.eroded_particles[0]
    assert 'force' in coupling.eroded_particles[0]
    assert 'time' in coupling.eroded_particles[0]

def test_bond_health_tracking(coupling, config):
    """Test bond health tracking."""
    # Initialize with coarse-grained model
    config['use_coarse_grained'] = True
    coupling = CFDDEMCoupling(config)
    coupling.initialize_simulation()
    
    # Track bond health
    coupling._track_bond_health()
    
    # Check results
    assert len(coupling.bond_health_history) == 1
    assert 0 <= coupling.bond_health_history[0] <= 1

def test_erosion_statistics(coupling):
    """Test erosion statistics computation."""
    # Initialize simulation
    coupling.initialize_simulation()
    
    # Add some eroded particles
    coupling.eroded_particles = [
        {'position': np.array([0.5, 0.5, 0.5]),
         'force': np.array([1.0, 0.0, 0.0]),
         'time': 0.0},
        {'position': np.array([0.6, 0.6, 0.6]),
         'force': np.array([2.0, 0.0, 0.0]),
         'time': 0.1}
    ]
    
    # Get statistics
    stats = coupling.get_erosion_statistics()
    
    # Check results
    assert stats['total_eroded'] == 2
    assert stats['erosion_rate'] > 0
    assert isinstance(stats['average_force'], np.ndarray)
    assert stats['average_force'].shape == (3,)

def test_bond_health_statistics(coupling, config):
    """Test bond health statistics computation."""
    # Initialize with coarse-grained model
    config['use_coarse_grained'] = True
    coupling = CFDDEMCoupling(config)
    coupling.initialize_simulation()
    
    # Add some bond health history
    coupling.bond_health_history = [1.0, 0.9, 0.8]
    
    # Get statistics
    stats = coupling.get_bond_health_statistics()
    
    # Check results
    assert stats['current_health'] == 0.8
    assert stats['degradation_rate'] > 0

def test_simulation_step(coupling):
    """Test simulation step."""
    # Initialize simulation
    coupling.initialize_simulation()
    
    # Perform step
    coupling.step()
    
    # Check that erosion tracking is working
    assert hasattr(coupling, 'eroded_particles')
    assert hasattr(coupling, 'bond_health_history')

def test_error_handling(coupling):
    """Test error handling."""
    # Test with invalid position
    with pytest.raises(Exception):
        coupling._interpolate_fluid_velocity([np.array([-1.0, -1.0, -1.0])])
    
    # Test with invalid radius
    with pytest.raises(Exception):
        coupling._compute_drag_force(np.array([1.0, 0.0, 0.0]), -0.01)
    
    # Test with invalid fluid velocity
    with pytest.raises(Exception):
        coupling._compute_erosion_force(np.array([np.inf, 0.0, 0.0]), 0.01) 