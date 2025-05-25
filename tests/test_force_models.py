"""
Tests for the force models module.
"""

import pytest
import numpy as np
from src.cfd.force_models import ForceModels


@pytest.fixture
def config():
    """Test configuration."""
    return {
        'cfd': {
            'fluid_density': 1000.0,
            'fluid_viscosity': 1e-3
        },
        'dem': {
            'particle_radius': 0.1
        }
    }


@pytest.fixture
def force_models(config):
    """Test force models instance."""
    return ForceModels(config)


def test_initialization(force_models, config):
    """Test force models initialization."""
    assert force_models.fluid_density == config['cfd']['fluid_density']
    assert force_models.fluid_viscosity == config['cfd']['fluid_viscosity']
    assert force_models.particle_radius == config['dem']['particle_radius']
    assert force_models.reynolds_threshold == 1000
    assert force_models.mach_threshold == 0.3


def test_drag_force_stokes(force_models):
    """Test drag force in Stokes flow regime."""
    particle_velocity = np.array([0.0, 0.0, 0.0])
    fluid_velocity = np.array([0.1, 0.0, 0.0])  # Low velocity for Stokes flow
    particle_radius = 0.1

    drag_force = force_models.compute_drag_force(
        particle_velocity, fluid_velocity, particle_radius
    )

    # Check direction
    assert np.all(np.sign(drag_force) == np.sign(fluid_velocity))

    # Check magnitude (approximate for Stokes flow)
    expected_magnitude = 6 * np.pi * force_models.fluid_viscosity * \
        particle_radius * np.linalg.norm(fluid_velocity)
    assert np.isclose(np.linalg.norm(drag_force), expected_magnitude, rtol=0.1)


def test_drag_force_turbulent(force_models):
    """Test drag force in turbulent flow regime."""
    particle_velocity = np.array([0.0, 0.0, 0.0])
    # High velocity for turbulent flow
    fluid_velocity = np.array([10.0, 0.0, 0.0])
    particle_radius = 0.1

    drag_force = force_models.compute_drag_force(
        particle_velocity, fluid_velocity, particle_radius
    )

    # Check direction
    assert np.all(np.sign(drag_force) == np.sign(fluid_velocity))

    # Check magnitude (approximate for turbulent flow)
    expected_magnitude = 0.5 * force_models.fluid_density * np.pi * particle_radius**2 * \
        0.44 * np.linalg.norm(fluid_velocity)**2
    assert np.isclose(np.linalg.norm(drag_force), expected_magnitude, rtol=0.1)


def test_pressure_force(force_models):
    """Test pressure force computation."""
    pressure_gradient = np.array([-1000.0, 0.0, 0.0])
    particle_radius = 0.1

    pressure_force = force_models.compute_pressure_force(
        pressure_gradient, particle_radius
    )

    # Check direction
    assert np.all(np.sign(pressure_force) == -np.sign(pressure_gradient))

    # Check magnitude
    expected_magnitude = np.pi * particle_radius**2 * \
        np.linalg.norm(pressure_gradient)
    assert np.isclose(np.linalg.norm(pressure_force), expected_magnitude)


def test_buoyancy_force(force_models):
    """Test buoyancy force computation."""
    particle_radius = 0.1
    gravity = np.array([0.0, 0.0, -9.81])

    buoyancy_force = force_models.compute_buoyancy_force(
        particle_radius, gravity
    )

    # Check direction
    assert np.all(np.sign(buoyancy_force) == -np.sign(gravity))

    # Check magnitude
    expected_magnitude = force_models.fluid_density * 4/3 * np.pi * particle_radius**3 * \
        np.linalg.norm(gravity)
    assert np.isclose(np.linalg.norm(buoyancy_force), expected_magnitude)


def test_lift_force(force_models):
    """Test lift force computation."""
    particle_velocity = np.array([0.0, 0.0, 0.0])
    fluid_velocity = np.array([1.0, 0.0, 0.0])
    fluid_vorticity = np.array([0.0, 0.0, 1.0])
    particle_radius = 0.1

    lift_force = force_models.compute_lift_force(
        particle_velocity, fluid_velocity, fluid_vorticity, particle_radius
    )

    # Check that lift force is perpendicular to relative velocity
    relative_velocity = fluid_velocity - particle_velocity
    assert np.isclose(np.dot(lift_force, relative_velocity), 0.0)


def test_virtual_mass_force(force_models):
    """Test virtual mass force computation."""
    particle_velocity = np.array([0.0, 0.0, 0.0])
    fluid_velocity = np.array([0.0, 0.0, 0.0])
    fluid_acceleration = np.array([1.0, 0.0, 0.0])
    particle_radius = 0.1

    virtual_mass_force = force_models.compute_virtual_mass_force(
        particle_velocity, fluid_velocity, fluid_acceleration, particle_radius
    )

    # Check direction
    assert np.all(np.sign(virtual_mass_force) == np.sign(fluid_acceleration))

    # Check magnitude
    expected_magnitude = 0.5 * force_models.fluid_density * 4/3 * np.pi * particle_radius**3 * \
        np.linalg.norm(fluid_acceleration)
    assert np.isclose(np.linalg.norm(virtual_mass_force), expected_magnitude)


def test_reynolds_number(force_models):
    """Test Reynolds number computation."""
    velocity = 1.0
    particle_radius = 0.1

    reynolds = force_models._compute_reynolds_number(velocity, particle_radius)

    # Check Reynolds number calculation
    expected_reynolds = 2 * particle_radius * velocity * \
        force_models.fluid_density / force_models.fluid_viscosity
    assert np.isclose(reynolds, expected_reynolds)


def test_drag_coefficient(force_models):
    """Test drag coefficient selection."""
    # Test Stokes flow
    assert np.isclose(force_models._get_drag_coefficient(0.5), 48.0)

    # Test intermediate flow
    assert force_models._get_drag_coefficient(500) > 0.44

    # Test turbulent flow
    assert np.isclose(force_models._get_drag_coefficient(2000), 0.44)


def test_compressibility(force_models):
    """Test compressibility effects."""
    # Test incompressible flow
    pressure_gradient = np.array([100.0, 0.0, 0.0])
    assert not force_models._is_compressible_flow(pressure_gradient)

    # Test compressible flow
    pressure_gradient = np.array([1000.0, 0.0, 0.0])
    assert force_models._is_compressible_flow(pressure_gradient)

    # Test compressibility correction
    correction = force_models._compressibility_correction(pressure_gradient)
    assert correction > 1.0
