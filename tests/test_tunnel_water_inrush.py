"""
Tests for the tunnel water inrush case study.
"""

import pytest
import numpy as np
from particle_flow.case_studies.tunnel_water_inrush import TunnelWaterInrush
import os


@pytest.fixture
def config():
    """Test configuration."""
    return {
        'simulation': {
            'time_step': 0.1,
            'total_time': 1.0,
            'domain_size': [100, 10, 10],
            'grid_resolution': [100, 10, 10]
        },
        'dem': {
            'particle_radius': 0.1,
            'particle_density': 2650.0,
            'youngs_modulus': 1e9,
            'poisson_ratio': 0.3,
            'friction_coefficient': 0.5,
            'restitution_coefficient': 0.5,
            'damping_coefficient': 0.1
        },
        'cfd': {
            'fluid_density': 1000.0,
            'fluid_viscosity': 1e-3,
            'gravity': [0.0, 0.0, -9.81]
        },
        'case_study': {
            'water_pressure': 1e6,
            'mud_concentration': 0.3
        }
    }


@pytest.fixture
def case_study(config):
    """Test case study instance."""
    return TunnelWaterInrush(config)


def test_initialization(case_study, config):
    """Test case study initialization."""
    assert case_study.tunnel_length == 80.0
    assert case_study.tunnel_diameter == 8.0
    assert case_study.water_pressure == config['case_study']['water_pressure']
    assert case_study.mud_concentration == config['case_study']['mud_concentration']
    assert len(case_study.time_steps) == 0
    assert len(case_study.erosion_stats) == 0
    assert len(case_study.bond_health_stats) == 0


def test_setup_initial_conditions(case_study):
    """Test setting up initial conditions."""
    case_study.setup_initial_conditions()

    # Check tunnel geometry
    assert 'tunnel_walls' in case_study.__dict__
    assert case_study.tunnel_walls['x_min'] == 0.0
    assert case_study.tunnel_walls['x_max'] > 0.0
    assert case_study.tunnel_walls['y_min'] < 0.0
    assert case_study.tunnel_walls['y_max'] > 0.0


def test_particle_distribution(case_study):
    """Test particle distribution setup."""
    case_study.setup_initial_conditions()

    # Get particle positions
    positions = case_study.coupling.dem_solver.get_particle_positions()

    # Check if particles are within tunnel bounds
    for pos in positions:
        assert case_study.tunnel_walls['x_min'] <= pos[0] <= case_study.tunnel_walls['x_max']
        assert case_study.tunnel_walls['y_min'] <= pos[1] <= case_study.tunnel_walls['y_max']
        assert case_study.tunnel_walls['z_min'] <= pos[2] <= case_study.tunnel_walls['z_max']


def test_water_pressure_setup(case_study):
    """Test water pressure setup."""
    case_study.setup_initial_conditions()

    # Check pressure gradient
    pressure_gradient = case_study.coupling.cfd_solver.config['pressure_gradient']
    assert pressure_gradient[0] < 0  # Negative gradient in x-direction
    assert pressure_gradient[1] == 0.0
    assert pressure_gradient[2] == 0.0


def test_simulation_run(case_study):
    """Test running the simulation."""
    case_study.setup_initial_conditions()
    case_study.run_simulation(0.5)  # Run for 0.5 seconds

    # Check if data was recorded
    assert len(case_study.time_steps) > 0
    assert len(case_study.erosion_stats) > 0
    assert len(case_study.bond_health_stats) > 0
    assert len(case_study.particle_data) > 0
    assert len(case_study.fluid_data) > 0
    assert len(case_study.erosion_data) > 0


def test_results_analysis(case_study):
    """Test results analysis."""
    case_study.setup_initial_conditions()
    case_study.run_simulation(0.5)
    case_study.analyze_results()

    # Check if visualizations were created
    assert case_study.visualizer.fig is not None


def test_save_results(case_study, tmp_path):
    """Test saving results."""
    case_study.setup_initial_conditions()
    case_study.run_simulation(0.5)

    # Save results
    output_dir = str(tmp_path)
    case_study.save_results(output_dir)

    # Check if files were created
    assert os.path.exists(f"{output_dir}/time_steps.npy")
    assert os.path.exists(f"{output_dir}/erosion_stats.npy")
    assert os.path.exists(f"{output_dir}/bond_health_stats.npy")
    assert os.path.exists(f"{output_dir}/particle_data.npy")
    assert os.path.exists(f"{output_dir}/fluid_data.npy")
    assert os.path.exists(f"{output_dir}/erosion_data.npy")
    assert os.path.exists(f"{output_dir}/simulation_animation.mp4")


def test_close(case_study):
    """Test closing resources."""
    case_study.setup_initial_conditions()
    case_study.close()

    # Check if visualizer was closed
    assert case_study.visualizer.fig is None
