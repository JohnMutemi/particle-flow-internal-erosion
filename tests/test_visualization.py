"""
Tests for the visualization module.
"""

import pytest
import numpy as np
from src.visualization.visualizer import SimulationVisualizer
import matplotlib.pyplot as plt

@pytest.fixture
def config():
    """Test configuration."""
    return {
        'simulation': {
            'time_step': 0.1,
            'total_time': 1.0
        }
    }

@pytest.fixture
def visualizer(config):
    """Test visualizer instance."""
    return SimulationVisualizer(config)

@pytest.fixture
def erosion_stats():
    """Sample erosion statistics."""
    return {
        'total_eroded': [0, 1, 2, 3, 4],
        'erosion_rate': [0.0, 1.0, 1.0, 1.0, 1.0],
        'average_force': [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.5, 0.5, 0.0]),
            np.array([2.0, 1.0, 0.5]),
            np.array([2.5, 1.5, 1.0])
        ]
    }

@pytest.fixture
def bond_health_stats():
    """Sample bond health statistics."""
    return {
        'current_health': [1.0, 0.9, 0.8, 0.7, 0.6],
        'degradation_rate': [0.0, 0.1, 0.1, 0.1, 0.1]
    }

@pytest.fixture
def time_steps():
    """Sample time steps."""
    return [0.0, 0.1, 0.2, 0.3, 0.4]

@pytest.fixture
def particle_data():
    """Sample particle data."""
    return {
        'positions': [
            np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]),
            np.array([[0.51, 0.51, 0.51], [0.61, 0.61, 0.61]]),
            np.array([[0.52, 0.52, 0.52], [0.62, 0.62, 0.62]])
        ]
    }

@pytest.fixture
def fluid_data():
    """Sample fluid data."""
    return {
        'velocity_field': [
            np.zeros((10, 10, 10, 3)),
            np.ones((10, 10, 10, 3)) * 0.1,
            np.ones((10, 10, 10, 3)) * 0.2
        ]
    }

@pytest.fixture
def erosion_data():
    """Sample erosion data."""
    return [
        [],
        [{'position': np.array([0.5, 0.5, 0.5]), 'force': np.array([1.0, 0.0, 0.0])}],
        [{'position': np.array([0.5, 0.5, 0.5]), 'force': np.array([1.0, 0.0, 0.0])},
         {'position': np.array([0.6, 0.6, 0.6]), 'force': np.array([1.5, 0.5, 0.0])}]
    ]

def test_plot_erosion_statistics(visualizer, erosion_stats, time_steps):
    """Test plotting erosion statistics."""
    fig = visualizer.plot_erosion_statistics(erosion_stats, time_steps)
    assert fig is not None
    visualizer.close()

def test_plot_bond_health(visualizer, bond_health_stats, time_steps):
    """Test plotting bond health statistics."""
    fig = visualizer.plot_bond_health(bond_health_stats, time_steps)
    assert fig is not None
    visualizer.close()

def test_create_particle_animation(visualizer, particle_data, fluid_data, 
                                 erosion_data, time_steps):
    """Test creating particle animation."""
    animation = visualizer.create_particle_animation(
        particle_data, fluid_data, erosion_data, time_steps[:3]
    )
    assert animation is not None
    visualizer.close()

def test_plot_erosion_pattern(visualizer, erosion_data):
    """Test plotting erosion pattern."""
    # Combine all eroded particles
    all_eroded = []
    for frame in erosion_data:
        all_eroded.extend(frame)
    
    fig = visualizer.plot_erosion_pattern(all_eroded)
    assert fig is not None
    visualizer.close()

def test_plot_force_distribution(visualizer, erosion_data):
    """Test plotting force distribution."""
    # Combine all eroded particles
    all_eroded = []
    for frame in erosion_data:
        all_eroded.extend(frame)
    
    fig = visualizer.plot_force_distribution(all_eroded)
    assert fig is not None
    visualizer.close()

def test_save_animation(visualizer, particle_data, fluid_data, 
                       erosion_data, time_steps, tmp_path):
    """Test saving animation."""
    # Create animation
    visualizer.create_particle_animation(
        particle_data, fluid_data, erosion_data, time_steps[:3]
    )
    
    # Save animation
    filename = tmp_path / "test_animation.mp4"
    visualizer.save_animation(str(filename))
    
    # Check if file exists
    assert filename.exists()
    visualizer.close()

def test_close(visualizer):
    """Test closing all figures."""
    # Create some figures
    plt.figure()
    plt.figure()
    
    # Close all figures
    visualizer.close()
    
    # Check if all figures are closed
    assert len(plt.get_fignums()) == 0 