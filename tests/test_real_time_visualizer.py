"""
Tests for the real-time visualization module.
"""

import pytest
import numpy as np
from src.visualization.real_time_visualizer import RealTimeVisualizer
import time
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
    return RealTimeVisualizer(config)

@pytest.fixture
def sample_data(sample_particle_data, sample_fluid_data, sample_erosion_data, sample_bond_health_data):
    """Sample data for visualization."""
    return {
        'time': 0.1,
        'total_eroded': sample_erosion_data['total_eroded'],
        'erosion_rate': sample_erosion_data['erosion_rate'],
        'average_force': sample_erosion_data['average_force'],
        'bond_health': sample_bond_health_data['current_health'],
        'particle_positions': sample_particle_data['positions'],
        'fluid_velocity': sample_fluid_data['velocity_field'],
        'eroded_positions': sample_erosion_data['eroded_positions']
    }

def test_initialization(visualizer, config):
    """Test visualizer initialization."""
    assert visualizer.fig_3d is not None
    assert visualizer.fig_stats is not None
    assert visualizer.ax_3d is not None
    assert visualizer.ax_erosion is not None
    assert visualizer.ax_rate is not None
    assert visualizer.ax_force is not None
    assert visualizer.ax_bond is not None

def test_start_stop(visualizer):
    """Test starting and stopping visualization."""
    visualizer.start()
    assert visualizer.update_thread is not None
    assert visualizer.update_thread.is_alive()
    
    visualizer.stop()
    assert not visualizer.update_thread.is_alive()

def test_update(visualizer, sample_data):
    """Test updating visualization with data."""
    visualizer.start()
    
    # Update with sample data
    visualizer.update(sample_data)
    
    # Wait for update to be processed
    time.sleep(0.1)
    
    # Check if plots were updated
    assert visualizer.scatter_3d is not None
    assert visualizer.quiver_3d is not None
    assert visualizer.eroded_scatter_3d is not None
    
    visualizer.stop()

def test_statistics_plots(visualizer, sample_data):
    """Test statistics plots update."""
    visualizer.start()
    
    # Update multiple times to test time series
    for t in range(5):
        data = sample_data.copy()
        data['time'] = t * 0.1
        data['total_eroded'] = t
        data['erosion_rate'] = t * 0.5
        data['average_force'] = np.array([t, 0.0, 0.0])
        data['bond_health'] = 1.0 - t * 0.1
        visualizer.update(data)
    
    # Wait for updates to be processed
    time.sleep(0.5)
    
    # Check if plots were updated
    assert len(visualizer.time_data) > 0
    assert len(visualizer.erosion_data) > 0
    assert len(visualizer.rate_data) > 0
    assert len(visualizer.force_data) > 0
    assert len(visualizer.bond_data) > 0
    
    visualizer.stop()

def test_3d_plot_update(visualizer, sample_data):
    """Test 3D plot update."""
    visualizer.start()
    
    # Update with sample data
    visualizer.update(sample_data)
    
    # Wait for update to be processed
    time.sleep(0.1)
    
    # Check if 3D plot was updated
    assert visualizer.scatter_3d is not None
    assert visualizer.quiver_3d is not None
    assert visualizer.eroded_scatter_3d is not None
    
    # Check plot limits
    assert visualizer.ax_3d.get_xlim() is not None
    assert visualizer.ax_3d.get_ylim() is not None
    assert visualizer.ax_3d.get_zlim() is not None
    
    visualizer.stop()

def test_close(visualizer):
    """Test closing visualizer."""
    visualizer.start()
    visualizer.close()
    
    # Check if figures were closed
    assert not plt.fignum_exists(visualizer.fig_3d.number)
    assert not plt.fignum_exists(visualizer.fig_stats.number)

def test_error_handling(visualizer):
    """Test error handling in update loop."""
    visualizer.start()
    
    # Update with invalid data
    visualizer.update({'invalid': 'data'})
    
    # Wait for update to be processed
    time.sleep(0.1)
    
    # Check if visualizer is still running
    assert visualizer.update_thread.is_alive()
    
    visualizer.stop()

def test_multiple_updates(visualizer, sample_data):
    """Test handling multiple rapid updates."""
    visualizer.start()
    
    # Send multiple updates
    for _ in range(10):
        visualizer.update(sample_data)
    
    # Wait for updates to be processed
    time.sleep(0.5)
    
    # Check if visualizer is still running
    assert visualizer.update_thread.is_alive()
    
    visualizer.stop() 