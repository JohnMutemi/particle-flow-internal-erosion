"""
Real-time visualization module for CFD-DEM simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class RealTimeVisualizer:
    def __init__(self, config: Dict):
        """Initialize the real-time visualizer."""
        # Validate required configuration parameters
        required_params = {
            'simulation': ['domain_size'],
            'output': ['directory', 'visualization_interval']
        }
        
        for section, params in required_params.items():
            if section not in config:
                raise ValueError(f"Missing configuration section: {section}")
            for param in params:
                if param not in config[section]:
                    raise ValueError(f"Missing parameter {param} in {section} section")
        
        self.config = config
        self.domain_size = np.array(config['simulation']['domain_size'])
        self.output_dir = config['output']['directory']
        self.visualization_interval = config['output']['visualization_interval']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize visualization state
        self.fig = None
        self.ax = None
        self.animation = None
        self.particle_scatter = None
        self.fluid_quiver = None
        self.time_text = None
        
        # Data storage
        self.particle_data = []
        self.fluid_data = []
        self.time_steps = []
        
        logger.info("Real-time visualizer initialized")

    def start(self):
        """Start the real-time visualization."""
        try:
            # Create figure and axes
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Set up initial plot
            self._setup_plot()
            
            # Create animation
            self.animation = FuncAnimation(
                self.fig,
                self._update,
                interval=100,  # Update every 100ms
                blit=True
            )
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error starting visualization: {str(e)}")
            raise

    def update_data(self, particle_positions: List[np.ndarray],
                   fluid_velocity: Optional[np.ndarray] = None,
                   time_step: Optional[float] = None):
        """Update visualization data."""
        try:
            # Validate input data
            if not isinstance(particle_positions, list):
                raise ValueError("particle_positions must be a list of numpy arrays")
            for pos in particle_positions:
                if not isinstance(pos, np.ndarray) or pos.shape != (3,):
                    raise ValueError("Each particle position must be a 3D numpy array")
            
            # Store data
            self.particle_data.append(particle_positions)
            if fluid_velocity is not None:
                self.fluid_data.append(fluid_velocity)
            if time_step is not None:
                self.time_steps.append(time_step)
            
            # Limit stored data to prevent memory issues
            max_stored_frames = 1000
            if len(self.particle_data) > max_stored_frames:
                self.particle_data = self.particle_data[-max_stored_frames:]
                self.fluid_data = self.fluid_data[-max_stored_frames:]
                self.time_steps = self.time_steps[-max_stored_frames:]
                
        except Exception as e:
            logger.error(f"Error updating visualization data: {str(e)}")
            raise

    def save_animation(self, filename: Optional[str] = None):
        """Save the animation to a file."""
        try:
            if self.animation is None:
                raise RuntimeError("No animation to save")
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"simulation_{timestamp}.mp4"
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Save animation
            self.animation.save(
                filepath,
                writer='ffmpeg',
                fps=30,
                dpi=100,
                bitrate=1800
            )
            
            logger.info(f"Animation saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving animation: {str(e)}")
            raise

    def _setup_plot(self):
        """Set up the initial plot."""
        # Set plot limits
        self.ax.set_xlim(0, self.domain_size[0])
        self.ax.set_ylim(0, self.domain_size[1])
        self.ax.set_zlim(0, self.domain_size[2])
        
        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Set title
        self.ax.set_title('CFD-DEM Simulation')
        
        # Initialize scatter plot for particles
        self.particle_scatter = self.ax.scatter([], [], [], c='b', alpha=0.6)
        
        # Initialize quiver plot for fluid velocity
        self.fluid_quiver = self.ax.quiver([], [], [], [], [], [], color='r', alpha=0.4)
        
        # Add time text
        self.time_text = self.ax.text2D(0.02, 0.95, '', transform=self.ax.transAxes)

    def _update(self, frame):
        """Update the plot for each frame."""
        try:
            if not self.particle_data:
                return self.particle_scatter, self.fluid_quiver, self.time_text
            
            # Get current data
            current_particles = self.particle_data[frame]
            current_fluid = self.fluid_data[frame] if self.fluid_data else None
            current_time = self.time_steps[frame] if self.time_steps else frame
            
            # Update particle positions
            positions = np.array(current_particles)
            self.particle_scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
            
            # Update fluid velocity if available
            if current_fluid is not None:
                # Create grid points for quiver plot
                x = np.linspace(0, self.domain_size[0], 5)
                y = np.linspace(0, self.domain_size[1], 5)
                z = np.linspace(0, self.domain_size[2], 5)
                X, Y, Z = np.meshgrid(x, y, z)
                
                # Update quiver plot
                self.fluid_quiver.remove()
                self.fluid_quiver = self.ax.quiver(
                    X, Y, Z,
                    current_fluid[0], current_fluid[1], current_fluid[2],
                    color='r', alpha=0.4
                )
            
            # Update time text
            self.time_text.set_text(f'Time: {current_time:.3f}')
            
            return self.particle_scatter, self.fluid_quiver, self.time_text
            
        except Exception as e:
            logger.error(f"Error updating plot: {str(e)}")
            return self.particle_scatter, self.fluid_quiver, self.time_text 