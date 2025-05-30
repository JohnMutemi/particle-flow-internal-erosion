"""
Visualization module for monitoring erosion and degradation in CFD-DEM simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple
import logging
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

class SimulationVisualizer:
    def __init__(self, config: Dict):
        """Initialize the visualization module."""
        self.config = config
        self.fig = None
        self.animation = None
        self.figsize = (12, 8)
        self.dpi = 300
        self.cmap = 'viridis'
        
        # Set style
        plt.style.use('seaborn-v0_8')  # Use a valid seaborn style
        sns.set_theme()  # Set seaborn theme
        
        # Set up output directory
        self.output_dir = Path(config['output']['directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Visualization module initialized")

    def plot_erosion_statistics(self, erosion_stats: Dict, time_steps: List[float]):
        """Plot erosion statistics over time."""
        plt.figure(figsize=(12, 8))
        
        # Plot total eroded particles
        plt.subplot(2, 2, 1)
        plt.plot(time_steps, erosion_stats['total_eroded'], 'b-', label='Total Eroded')
        plt.xlabel('Time (s)')
        plt.ylabel('Number of Particles')
        plt.title('Total Eroded Particles')
        plt.grid(True)
        
        # Plot erosion rate
        plt.subplot(2, 2, 2)
        plt.plot(time_steps, erosion_stats['erosion_rate'], 'r-', label='Erosion Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Erosion Rate (particles/s)')
        plt.title('Erosion Rate')
        plt.grid(True)
        
        # Plot average force magnitude
        plt.subplot(2, 2, 3)
        force_magnitudes = [np.linalg.norm(f) for f in erosion_stats['average_force']]
        plt.plot(time_steps, force_magnitudes, 'g-', label='Average Force')
        plt.xlabel('Time (s)')
        plt.ylabel('Force Magnitude (N)')
        plt.title('Average Erosion Force')
        plt.grid(True)
        
        # Plot force components
        plt.subplot(2, 2, 4)
        forces = np.array(erosion_stats['average_force'])
        plt.plot(time_steps, forces[:, 0], 'r-', label='Fx')
        plt.plot(time_steps, forces[:, 1], 'g-', label='Fy')
        plt.plot(time_steps, forces[:, 2], 'b-', label='Fz')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.title('Force Components')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        return plt.gcf()

    def plot_bond_health(self, bond_health_stats: Dict, time_steps: List[float]):
        """Plot bond health statistics over time."""
        plt.figure(figsize=(10, 6))
        
        # Plot bond health
        plt.subplot(1, 2, 1)
        plt.plot(time_steps, bond_health_stats['current_health'], 'b-', label='Bond Health')
        plt.xlabel('Time (s)')
        plt.ylabel('Bond Health')
        plt.title('Bond Health Evolution')
        plt.grid(True)
        
        # Plot degradation rate
        plt.subplot(1, 2, 2)
        plt.plot(time_steps, bond_health_stats['degradation_rate'], 'r-', label='Degradation Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Degradation Rate (1/s)')
        plt.title('Bond Degradation Rate')
        plt.grid(True)
        
        plt.tight_layout()
        return plt.gcf()

    def create_particle_animation(self, particle_data: Dict, 
                                fluid_data: Dict,
                                erosion_data: List[Dict],
                                time_steps: List[float]):
        """Create an animated visualization of particles and fluid flow."""
        self.fig = plt.figure(figsize=(12, 8))
        ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize scatter plot for particles
        scatter = ax.scatter([], [], [], c='b', alpha=0.6)
        
        # Initialize quiver plot for fluid velocity
        quiver = ax.quiver([], [], [], [], [], [], color='r', alpha=0.3)
        
        # Initialize scatter plot for eroded particles
        eroded_scatter = ax.scatter([], [], [], c='r', alpha=0.8)
        
        def update(frame):
            nonlocal quiver
            # Update particle positions
            positions = particle_data['positions'][frame]
            scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
            
            # Update fluid velocity
            fluid_vel = fluid_data['velocity_field'][frame]
            x, y, z = np.meshgrid(np.linspace(0, 1, 10),
                                np.linspace(0, 1, 10),
                                np.linspace(0, 1, 10))
            
            # Update quiver plot
            quiver.remove()
            quiver = ax.quiver(x, y, z,
                             fluid_vel[::10, ::10, ::10, 0],
                             fluid_vel[::10, ::10, ::10, 1],
                             fluid_vel[::10, ::10, ::10, 2],
                             color='r', alpha=0.3)
            
            # Update eroded particles
            eroded_positions = [p['position'] for p in erosion_data[frame]]
            if eroded_positions:
                eroded_positions = np.array(eroded_positions)
                eroded_scatter._offsets3d = (eroded_positions[:, 0],
                                          eroded_positions[:, 1],
                                          eroded_positions[:, 2])
            
            ax.set_title(f'Time: {time_steps[frame]:.2f}s')
            return scatter, quiver, eroded_scatter
        
        # Create animation
        self.animation = FuncAnimation(self.fig, update,
                                     frames=len(time_steps),
                                     interval=100,
                                     blit=True)
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        return self.animation

    def plot_erosion_pattern(self, eroded_particles: List[Dict]):
        """Plot the spatial pattern of erosion."""
        plt.figure(figsize=(10, 8))
        
        # Extract positions
        positions = np.array([p['position'] for p in eroded_particles])
        forces = np.array([p['force'] for p in eroded_particles])
        
        # Create 3D scatter plot
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=np.linalg.norm(forces, axis=1),
                           cmap='viridis',
                           alpha=0.6)
        
        # Add colorbar
        plt.colorbar(scatter, label='Erosion Force Magnitude (N)')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Spatial Pattern of Erosion')
        
        return plt.gcf()

    def plot_force_distribution(self, eroded_particles: List[Dict]):
        """Plot the distribution of erosion forces."""
        plt.figure(figsize=(12, 6))
        
        # Extract forces
        forces = np.array([p['force'] for p in eroded_particles])
        force_magnitudes = np.linalg.norm(forces, axis=1)
        
        # Plot force magnitude distribution
        plt.subplot(1, 2, 1)
        sns.histplot(force_magnitudes, bins=30, kde=True)
        plt.xlabel('Force Magnitude (N)')
        plt.ylabel('Count')
        plt.title('Distribution of Erosion Forces')
        
        # Plot force components
        plt.subplot(1, 2, 2)
        sns.boxplot(data=forces)
        plt.xlabel('Force Component')
        plt.ylabel('Force (N)')
        plt.title('Force Component Distribution')
        
        plt.tight_layout()
        return plt.gcf()

    def save_animation(self, filename: str):
        """Save the animation to a file."""
        if not self.animation:
            raise ValueError("No animation has been created")
            
        try:
            if filename.endswith('.mp4'):
                # Try to save as MP4 first
                try:
                    self.animation.save(filename, writer='ffmpeg')
                    logger.info(f"Successfully saved animation as MP4: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to save as MP4: {str(e)}")
                    # Fall back to GIF
                    gif_filename = filename.replace('.mp4', '.gif')
                    self.animation.save(gif_filename, writer='pillow')
                    logger.info(f"Saved animation as GIF instead: {gif_filename}")
            elif filename.endswith('.gif'):
                self.animation.save(filename, writer='pillow')
                logger.info(f"Successfully saved animation as GIF: {filename}")
            else:
                # Default to GIF for unknown formats
                gif_filename = filename + '.gif'
                self.animation.save(gif_filename, writer='pillow')
                logger.info(f"Saved animation as GIF: {gif_filename}")
        except Exception as e:
            logger.error(f"Failed to save animation: {str(e)}")
            raise

    def close(self):
        """Close all figures."""
        plt.close('all')
        logger.info("All figures closed")

    def plot_particle_distribution(self, positions: np.ndarray,
                                 bond_health: np.ndarray,
                                 output_file: Path):
        """Plot particle distribution with bond health coloring."""
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Create 3D scatter plot
        scatter = plt.scatter(
            positions[:, 0],
            positions[:, 1],
            c=bond_health,
            cmap=self.cmap,
            s=50,
            alpha=0.6
        )
        
        # Add colorbar
        plt.colorbar(scatter, label='Bond Health')
        
        # Set labels and title
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Particle Distribution with Bond Health')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
    
    def plot_fluid_field(self, velocity_field: np.ndarray,
                        pressure_field: np.ndarray,
                        output_file: Path):
        """Plot fluid velocity and pressure fields."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Plot velocity magnitude
        velocity_magnitude = np.linalg.norm(velocity_field, axis=-1)
        im1 = ax1.imshow(
            velocity_magnitude[:, :, velocity_field.shape[2]//2],
            cmap=self.cmap,
            origin='lower'
        )
        ax1.set_title('Fluid Velocity Magnitude')
        plt.colorbar(im1, ax=ax1, label='Velocity (m/s)')
        
        # Plot pressure
        im2 = ax2.imshow(
            pressure_field[:, :, pressure_field.shape[2]//2],
            cmap=self.cmap,
            origin='lower'
        )
        ax2.set_title('Fluid Pressure')
        plt.colorbar(im2, ax=ax2, label='Pressure (Pa)')
        
        # Set labels
        for ax in [ax1, ax2]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
    
    def plot_erosion_progress(self, time: List[float],
                            eroded_particles: List[int],
                            bond_health: List[float],
                            output_file: Path):
        """Plot erosion progress over time."""
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Plot number of eroded particles
        ax1.plot(time, eroded_particles, 'r-', label='Eroded Particles')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Number of Eroded Particles')
        ax1.set_title('Erosion Progress')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot average bond health
        ax2.plot(time, bond_health, 'b-', label='Average Bond Health')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Bond Health')
        ax2.set_title('Bond Degradation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
    
    def plot_force_distribution(self, forces: np.ndarray,
                              output_file: Path):
        """Plot distribution of forces on particles."""
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Compute force magnitudes
        force_magnitudes = np.linalg.norm(forces, axis=-1)
        
        # Create histogram
        plt.hist(force_magnitudes, bins=50, alpha=0.7)
        plt.xlabel('Force Magnitude (N)')
        plt.ylabel('Number of Particles')
        plt.title('Distribution of Forces on Particles')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
    
    def plot_velocity_distribution(self, velocities: np.ndarray,
                                 output_file: Path):
        """Plot distribution of particle velocities."""
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Compute velocity magnitudes
        velocity_magnitudes = np.linalg.norm(velocities, axis=-1)
        
        # Create histogram
        plt.hist(velocity_magnitudes, bins=50, alpha=0.7)
        plt.xlabel('Velocity Magnitude (m/s)')
        plt.ylabel('Number of Particles')
        plt.title('Distribution of Particle Velocities')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(output_file, bbox_inches='tight')
        plt.close() 