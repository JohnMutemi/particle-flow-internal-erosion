"""
Tunnel water inrush case study implementation.
This module simulates the water inrush scenario in an 80m tunnel with muddy water.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
from src.cfd.coupling import CFDDEMCoupling
from src.visualization.real_time_visualizer import RealTimeVisualizer
from src.cfd.force_models import ForceModels

logger = logging.getLogger(__name__)

class TunnelWaterInrush:
    def __init__(self, config: Dict):
        """Initialize the tunnel water inrush case study."""
        self.config = config
        
        # Initialize coupling framework
        self.coupling = CFDDEMCoupling(config)
        
        # Initialize force models
        self.force_models = ForceModels(config)
        
        # Initialize real-time visualizer
        self.visualizer = RealTimeVisualizer(config)
        
        # Case study specific parameters
        self.tunnel_length = 80.0  # meters
        self.tunnel_diameter = 8.0  # meters
        self.water_pressure = config['case_study'].get('water_pressure', 1e6)  # Pa
        self.mud_concentration = config['case_study'].get('mud_concentration', 0.3)  # 30% by volume
        
        # Initialize tunnel geometry
        self._initialize_tunnel_geometry()
        
        # State tracking
        self.time_steps = []
        self.erosion_stats = []
        self.bond_health_stats = []
        self.particle_data = []
        self.fluid_data = []
        self.erosion_data = []
        
        logger.info("Tunnel water inrush case study initialized")

    def _initialize_tunnel_geometry(self):
        """Initialize tunnel geometry and boundary conditions."""
        # Convert tunnel dimensions to simulation units
        length_scale = self.config['simulation']['domain_size'][0] / self.tunnel_length
        diameter_scale = self.config['simulation']['domain_size'][1] / self.tunnel_diameter
        
        # Set up tunnel walls with proper boundary checking
        self.tunnel_walls = {
            'x_min': max(0.0, 0.0),
            'x_max': min(self.tunnel_length * length_scale, self.config['simulation']['domain_size'][0]),
            'y_min': max(-self.tunnel_diameter/2 * diameter_scale, -self.config['simulation']['domain_size'][1]/2),
            'y_max': min(self.tunnel_diameter/2 * diameter_scale, self.config['simulation']['domain_size'][1]/2),
            'z_min': max(-self.tunnel_diameter/2 * diameter_scale, -self.config['simulation']['domain_size'][2]/2),
            'z_max': min(self.tunnel_diameter/2 * diameter_scale, self.config['simulation']['domain_size'][2]/2)
        }
        
        # Set up tunnel geometry for CFD solver
        tunnel_params = {
            'length': min(self.tunnel_length * length_scale, self.config['simulation']['domain_size'][0]),
            'diameter': min(self.tunnel_diameter * diameter_scale, min(self.config['simulation']['domain_size'][1:])),
            'center': np.array([
                min(self.tunnel_length * length_scale / 2, self.config['simulation']['domain_size'][0] / 2),
                0.0,
                0.0
            ]),
            'roughness': 0.001  # 1mm roughness
        }
        
        # Set boundary conditions
        boundary_conditions = {
            'inlet': {
                'type': 'pressure',
                'value': self.water_pressure,
                'position': 0.0
            },
            'outlet': {
                'type': 'pressure',
                'value': 0.0,
                'position': min(self.tunnel_length * length_scale, self.config['simulation']['domain_size'][0])
            },
            'walls': {
                'type': 'no-slip',
                'roughness': 0.001
            }
        }
        
        # Update CFD solver with tunnel geometry and boundary conditions
        self.coupling.cfd_solver.set_tunnel_geometry(tunnel_params)
        self.coupling.cfd_solver.set_boundary_conditions(boundary_conditions)
        
        logger.info("Tunnel geometry and boundary conditions initialized")

    def setup_initial_conditions(self):
        """Set up initial conditions for the tunnel water inrush scenario."""
        # Initialize simulation
        self.coupling.initialize_simulation()
        
        # Set up initial water pressure
        self._setup_water_pressure()
        
        # Set up initial particle distribution
        self._setup_particle_distribution()
        
        # Start real-time visualization
        self.visualizer.start()
        
        logger.info("Initial conditions set up")

    def _setup_water_pressure(self):
        """Set up initial water pressure distribution."""
        # Get grid points
        x = np.linspace(0, self.tunnel_length, self.config['simulation']['grid_resolution'][0])
        y = np.linspace(-self.tunnel_diameter/2, self.tunnel_diameter/2, 
                       self.config['simulation']['grid_resolution'][1])
        z = np.linspace(-self.tunnel_diameter/2, self.tunnel_diameter/2,
                       self.config['simulation']['grid_resolution'][2])
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Compute distance from tunnel center
        R = np.sqrt(Y**2 + Z**2)
        
        # Set pressure based on radial distance
        pressure = np.zeros_like(X)
        mask = R <= self.tunnel_diameter/2
        pressure[mask] = self.water_pressure * (1 - R[mask]/(self.tunnel_diameter/2))
        
        # Update pressure field in CFD solver
        self.coupling.cfd_solver.pressure_field = pressure
        
        logger.info("Initial water pressure distribution set up")

    def _setup_particle_distribution(self):
        """Set up initial particle distribution in the tunnel."""
        num_particles = self.config['simulation']['initial_particles']
        particles = []
        
        # Get scaling factors
        length_scale = self.config['simulation']['domain_size'][0] / self.tunnel_length
        diameter_scale = self.config['simulation']['domain_size'][1] / self.tunnel_diameter
        
        for _ in range(num_particles):
            # Random position within tunnel in real-world coordinates
            r = np.random.uniform(0, min(self.tunnel_diameter/2, self.config['simulation']['domain_size'][1]/2))
            theta = np.random.uniform(0, 2*np.pi)
            x = np.random.uniform(0, min(self.tunnel_length, self.config['simulation']['domain_size'][0]))
            
            # Convert to Cartesian coordinates
            y = r * np.cos(theta)
            z = r * np.sin(theta)
            
            # Scale to simulation domain
            position = np.array([
                x * length_scale,
                y * diameter_scale,
                z * diameter_scale
            ])
            
            # Add particle with boundary checking
            particles.append({
                'position': position,
                'velocity': np.zeros(3),
                'radius': min(self.config['dem']['particle_radius'], 
                            min(self.tunnel_diameter/4, min(self.config['simulation']['domain_size'][1:])/4)),
                'density': self.config['dem']['particle_density']
            })
        
        # Update particle positions in DEM solver
        self.coupling.dem_solver.particles = particles
        
        logger.info(f"Initial particle distribution set up with {num_particles} particles")

    def run_simulation(self, num_steps: int):
        """Run the tunnel water inrush simulation."""
        logger.info(f"Starting simulation for {num_steps} steps")
        
        for step in range(num_steps):
            try:
                # Update fluid state
                self.coupling.cfd_solver.update_fluid_state()
                
                # Compute fluid forces on particles
                fluid_forces = self.coupling.cfd_solver.compute_fluid_forces(
                    [p['position'] for p in self.coupling.dem_solver.particles]
                )
                
                # Update particle states
                self.coupling.dem_solver.update_particle_states(fluid_forces)
                
                # Update erosion
                self._update_erosion()
                
                # Update visualization
                self._update_visualization(step)
                
                # Store statistics
                self._store_statistics(step)
                
                if step % 100 == 0:
                    logger.info(f"Completed step {step}/{num_steps}")
            except Exception as e:
                logger.error(f"Error during simulation step {step}: {e}")
                continue

    def _update_erosion(self):
        """Update erosion state based on fluid forces and particle properties."""
        # Get current fluid velocity and pressure
        fluid_data = self.coupling.cfd_solver.get_fluid_data()
        velocity_field = fluid_data['velocity_field']
        pressure_field = fluid_data['pressure_field']
        
        # Update erosion for each particle
        for particle in self.coupling.dem_solver.particles:
            # Check if particle position is within bounds
            if not self._is_position_in_bounds(particle['position']):
                continue
                
            # Interpolate fluid properties at particle position
            try:
                velocity = self.coupling.cfd_solver._interpolate_field(
                    particle['position'], velocity_field
                )
                pressure = self.coupling.cfd_solver._interpolate_field(
                    particle['position'], pressure_field
                )
                
                # Compute shear stress
                shear_stress = self._compute_shear_stress(velocity, particle['radius'])
                
                # Update particle erosion state
                self._update_particle_erosion(particle, shear_stress)
            except IndexError as e:
                logger.warning(f"Index error during interpolation: {e}")
                continue

    def _is_position_in_bounds(self, position: np.ndarray) -> bool:
        """Check if a position is within the simulation domain bounds."""
        return (
            self.tunnel_walls['x_min'] <= position[0] <= self.tunnel_walls['x_max'] and
            self.tunnel_walls['y_min'] <= position[1] <= self.tunnel_walls['y_max'] and
            self.tunnel_walls['z_min'] <= position[2] <= self.tunnel_walls['z_max']
        )

    def _compute_shear_stress(self, velocity: np.ndarray, particle_radius: float) -> float:
        """Compute shear stress on particle surface."""
        velocity_magnitude = np.linalg.norm(velocity)
        reynolds = 2 * particle_radius * velocity_magnitude * \
                  self.config['cfd']['fluid_density'] / \
                  self.config['cfd']['fluid_viscosity']
        
        if reynolds < 1:
            # Stokes flow
            return 3 * self.config['cfd']['fluid_viscosity'] * velocity_magnitude / particle_radius
        else:
            # Turbulent flow
            return 0.5 * self.config['cfd']['fluid_density'] * velocity_magnitude**2

    def _update_particle_erosion(self, particle: Dict, shear_stress: float):
        """Update particle erosion state based on shear stress."""
        critical_shear_stress = self.config['erosion']['critical_shear_stress']
        erosion_rate = self.config['erosion']['erosion_rate_coefficient']
        
        if shear_stress > critical_shear_stress:
            # Compute erosion rate
            erosion = erosion_rate * (shear_stress - critical_shear_stress)
            
            # Update particle properties with boundary checking
            new_radius = particle['radius'] * (1 - erosion)
            if new_radius >= 0.1 * self.config['dem']['particle_radius']:
                particle['radius'] = new_radius
            else:
                particle['eroded'] = True

    def _update_visualization(self, step: int):
        """Update visualization with current simulation state."""
        if step % self.config['output']['visualization_interval'] == 0:
            try:
                # Get current particle positions
                particle_positions = [p['position'] for p in self.coupling.dem_solver.particles]
                
                # Get current fluid velocity
                fluid_data = self.coupling.cfd_solver.get_fluid_data()
                fluid_velocity = fluid_data['velocity_field']
                
                # Update visualization
                self.visualizer.update_data(
                    particle_positions=particle_positions,
                    fluid_velocity=fluid_velocity,
                    time_step=step * self.config['simulation']['time_step']
                )
            except Exception as e:
                logger.warning(f"Error during visualization update: {e}")

    def _store_statistics(self, step: int):
        """Store simulation statistics."""
        try:
            # Store time step
            self.time_steps.append(step * self.config['simulation']['time_step'])
            
            # Compute erosion statistics
            eroded_particles = sum(1 for p in self.coupling.dem_solver.particles if p.get('eroded', False))
            self.erosion_stats.append(eroded_particles)
            
            # Compute bond health statistics
            if hasattr(self.coupling, 'bond_model'):
                bond_health = self.coupling.bond_model.get_bond_health()
                self.bond_health_stats.append(bond_health)
            
            # Store particle data
            self.particle_data.append([p['position'] for p in self.coupling.dem_solver.particles])
            
            # Store fluid data
            fluid_data = self.coupling.cfd_solver.get_fluid_data()
            self.fluid_data.append(fluid_data)
        except Exception as e:
            logger.warning(f"Error during statistics storage: {e}")

    def save_results(self, filename: str = None):
        """Save simulation results."""
        if filename is None:
            filename = f"tunnel_water_inrush_{len(self.time_steps)}.npz"
        
        try:
            np.savez(
                filename,
                time_steps=np.array(self.time_steps),
                erosion_stats=np.array(self.erosion_stats),
                bond_health_stats=np.array(self.bond_health_stats),
                particle_data=np.array(self.particle_data),
                fluid_data=np.array(self.fluid_data)
            )
            
            logger.info(f"Results saved to {filename}")
            
            # Save visualization
            self.visualizer.save_animation()
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def close(self):
        """Clean up resources."""
        try:
            self.visualizer.close()
            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}") 