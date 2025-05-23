"""
CFD-DEM coupling framework for fluid-particle interaction simulation.

This module implements the coupling between CFD (fluid) and DEM (particles), using geotechnical parameters such as density, specific gravity, water content, Cu, Cc, clay content, permeability, etc., to compute fluid-particle interaction forces and update particle bonds.

References:
- Gu et al. (2019), Acta Geotechnica
- User's personal geotechnical parameters (see config)
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
from src.cfd.solver import CFDSolver
from src.dem.solver import DEMSolver
from src.dem.coarse_grained.coarse_grained_model import CoarseGrainedModel
from src.dem.constitutive.bond_model import SeepageErosionBondModel

logger = logging.getLogger(__name__)

class CFDDEMCoupling:
    def __init__(self, config: Dict):
        """Initialize the CFD-DEM coupling framework."""
        self.config = config
        
        # Initialize solvers
        self.cfd_solver = CFDSolver(config)
        self.dem_solver = DEMSolver(config)
        
        # Initialize bond model for erosion
        self.bond_model = SeepageErosionBondModel(config)
        
        # Initialize coarse-grained model if needed
        if config.get('use_coarse_grained', False):
            self.coarse_model = CoarseGrainedModel(config)
        else:
            self.coarse_model = None
        
        # Load geotechnical parameters
        self.geotech_params = self._load_geotech_params()
        
        # Coupling parameters
        self.coupling_interval = config['coupling'].get('interval', 1)
        self.interpolation_method = config['coupling'].get('interpolation', 'linear')
        
        # Erosion parameters (calibrated with geotechnical parameters)
        self.critical_shear_stress = self._calibrate_critical_shear_stress()
        self.erosion_rate_coefficient = self._calibrate_erosion_rate()
        
        # State tracking
        self.eroded_particles = []
        self.bond_health_history = []
        self.fluid_particle_interactions = []
        
        logger.info("CFD-DEM coupling framework initialized with geotechnical parameters")

    def _load_geotech_params(self) -> Dict:
        """Load geotechnical parameters from config."""
        return {
            'clay_content': self.config['geotechnical']['clay_content'],
            'water_content': self.config['geotechnical']['water_content'],
            'Cu': self.config['geotechnical']['Cu'],
            'Cc': self.config['geotechnical']['Cc'],
            'cohesion': self.config['geotechnical']['cohesion'],
            'permeability': self.config['geotechnical']['permeability'],
            'density': self.config['geotechnical']['density'],
            'specific_gravity': self.config['geotechnical']['specific_gravity'],
            'porosity': self.config['geotechnical']['porosity'],
            'void_ratio': self.config['geotechnical']['void_ratio']
        }

    def _calibrate_critical_shear_stress(self) -> float:
        """Calibrate critical shear stress based on geotechnical parameters."""
        base_stress = self.config['erosion'].get('critical_shear_stress', 1.0)
        
        # Adjust based on clay content
        clay_factor = 1.0 + 0.2 * (self.geotech_params['clay_content'] / 20.0)
        
        # Adjust based on water content
        water_factor = 1.0 - 0.3 * (self.geotech_params['water_content'] / 20.0)
        
        # Adjust based on Cu
        cu_factor = 1.0 + 0.1 * (self.geotech_params['Cu'] / 10.0)
        
        return base_stress * clay_factor * water_factor * cu_factor

    def _calibrate_erosion_rate(self) -> float:
        """Calibrate erosion rate coefficient based on geotechnical parameters."""
        base_rate = self.config['erosion'].get('erosion_rate_coefficient', 1e-6)
        
        # Adjust based on permeability
        perm_factor = self.geotech_params['permeability'] / 1e-6  # Normalize to 1e-6 m/s
        
        # Adjust based on porosity
        porosity_factor = self.geotech_params['porosity'] / 0.3  # Normalize to 0.3
        
        # Adjust based on void ratio
        void_ratio_factor = self.geotech_params['void_ratio'] / 0.6  # Normalize to 0.6
        
        return base_rate * perm_factor * porosity_factor * void_ratio_factor

    def initialize_simulation(self):
        """Initialize the coupled simulation."""
        # Initialize fluid fields
        self.cfd_solver.initialize_fields()
        
        # Initialize particles
        if self.coarse_model:
            num_particles = self.config['simulation']['initial_particles']
            self.coarse_model.initialize_coarse_particles(num_particles)
        else:
            num_particles = self.config['simulation']['initial_particles']
            self.dem_solver.initialize_particles(num_particles)
        
        # Initialize erosion tracking
        self.eroded_particles = []
        self.bond_health_history = []
        self.fluid_particle_interactions = []
        
        logger.info("Coupled simulation initialized")

    def compute_fluid_forces(self, particle_positions: List[np.ndarray],
                           particle_radii: List[float]) -> List[np.ndarray]:
        """Compute fluid forces on particles."""
        # Get fluid velocity at particle positions
        fluid_velocities = self._interpolate_fluid_velocity(particle_positions)
        
        # Compute forces for each particle
        forces = []
        for i, (pos, vel, radius) in enumerate(zip(particle_positions, 
                                                 fluid_velocities, 
                                                 particle_radii)):
            # Drag force (adjusted for geotechnical parameters)
            drag_force = self._compute_drag_force(vel, radius)
            
            # Pressure gradient force (adjusted for geotechnical parameters)
            pressure_force = self._compute_pressure_force(pos)
            
            # Buoyancy force (adjusted for geotechnical parameters)
            buoyancy_force = self._compute_buoyancy_force(radius)
            
            # Erosion force (calibrated with geotechnical parameters)
            erosion_force = self._compute_erosion_force(vel, radius)
            
            # Total force
            total_force = drag_force + pressure_force + buoyancy_force + erosion_force
            
            # Track fluid-particle interaction
            self._track_fluid_particle_interaction(pos, vel, total_force)
            
            forces.append(total_force)
        
        return forces

    def _interpolate_fluid_velocity(self, positions: List[np.ndarray]) -> List[np.ndarray]:
        """Interpolate fluid velocity to particle positions."""
        velocities = []
        for pos in positions:
            if self.interpolation_method == 'linear':
                vel = self._linear_interpolation(pos)
            else:
                vel = self._nearest_neighbor_interpolation(pos)
            velocities.append(vel)
        return velocities

    def _linear_interpolation(self, position: np.ndarray) -> np.ndarray:
        """Linear interpolation of fluid velocity."""
        # Get fluid data
        fluid_data = self.cfd_solver.get_fluid_data()
        if fluid_data is None or 'velocity_field' not in fluid_data:
            raise ValueError("Fluid data not available for interpolation")
            
        velocity_field = fluid_data['velocity_field']
        if velocity_field is None:
            raise ValueError("Velocity field not initialized")
            
        # Convert position to grid indices
        grid_size = velocity_field.shape
        if not (0 <= position[0] <= 1 and 0 <= position[1] <= 1 and 0 <= position[2] <= 1):
            raise ValueError(f"Position {position} is outside the domain [0,1]^3")
        
        indices = position * np.array(grid_size)
        
        # Get surrounding grid points
        i0, j0, k0 = np.floor(indices).astype(int)
        i1, j1, k1 = np.ceil(indices).astype(int)
        
        # Interpolate
        weights = indices - np.floor(indices)
        v000 = velocity_field[i0, j0, k0]
        v001 = velocity_field[i0, j0, k1]
        v010 = velocity_field[i0, j1, k0]
        v011 = velocity_field[i0, j1, k1]
        v100 = velocity_field[i1, j0, k0]
        v101 = velocity_field[i1, j0, k1]
        v110 = velocity_field[i1, j1, k0]
        v111 = velocity_field[i1, j1, k1]
        
        v00 = v000 * (1 - weights[2]) + v001 * weights[2]
        v01 = v010 * (1 - weights[2]) + v011 * weights[2]
        v10 = v100 * (1 - weights[2]) + v101 * weights[2]
        v11 = v110 * (1 - weights[2]) + v111 * weights[2]
        
        v0 = v00 * (1 - weights[1]) + v01 * weights[1]
        v1 = v10 * (1 - weights[1]) + v11 * weights[1]
        
        velocity = v0 * (1 - weights[0]) + v1 * weights[0]
        
        return velocity

    def _nearest_neighbor_interpolation(self, position: np.ndarray) -> np.ndarray:
        """Nearest neighbor interpolation of fluid velocity."""
        fluid_data = self.cfd_solver.get_fluid_data()
        velocity_field = fluid_data['velocity_field']
        
        # Convert position to grid indices
        grid_size = velocity_field.shape
        indices = np.round(position * np.array(grid_size)).astype(int)
        
        # Get velocity at nearest grid point
        velocity = velocity_field[indices[0], indices[1], indices[2]]
        
        return velocity

    def _compute_drag_force(self, fluid_velocity: np.ndarray,
                          particle_radius: float) -> np.ndarray:
        """Compute drag force on particle (adjusted for geotechnical parameters)."""
        # Reynolds number (adjusted for geotechnical parameters)
        Re = (np.linalg.norm(fluid_velocity) * particle_radius * 
              self.config['cfd']['fluid_density'] / 
              self.config['cfd']['fluid_viscosity'])
        
        # Adjust Reynolds number based on clay content
        clay_factor = 1.0 + 0.1 * (self.geotech_params['clay_content'] / 20.0)
        Re *= clay_factor
        
        # Drag coefficient
        if Re < 1:
            Cd = 24/Re
        else:
            Cd = 24/Re * (1 + 0.15 * Re**0.687)
        
        # Adjust drag coefficient based on water content
        water_factor = 1.0 - 0.2 * (self.geotech_params['water_content'] / 20.0)
        Cd *= water_factor
        
        # Drag force
        drag_force = (0.5 * self.config['cfd']['fluid_density'] * 
                     np.pi * particle_radius**2 * 
                     Cd * np.linalg.norm(fluid_velocity) * 
                     fluid_velocity)
        
        return drag_force

    def _compute_pressure_force(self, position: np.ndarray) -> np.ndarray:
        """Compute pressure gradient force on particle (adjusted for geotechnical parameters)."""
        # Get pressure gradient from configuration
        pressure_gradient = np.array(self.config['cfd']['pressure_gradient'])
        
        # Adjust pressure gradient based on permeability
        perm_factor = self.geotech_params['permeability'] / 1e-6  # Normalize to 1e-6 m/s
        pressure_gradient *= perm_factor
        
        # Pressure force
        pressure_force = -pressure_gradient * (4/3 * np.pi * 
                                             self.config['dem']['particle_radius']**3)
        
        return pressure_force

    def _compute_buoyancy_force(self, particle_radius: float) -> np.ndarray:
        """Compute buoyancy force on particle (adjusted for geotechnical parameters)."""
        # Adjust fluid density based on water content
        water_factor = 1.0 + 0.1 * (self.geotech_params['water_content'] / 20.0)
        adjusted_density = self.config['cfd']['fluid_density'] * water_factor
        
        # Buoyancy force
        buoyancy_force = (adjusted_density * 
                         self.config['simulation']['gravity'] * 
                         (4/3 * np.pi * particle_radius**3))
        
        return buoyancy_force

    def _compute_erosion_force(self, fluid_velocity: np.ndarray,
                             particle_radius: float) -> np.ndarray:
        """Compute erosion force based on fluid velocity (calibrated with geotechnical parameters)."""
        velocity_magnitude = np.linalg.norm(fluid_velocity)
        
        # Check if velocity is below threshold for erosion
        if velocity_magnitude < 1e-6:  # Very low velocity threshold
            return np.zeros(3)
            
        # Compute shear stress (adjusted for geotechnical parameters)
        shear_stress = self._compute_fluid_shear_stress(fluid_velocity, particle_radius)
        
        # Check if shear stress exceeds critical value
        if shear_stress < self.critical_shear_stress:
            return np.zeros(3)
            
        # Compute erosion force (calibrated with geotechnical parameters)
        force_magnitude = self.erosion_rate_coefficient * shear_stress * np.pi * particle_radius**2
        
        # Force direction is opposite to fluid velocity
        return -force_magnitude * fluid_velocity / velocity_magnitude

    def _compute_fluid_shear_stress(self, fluid_velocity: np.ndarray,
                                  particle_radius: float) -> float:
        """Compute fluid-induced shear stress on particle (adjusted for geotechnical parameters)."""
        # Reynolds number (adjusted for geotechnical parameters)
        Re = (np.linalg.norm(fluid_velocity) * particle_radius * 
              self.config['cfd']['fluid_density'] / 
              self.config['cfd']['fluid_viscosity'])
        
        # Adjust Reynolds number based on clay content
        clay_factor = 1.0 + 0.1 * (self.geotech_params['clay_content'] / 20.0)
        Re *= clay_factor
        
        # Drag coefficient
        if Re < 1:
            Cd = 24/Re
        else:
            Cd = 24/Re * (1 + 0.15 * Re**0.687)
        
        # Adjust drag coefficient based on water content
        water_factor = 1.0 - 0.2 * (self.geotech_params['water_content'] / 20.0)
        Cd *= water_factor
        
        # Fluid shear stress (adjusted for geotechnical parameters)
        shear_stress = 0.5 * self.config['cfd']['fluid_density'] * np.linalg.norm(fluid_velocity)**2 * Cd
        
        # Adjust shear stress based on Cu
        cu_factor = 1.0 + 0.1 * (self.geotech_params['Cu'] / 10.0)
        shear_stress *= cu_factor
        
        return shear_stress

    def _track_fluid_particle_interaction(self, position: np.ndarray,
                                        velocity: np.ndarray,
                                        force: np.ndarray):
        """Track fluid-particle interaction for analysis."""
        self.fluid_particle_interactions.append({
            'position': position,
            'velocity': velocity,
            'force': force,
            'time': len(self.fluid_particle_interactions) * self.config['simulation']['time_step']
        })

    def step(self):
        """Perform one time step of the coupled simulation."""
        # Update fluid state
        self.cfd_solver.step()
        
        # Get particle data
        if self.coarse_model:
            particle_data = self.coarse_model.get_coarse_particle_data()
        else:
            particle_data = self.dem_solver.get_particle_data()
        
        # Compute fluid forces
        fluid_forces = self.compute_fluid_forces(
            particle_data['positions'],
            [self.config['dem']['particle_radius']] * len(particle_data['positions'])
        )
        
        # Update particle states and track erosion
        if self.coarse_model:
            # Compute bond forces
            bond_forces = self.coarse_model.compute_coarse_bond_forces(
                self.cfd_solver.get_fluid_data()['velocity_field'].mean(axis=(0,1,2)),
                self.config['simulation']['time_step']
            )
            
            # Update particle states
            self.coarse_model.update_particle_states(
                bond_forces,
                self.config['simulation']['time_step']
            )
            
            # Track bond health
            self._track_bond_health()
        else:
            # Update DEM particles
            self.dem_solver.step()
            
            # Track erosion
            self._track_erosion(particle_data, fluid_forces)
        
        logger.debug("Completed one time step of coupled simulation")

    def _track_bond_health(self):
        """Track bond health for coarse-grained particles."""
        if self.coarse_model:
            bond_health = self.coarse_model.bond_model.bond_health
            self.bond_health_history.append(bond_health)
            
            # Log significant bond degradation
            if bond_health < 0.5 and bond_health > 0.4:
                logger.warning("Significant bond degradation detected")

    def _track_erosion(self, particle_data: Dict, fluid_forces: List[np.ndarray]):
        """Track particle erosion."""
        for i, (pos, force) in enumerate(zip(particle_data['positions'], fluid_forces)):
            # Check if particle is being eroded
            if np.linalg.norm(force) > self.critical_shear_stress:
                self.eroded_particles.append({
                    'position': pos,
                    'force': force,
                    'time': len(self.eroded_particles) * self.config['simulation']['time_step']
                })
                
                # Log erosion event
                logger.info(f"Particle erosion detected at position {pos}")

    def get_erosion_statistics(self) -> Dict:
        """Get statistics about particle erosion."""
        if not self.eroded_particles:
            return {
                'total_eroded': 0,
                'erosion_rate': 0.0,
                'average_force': np.zeros(3)
            }
        
        # Compute statistics
        total_eroded = len(self.eroded_particles)
        erosion_rate = total_eroded / (len(self.eroded_particles) * 
                                     self.config['simulation']['time_step'])
        average_force = np.mean([p['force'] for p in self.eroded_particles], axis=0)
        
        return {
            'total_eroded': total_eroded,
            'erosion_rate': erosion_rate,
            'average_force': average_force
        }

    def get_fluid_particle_interactions(self) -> List[Dict]:
        """Get history of fluid-particle interactions."""
        return self.fluid_particle_interactions

    def get_bond_health_history(self) -> List[float]:
        """Get history of bond health values."""
        return self.bond_health_history
