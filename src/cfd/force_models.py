"""
Advanced force models for CFD-DEM coupling.
Includes models for different flow regimes and particle interactions.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ForceModels:
    def __init__(self, config: Dict):
        """Initialize force models with configuration."""
        # Validate required configuration parameters
        required_params = {
            'cfd': ['fluid_density', 'fluid_viscosity'],
            'dem': ['particle_radius']
        }
        
        for section, params in required_params.items():
            if section not in config:
                raise ValueError(f"Missing configuration section: {section}")
            for param in params:
                if param not in config[section]:
                    raise ValueError(f"Missing parameter {param} in {section} section")
        
        self.config = config
        self.fluid_density = config['cfd']['fluid_density']
        self.fluid_viscosity = config['cfd']['fluid_viscosity']
        self.particle_radius = config['dem']['particle_radius']
        
        # Flow regime parameters
        self.reynolds_threshold = 1000  # Transition to turbulent flow
        self.mach_threshold = 0.3  # Transition to compressible flow
        self.sound_speed = 343.0  # Speed of sound in air at 20°C (m/s)
        
        logger.info("Force models initialized")

    def compute_drag_force(self, particle_velocity: np.ndarray,
                          fluid_velocity: np.ndarray,
                          particle_radius: float) -> np.ndarray:
        """Compute drag force using regime-appropriate model."""
        # Input validation
        if not isinstance(particle_velocity, np.ndarray) or particle_velocity.shape != (3,):
            raise ValueError("particle_velocity must be a 3D numpy array")
        if not isinstance(fluid_velocity, np.ndarray) or fluid_velocity.shape != (3,):
            raise ValueError("fluid_velocity must be a 3D numpy array")
        if not isinstance(particle_radius, (int, float)) or particle_radius <= 0:
            raise ValueError("particle_radius must be a positive number")
            
        relative_velocity = fluid_velocity - particle_velocity
        velocity_magnitude = np.linalg.norm(relative_velocity)
        
        if velocity_magnitude == 0:
            return np.zeros(3)
        
        # Calculate Reynolds number
        reynolds = self._compute_reynolds_number(velocity_magnitude, particle_radius)
        
        # Select drag coefficient based on flow regime
        drag_coefficient = self._get_drag_coefficient(reynolds)
        
        # Compute drag force with improved accuracy
        if reynolds < 1:  # Stokes flow
            drag_force = 6 * np.pi * self.fluid_viscosity * particle_radius * relative_velocity
        elif reynolds < self.reynolds_threshold:  # Intermediate flow
            # Schiller-Naumann correlation
            drag_force = 0.5 * self.fluid_density * np.pi * particle_radius**2 * \
                        drag_coefficient * velocity_magnitude * relative_velocity
        else:  # Turbulent flow
            # Standard drag law for turbulent flow
            drag_force = 0.5 * self.fluid_density * np.pi * particle_radius**2 * \
                        0.44 * velocity_magnitude * relative_velocity
        
        return drag_force

    def compute_pressure_force(self, pressure_gradient: np.ndarray,
                             particle_radius: float) -> np.ndarray:
        """Compute pressure force with compressibility effects."""
        # Input validation
        if not isinstance(pressure_gradient, np.ndarray) or pressure_gradient.shape != (3,):
            raise ValueError("pressure_gradient must be a 3D numpy array")
        if not isinstance(particle_radius, (int, float)) or particle_radius <= 0:
            raise ValueError("particle_radius must be a positive number")
            
        # Basic pressure force
        pressure_force = -np.pi * particle_radius**2 * pressure_gradient
        
        # Add compressibility effects if needed
        if self._is_compressible_flow(pressure_gradient):
            pressure_force *= self._compressibility_correction(pressure_gradient)
        
        return pressure_force

    def compute_buoyancy_force(self, particle_radius: float,
                             gravity: np.ndarray) -> np.ndarray:
        """Compute buoyancy force with density variations."""
        # Input validation
        if not isinstance(particle_radius, (int, float)) or particle_radius <= 0:
            raise ValueError("particle_radius must be a positive number")
        if not isinstance(gravity, np.ndarray) or gravity.shape != (3,):
            raise ValueError("gravity must be a 3D numpy array")
            
        particle_volume = 4/3 * np.pi * particle_radius**3
        buoyancy_force = -self.fluid_density * particle_volume * gravity
        
        return buoyancy_force

    def compute_lift_force(self, particle_velocity: np.ndarray,
                          fluid_velocity: np.ndarray,
                          fluid_vorticity: np.ndarray,
                          particle_radius: float) -> np.ndarray:
        """Compute lift force using Saffman and Magnus effects."""
        relative_velocity = fluid_velocity - particle_velocity
        velocity_magnitude = np.linalg.norm(relative_velocity)
        
        if velocity_magnitude == 0:
            return np.zeros(3)
        
        # Saffman lift force
        reynolds = self._compute_reynolds_number(velocity_magnitude, particle_radius)
        shear_reynolds = self._compute_shear_reynolds(fluid_vorticity, particle_radius)
        
        saffman_lift = self._compute_saffman_lift(relative_velocity, fluid_vorticity,
                                                particle_radius, reynolds, shear_reynolds)
        
        # Magnus lift force
        magnus_lift = self._compute_magnus_lift(relative_velocity, particle_radius)
        
        return saffman_lift + magnus_lift

    def compute_virtual_mass_force(self, particle_velocity: np.ndarray,
                                 fluid_velocity: np.ndarray,
                                 fluid_acceleration: np.ndarray,
                                 particle_radius: float) -> np.ndarray:
        """Compute virtual mass force with acceleration effects."""
        relative_acceleration = fluid_acceleration - np.zeros_like(particle_velocity)  # Particle acceleration
        particle_volume = 4/3 * np.pi * particle_radius**3
        
        # Virtual mass coefficient (typically 0.5 for spherical particles)
        virtual_mass_coefficient = 0.5
        
        virtual_mass_force = virtual_mass_coefficient * self.fluid_density * \
                           particle_volume * relative_acceleration
        
        return virtual_mass_force

    def _compute_reynolds_number(self, velocity: float, particle_radius: float) -> float:
        """Compute particle Reynolds number."""
        return 2 * particle_radius * velocity * self.fluid_density / self.fluid_viscosity

    def _compute_shear_reynolds(self, vorticity: np.ndarray,
                              particle_radius: float) -> float:
        """Compute shear Reynolds number."""
        shear_rate = np.linalg.norm(vorticity)
        return 4 * particle_radius**2 * shear_rate * self.fluid_density / self.fluid_viscosity

    def _get_drag_coefficient(self, reynolds: float) -> float:
        """Get drag coefficient based on flow regime."""
        if reynolds < 1:
            # Stokes flow
            return 24 / reynolds
        elif reynolds < 1000:
            # Schiller-Naumann correlation for intermediate flow
            return 24 / reynolds * (1 + 0.15 * reynolds**0.687)
        else:
            # Turbulent flow
            return 0.44

    def _is_compressible_flow(self, pressure_gradient: np.ndarray) -> bool:
        """Check if flow is compressible based on pressure gradient."""
        return np.linalg.norm(pressure_gradient) > 5000.0  # Adjusted threshold

    def _compressibility_correction(self, pressure_gradient: np.ndarray) -> float:
        """Compute compressibility correction factor."""
        velocity_magnitude = np.linalg.norm(pressure_gradient) / (self.fluid_density * self.sound_speed)
        mach_number = velocity_magnitude / self.sound_speed
        return 1 + 0.25 * (mach_number / self.mach_threshold)**2

    def _compute_saffman_lift(self, relative_velocity: np.ndarray,
                            vorticity: np.ndarray,
                            particle_radius: float,
                            reynolds: float,
                            shear_reynolds: float) -> np.ndarray:
        """Compute Saffman lift force."""
        if reynolds == 0 or shear_reynolds == 0:
            return np.zeros(3)
        
        # Saffman lift coefficient
        lift_coefficient = 1.615 * np.sqrt(shear_reynolds / reynolds)
        
        # Compute lift force
        lift_force = lift_coefficient * self.fluid_density * \
                    np.pi * particle_radius**2 * \
                    np.cross(relative_velocity, vorticity)
        
        return lift_force

    def _compute_magnus_lift(self, relative_velocity: np.ndarray,
                           particle_radius: float) -> np.ndarray:
        """Compute Magnus lift force."""
        # Simplified Magnus effect
        magnus_coefficient = 0.5
        
        # Compute lift force
        lift_force = magnus_coefficient * self.fluid_density * \
                    np.pi * particle_radius**2 * \
                    np.cross(relative_velocity, np.array([0, 0, 1]))  # Assuming rotation around z-axis
        
        return lift_force 