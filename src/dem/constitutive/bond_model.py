"""
Constitutive model for particle bonding with seepage erosion effects.
This model extends the parallel bond model to account for fluid-induced bond degradation.
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class SeepageErosionBondModel:
    def __init__(self, config: Dict):
        """Initialize the seepage erosion bond model with configuration parameters."""
        self.config = config
        
        # Basic bond parameters
        self.youngs_modulus = config['dem']['youngs_modulus']
        self.poisson_ratio = config['dem']['poisson_ratio']
        self.bond_radius = config['dem']['particle_radius']  # Initial bond radius
        self.bond_strength = config['dem'].get('bond_strength', 1e6)  # Initial bond strength
        
        # Seepage erosion parameters
        self.critical_shear_stress = config['erosion']['critical_shear_stress']
        self.erosion_rate_coefficient = config['erosion']['erosion_rate_coefficient']
        self.fluid_viscosity = config['cfd']['fluid_viscosity']
        self.fluid_density = config['cfd']['fluid_density']
        
        # State variables
        self.bond_health = 1.0  # 1.0 = fully intact, 0.0 = fully degraded
        self.accumulated_damage = 0.0
        
        logger.info("Seepage erosion bond model initialized")

    def compute_bond_stress(self, normal_force: np.ndarray, 
                          shear_force: np.ndarray,
                          moment: np.ndarray) -> Tuple[float, float]:
        """Compute normal and shear stresses in the bond."""
        # Normal stress
        normal_stress = np.linalg.norm(normal_force) / (np.pi * self.bond_radius**2)
        
        # Shear stress (including moment contribution)
        shear_stress = (np.linalg.norm(shear_force) + 
                       np.linalg.norm(moment) * self.bond_radius / 
                       (np.pi * self.bond_radius**4/4))
        
        return normal_stress, shear_stress

    def compute_fluid_shear_stress(self, fluid_velocity: np.ndarray, 
                                 particle_radius: float) -> float:
        """Compute fluid-induced shear stress on the bond."""
        # Reynolds number
        Re = (np.linalg.norm(fluid_velocity) * particle_radius * 
              self.fluid_density / self.fluid_viscosity)
        
        # Drag coefficient (empirical correlation)
        if Re < 1:
            Cd = 24/Re
        else:
            Cd = 24/Re * (1 + 0.15 * Re**0.687)
        
        # Fluid shear stress
        shear_stress = 0.5 * self.fluid_density * np.linalg.norm(fluid_velocity)**2 * Cd
        
        return shear_stress

    def update_bond_health(self, fluid_velocity: np.ndarray, 
                          particle_radius: float,
                          time_step: float) -> float:
        """Update bond health based on fluid-induced erosion."""
        # Compute fluid shear stress
        fluid_shear_stress = self.compute_fluid_shear_stress(fluid_velocity, particle_radius)
        
        # Compute erosion rate
        if fluid_shear_stress > self.critical_shear_stress:
            erosion_rate = (self.erosion_rate_coefficient * 
                          (fluid_shear_stress - self.critical_shear_stress))
        else:
            erosion_rate = 0.0
        
        # Update accumulated damage
        self.accumulated_damage += erosion_rate * time_step
        
        # Update bond health
        self.bond_health = max(0.0, 1.0 - self.accumulated_damage)
        
        return self.bond_health

    def compute_bond_forces(self, normal_force: np.ndarray,
                          shear_force: np.ndarray,
                          moment: np.ndarray,
                          fluid_velocity: np.ndarray,
                          particle_radius: float,
                          time_step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute bond forces considering seepage erosion effects."""
        # Update bond health
        self.update_bond_health(fluid_velocity, particle_radius, time_step)
        
        # Compute stresses
        normal_stress, shear_stress = self.compute_bond_stress(normal_force, shear_force, moment)
        
        # Apply bond health to forces
        effective_normal_force = normal_force * self.bond_health
        effective_shear_force = shear_force * self.bond_health
        effective_moment = moment * self.bond_health
        
        return effective_normal_force, effective_shear_force, effective_moment 