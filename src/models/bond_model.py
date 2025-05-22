"""
Seepage bond model for simulating particle bond degradation under fluid flow.

This model uses geotechnical parameters such as density, specific gravity, water content, Cu, Cc, clay content, permeability, etc.,
to initialize and calibrate the bond strength and erosion law. The bond degradation law is implemented based on local fluid shear stress and pressure effects.

References:
- Gu et al. (2019), Acta Geotechnica
- User's personal geotechnical parameters (see config)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

class SeepageBondModel:
    def __init__(self, config: Dict):
        """Initialize the seepage bond model.
        
        Args:
            config: Configuration dictionary containing model parameters, including geotechnical values such as density, specific gravity, water content, Cu, Cc, clay content, permeability, etc.
        """
        self.config = config
        
        # Load model parameters from config (geotechnical values)
        self.bond_strength = config['dem']['bond_strength']  # Calibrated using cohesion, clay content, etc.
        self.fluid_viscosity = config['cfd']['fluid_viscosity']  # From permeability and fluid properties
        self.fluid_density = config['cfd']['fluid_density']
        self.erosion_rate = config['erosion']['erosion_rate_coefficient']  # Calibrated using triaxial/erosion tests
        self.critical_shear_stress = config['erosion']['critical_shear_stress']  # Related to clay content, Cu, Cc
        # Additional geotechnical parameters can be accessed as needed
        # e.g., config['sample']['density'], config['particles']['clay_content_mass_percent'], etc.
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initialized seepage bond model with geotechnical parameters")
    
    def compute_bond_degradation(self, 
                               fluid_velocity: np.ndarray,
                               pressure_gradient: np.ndarray,
                               bond_radius: float,
                               current_strength: float) -> Tuple[float, float]:
        """Compute bond degradation under fluid flow.
        
        Args:
            fluid_velocity: Fluid velocity vector at particle position
            pressure_gradient: Pressure gradient vector at particle position
            bond_radius: Current bond radius
            current_strength: Current bond strength
            
        Returns:
            Tuple of (degradation_rate, new_strength)
        """
        # Compute fluid shear stress (see _compute_shear_stress)
        shear_stress = self._compute_shear_stress(
            fluid_velocity, bond_radius)
        
        # Compute pressure effect (see _compute_pressure_effect)
        pressure_effect = self._compute_pressure_effect(
            pressure_gradient, bond_radius)
        
        # Compute total degradation rate (see _compute_degradation_rate)
        degradation_rate = self._compute_degradation_rate(
            shear_stress, pressure_effect, current_strength)
        
        # Update bond strength (see _update_bond_strength)
        new_strength = self._update_bond_strength(
            current_strength, degradation_rate)
        
        return degradation_rate, new_strength
    
    def _compute_shear_stress(self, 
                            fluid_velocity: np.ndarray,
                            bond_radius: float) -> float:
        """Compute fluid shear stress on bond.
        
        Args:
            fluid_velocity: Fluid velocity vector
            bond_radius: Bond radius
            
        Returns:
            Shear stress magnitude
        """
        # Compute velocity gradient
        velocity_magnitude = np.linalg.norm(fluid_velocity)
        
        # Compute Reynolds number (depends on fluid_density, fluid_viscosity)
        reynolds = 2 * bond_radius * velocity_magnitude * self.fluid_density / self.fluid_viscosity
        
        # Compute shear stress based on flow regime
        if reynolds < 1:
            # Stokes flow
            shear_stress = 3 * self.fluid_viscosity * velocity_magnitude / bond_radius
        else:
            # Turbulent flow
            shear_stress = 0.5 * self.fluid_density * velocity_magnitude**2
        
        return shear_stress
    
    def _compute_pressure_effect(self, 
                               pressure_gradient: np.ndarray,
                               bond_radius: float) -> float:
        """Compute pressure effect on bond.
        
        Args:
            pressure_gradient: Pressure gradient vector
            bond_radius: Bond radius
            
        Returns:
            Pressure effect magnitude
        """
        # Compute pressure gradient magnitude
        grad_magnitude = np.linalg.norm(pressure_gradient)
        
        # Compute pressure effect (area * grad)
        pressure_effect = np.pi * bond_radius**2 * grad_magnitude
        
        return pressure_effect
    
    def _compute_degradation_rate(self, 
                                shear_stress: float,
                                pressure_effect: float,
                                current_strength: float) -> float:
        """Compute bond degradation rate using erosion law.
        
        Args:
            shear_stress: Fluid shear stress
            pressure_effect: Pressure effect
            current_strength: Current bond strength
            
        Returns:
            Degradation rate
        """
        # Compute total stress
        total_stress = shear_stress + pressure_effect
        
        # Check if stress exceeds critical value (from config, related to clay content, Cu, Cc)
        if total_stress < self.critical_shear_stress:
            return 0.0
        
        # Compute degradation rate using power law (Gu et al. 2019)
        degradation_rate = self.erosion_rate * \
            (total_stress - self.critical_shear_stress)**2 / current_strength
        
        return degradation_rate
    
    def _update_bond_strength(self, 
                            current_strength: float,
                            degradation_rate: float) -> float:
        """Update bond strength based on degradation.
        
        Args:
            current_strength: Current bond strength
            degradation_rate: Degradation rate
            
        Returns:
            Updated bond strength
        """
        # Update strength
        new_strength = current_strength * (1 - degradation_rate)
        
        # Ensure strength doesn't go below zero
        return max(0.0, new_strength)
    
    def compute_bond_force(self, 
                          bond_vector: np.ndarray,
                          bond_strength: float,
                          bond_radius: float) -> np.ndarray:
        """Compute bond force between particles.
        
        Args:
            bond_vector: Vector from particle 1 to particle 2
            bond_strength: Current bond strength
            bond_radius: Bond radius
            
        Returns:
            Bond force vector
        """
        # Compute bond length
        bond_length = np.linalg.norm(bond_vector)
        
        # Compute bond area
        bond_area = np.pi * bond_radius**2
        
        # Compute normal force (proportional to bond strength, which is calibrated using geotechnical parameters)
        normal_force = bond_strength * bond_area
        
        # Compute force vector
        force = normal_force * bond_vector / bond_length
        
        return force
    
    def compute_bond_moment(self, 
                          relative_rotation: np.ndarray,
                          bond_strength: float,
                          bond_radius: float) -> np.ndarray:
        """Compute bond moment between particles.
        
        Args:
            relative_rotation: Relative rotation vector
            bond_strength: Current bond strength
            bond_radius: Bond radius
            
        Returns:
            Bond moment vector
        """
        # Compute moment of inertia
        moment_of_inertia = np.pi * bond_radius**4 / 4
        
        # Compute moment
        moment = bond_strength * moment_of_inertia * relative_rotation
        
        return moment 