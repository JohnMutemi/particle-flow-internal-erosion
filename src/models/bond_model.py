"""
Seepage bond model for simulating particle bond degradation under fluid flow.

This model uses geotechnical parameters such as density, specific gravity, water content, Cu, Cc, clay content, permeability, etc.,
to initialize and calibrate the bond strength and erosion law. The bond degradation law is implemented based on local fluid shear stress and pressure effects.

References:
- Gu et al. (2019), Acta Geotechnica
- Wang et al. (2020), Evolution mechanism of seepage damage in fault fracture zone
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
        
        # Load geotechnical parameters
        self.clay_content = config['geotechnical']['clay_content']  # Percentage
        self.water_content = config['geotechnical']['water_content']  # Percentage
        self.Cu = config['geotechnical']['Cu']  # Uniformity coefficient
        self.Cc = config['geotechnical']['Cc']  # Curvature coefficient
        self.cohesion = config['geotechnical']['cohesion']  # kPa
        self.permeability = config['geotechnical']['permeability']  # m/s
        
        # Load model parameters
        self.bond_strength = self._calibrate_bond_strength()  # Calibrated using geotechnical parameters
        self.fluid_viscosity = config['cfd']['fluid_viscosity']
        self.fluid_density = config['cfd']['fluid_density']
        self.erosion_rate = self._calibrate_erosion_rate()  # Calibrated using geotechnical parameters
        self.critical_shear_stress = self._calibrate_critical_shear_stress()
        
        # Initialize state variables
        self.bond_health = 1.0  # 1.0 = fully intact, 0.0 = fully degraded
        self.accumulated_damage = 0.0
        self.degradation_history = []
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initialized seepage bond model with geotechnical parameters")
    
    def _calibrate_bond_strength(self) -> float:
        """Calibrate initial bond strength based on geotechnical parameters."""
        # Base strength from cohesion
        base_strength = self.cohesion * 1000  # Convert kPa to Pa
        
        # Clay content effect (increases strength)
        clay_factor = 1.0 + 0.1 * (self.clay_content / 20.0)  # Normalized to 20% clay
        
        # Water content effect (decreases strength)
        water_factor = 1.0 - 0.2 * (self.water_content / 20.0)  # Normalized to 20% water
        
        # Cu effect (well-graded soils have higher strength)
        cu_factor = 1.0 + 0.1 * (self.Cu / 10.0)  # Normalized to Cu=10
        
        return base_strength * clay_factor * water_factor * cu_factor
    
    def _calibrate_erosion_rate(self) -> float:
        """Calibrate erosion rate based on geotechnical parameters."""
        # Base erosion rate from config
        base_rate = self.config['erosion']['erosion_rate_coefficient']
        
        # Clay content effect (reduces erosion)
        clay_factor = 1.0 - 0.5 * (self.clay_content / 20.0)
        
        # Cu effect (well-graded soils resist erosion better)
        cu_factor = 1.0 - 0.3 * (self.Cu / 10.0)
        
        # Cc effect (poorly graded soils erode faster)
        cc_factor = 1.0 + 0.2 * (abs(self.Cc - 1.0))
        
        return base_rate * clay_factor * cu_factor * cc_factor
    
    def _calibrate_critical_shear_stress(self) -> float:
        """Calibrate critical shear stress based on geotechnical parameters."""
        # Base critical stress from config
        base_stress = self.config['erosion']['critical_shear_stress']
        
        # Clay content effect (increases critical stress)
        clay_factor = 1.0 + 0.5 * (self.clay_content / 20.0)
        
        # Water content effect (decreases critical stress)
        water_factor = 1.0 - 0.3 * (self.water_content / 20.0)
        
        # Cu effect (well-graded soils have higher critical stress)
        cu_factor = 1.0 + 0.2 * (self.Cu / 10.0)
        
        return base_stress * clay_factor * water_factor * cu_factor
    
    def compute_bond_degradation(self, 
                               fluid_velocity: np.ndarray,
                               pressure_gradient: np.ndarray,
                               bond_radius: float,
                               current_strength: float,
                               time_step: float) -> Tuple[float, float]:
        """Compute bond degradation under fluid flow.
        
        Args:
            fluid_velocity: Fluid velocity vector at particle position
            pressure_gradient: Pressure gradient vector at particle position
            bond_radius: Current bond radius
            current_strength: Current bond strength
            time_step: Time step for degradation
            
        Returns:
            Tuple of (degradation_rate, new_strength)
        """
        # Compute fluid shear stress
        shear_stress = self._compute_shear_stress(fluid_velocity, bond_radius)
        
        # Compute pressure effect
        pressure_effect = self._compute_pressure_effect(pressure_gradient, bond_radius)
        
        # Compute total degradation rate
        degradation_rate = self._compute_degradation_rate(
            shear_stress, pressure_effect, current_strength)
        
        # Update bond health
        self.bond_health = self._update_bond_health(degradation_rate, time_step)
        
        # Update bond strength
        new_strength = self._update_bond_strength(current_strength, degradation_rate)
        
        # Store degradation history
        self.degradation_history.append({
            'time': time_step,
            'shear_stress': shear_stress,
            'pressure_effect': pressure_effect,
            'degradation_rate': degradation_rate,
            'bond_health': self.bond_health,
            'strength': new_strength
        })
        
        return degradation_rate, new_strength
    
    def _compute_shear_stress(self, 
                            fluid_velocity: np.ndarray,
                            bond_radius: float) -> float:
        """Compute fluid shear stress on bond."""
        velocity_magnitude = np.linalg.norm(fluid_velocity)
        
        # Compute Reynolds number
        reynolds = 2 * bond_radius * velocity_magnitude * self.fluid_density / self.fluid_viscosity
        
        # Compute shear stress based on flow regime
        if reynolds < 1:
            # Stokes flow
            shear_stress = 3 * self.fluid_viscosity * velocity_magnitude / bond_radius
        elif reynolds < 1000:
            # Transitional flow
            cd = 24/reynolds * (1 + 0.15 * reynolds**0.687)
            shear_stress = 0.5 * self.fluid_density * velocity_magnitude**2 * cd
        else:
            # Turbulent flow
            shear_stress = 0.5 * self.fluid_density * velocity_magnitude**2
        
        return shear_stress
    
    def _compute_pressure_effect(self, 
                               pressure_gradient: np.ndarray,
                               bond_radius: float) -> float:
        """Compute pressure effect on bond."""
        grad_magnitude = np.linalg.norm(pressure_gradient)
        
        # Consider clay content effect on pressure sensitivity
        clay_factor = 1.0 - 0.3 * (self.clay_content / 20.0)
        
        # Compute pressure effect
        pressure_effect = np.pi * bond_radius**2 * grad_magnitude * clay_factor
        
        return pressure_effect
    
    def _compute_degradation_rate(self, 
                                shear_stress: float,
                                pressure_effect: float,
                                current_strength: float) -> float:
        """Compute bond degradation rate using enhanced erosion law."""
        # Compute total stress
        total_stress = shear_stress + pressure_effect
        
        # Check if stress exceeds critical value
        if total_stress < self.critical_shear_stress:
            return 0.0
        
        # Compute degradation rate using enhanced power law
        stress_excess = total_stress - self.critical_shear_stress
        
        # Consider clay content effect on degradation
        clay_factor = 1.0 - 0.5 * (self.clay_content / 20.0)
        
        # Consider water content effect
        water_factor = 1.0 + 0.3 * (self.water_content / 20.0)
        
        # Compute degradation rate
        degradation_rate = (self.erosion_rate * clay_factor * water_factor * 
                          stress_excess**2 / current_strength)
        
        return degradation_rate
    
    def _update_bond_health(self, degradation_rate: float, time_step: float) -> float:
        """Update bond health based on degradation rate."""
        # Update accumulated damage
        self.accumulated_damage += degradation_rate * time_step
        
        # Update bond health
        self.bond_health = max(0.0, 1.0 - self.accumulated_damage)
        
        return self.bond_health
    
    def _update_bond_strength(self, 
                            current_strength: float,
                            degradation_rate: float) -> float:
        """Update bond strength based on degradation."""
        # Update strength
        new_strength = current_strength * (1 - degradation_rate)
        
        # Ensure strength doesn't go below zero
        return max(0.0, new_strength)
    
    def get_bond_health(self) -> float:
        """Get current bond health."""
        return self.bond_health
    
    def get_degradation_history(self) -> list:
        """Get degradation history."""
        return self.degradation_history
    
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