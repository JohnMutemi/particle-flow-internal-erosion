"""
Coupling manager for CFD-DEM simulations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from scipy import interpolate

class CouplingManager:
    def __init__(self, config: Dict):
        """Initialize the coupling manager.
        
        Args:
            config: Configuration dictionary containing coupling parameters
        """
        self.config = config
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load coupling parameters
        self.coupling_params = self._load_coupling_params()
        
        # Initialize output directory
        self.output_dir = Path(config['output']['directory']) / 'coupling'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Initialized coupling manager")
    
    def _load_coupling_params(self) -> Dict:
        """Load coupling parameters from configuration.
        
        Returns:
            Dictionary of coupling parameters
        """
        params = {
            'coupling_interval': self.config['coupling'].get('interval', 1),
            'interpolation_order': self.config['coupling'].get('interpolation_order', 1),
            'force_coupling': self.config['coupling'].get('force_coupling', True),
            'heat_coupling': self.config['coupling'].get('heat_coupling', False),
            'mass_coupling': self.config['coupling'].get('mass_coupling', False)
        }
        
        return params
    
    def compute_coupling_forces(self, particles: Dict,
                              fluid_fields: Dict) -> Dict:
        """Compute coupling forces between particles and fluid.
        
        Args:
            particles: Dictionary containing particle data
            fluid_fields: Dictionary containing fluid field data
            
        Returns:
            Dictionary containing coupling forces
        """
        forces = {}
        
        # Get particle data
        positions = particles['positions']
        velocities = particles['velocities']
        radii = particles['radii']
        
        # Get fluid data
        fluid_velocity = fluid_fields['velocity']
        fluid_pressure = fluid_fields['pressure']
        fluid_density = fluid_fields['density']
        fluid_viscosity = fluid_fields['viscosity']
        
        # Interpolate fluid properties to particle positions
        fluid_props = self._interpolate_fluid_properties(
            positions, fluid_velocity, fluid_pressure,
            fluid_density, fluid_viscosity)
        
        # Compute drag forces
        forces['drag'] = self._compute_drag_forces(
            positions, velocities, fluid_props, radii)
        
        # Compute pressure forces (pass full 3D field)
        forces['pressure'] = self._compute_pressure_forces(
            positions, fluid_props, radii, fluid_pressure_field=fluid_pressure)
        
        # Compute lift forces (pass full 3D velocity field)
        forces['lift'] = self._compute_lift_forces(
            positions, velocities, fluid_props, radii, fluid_velocity_field=fluid_velocity)
        
        # Compute virtual mass forces
        forces['virtual_mass'] = self._compute_virtual_mass_forces(
            positions, velocities, fluid_props, radii)
        
        return forces
    
    def _interpolate_fluid_properties(self, positions: np.ndarray,
                                    fluid_velocity: np.ndarray,
                                    fluid_pressure: np.ndarray,
                                    fluid_density: np.ndarray,
                                    fluid_viscosity: np.ndarray) -> Dict:
        """Interpolate fluid properties to particle positions.
        
        Args:
            positions: Particle positions
            fluid_velocity: Fluid velocity field
            fluid_pressure: Fluid pressure field
            fluid_density: Fluid density field
            fluid_viscosity: Fluid viscosity field
            
        Returns:
            Dictionary of interpolated fluid properties
        """
        # Create interpolation functions
        x = np.arange(fluid_velocity.shape[0])
        y = np.arange(fluid_velocity.shape[1])
        z = np.arange(fluid_velocity.shape[2])
        
        interp_velocity = interpolate.RegularGridInterpolator(
            (x, y, z), fluid_velocity,
            method='linear', bounds_error=False, fill_value=0.0)
        
        interp_pressure = interpolate.RegularGridInterpolator(
            (x, y, z), fluid_pressure,
            method='linear', bounds_error=False, fill_value=0.0)
        
        interp_density = interpolate.RegularGridInterpolator(
            (x, y, z), fluid_density,
            method='linear', bounds_error=False, fill_value=0.0)
        
        interp_viscosity = interpolate.RegularGridInterpolator(
            (x, y, z), fluid_viscosity,
            method='linear', bounds_error=False, fill_value=0.0)
        
        # Interpolate properties
        properties = {
            'velocity': interp_velocity(positions),
            'pressure': interp_pressure(positions),
            'density': interp_density(positions),
            'viscosity': interp_viscosity(positions)
        }
        
        return properties
    
    def _compute_drag_forces(self, positions: np.ndarray,
                           velocities: np.ndarray,
                           fluid_props: Dict,
                           radii: np.ndarray) -> np.ndarray:
        """Compute drag forces on particles.
        
        Args:
            positions: Particle positions
            velocities: Particle velocities
            fluid_props: Interpolated fluid properties
            radii: Particle radii
            
        Returns:
            Array of drag forces
        """
        # Get relative velocities
        rel_velocities = velocities - fluid_props['velocity']
        rel_vel_magnitudes = np.linalg.norm(rel_velocities, axis=1)
        
        # Compute Reynolds numbers
        re = 2 * radii * rel_vel_magnitudes * fluid_props['density'] / fluid_props['viscosity']
        
        # Compute drag coefficients
        cd = self._compute_drag_coefficients(re)
        
        # Compute drag forces
        drag_forces = -0.5 * cd[:, np.newaxis] * fluid_props['density'][:, np.newaxis] * \
                     np.pi * radii[:, np.newaxis]**2 * \
                     rel_vel_magnitudes[:, np.newaxis] * rel_velocities
        
        return drag_forces
    
    def _compute_drag_coefficients(self, re: np.ndarray) -> np.ndarray:
        """Compute drag coefficients based on Reynolds number.
        
        Args:
            re: Reynolds numbers
            
        Returns:
            Array of drag coefficients
        """
        # Schiller-Naumann correlation
        cd = np.zeros_like(re)
        
        # Stokes regime
        mask = re < 0.1
        cd[mask] = 24.0 / re[mask]
        
        # Transition regime
        mask = (re >= 0.1) & (re < 1000)
        cd[mask] = 24.0 / re[mask] * (1 + 0.15 * re[mask]**0.687)
        
        # Newton regime
        mask = re >= 1000
        cd[mask] = 0.44
        
        return cd
    
    def _compute_pressure_forces(self, positions: np.ndarray,
                               fluid_props: Dict,
                               radii: np.ndarray, fluid_pressure_field=None) -> np.ndarray:
        """Compute pressure forces on particles.
        
        Args:
            positions: Particle positions
            fluid_props: Interpolated fluid properties
            radii: Particle radii
            fluid_pressure_field: Full 3D pressure field (optional, for gradient calculation)
        
        Returns:
            Array of pressure forces
        """
        # Compute pressure gradients
        if fluid_pressure_field is None:
            raise ValueError("Full 3D pressure field must be provided for gradient calculation.")
        pressure_gradients = self._compute_pressure_gradients(
            positions, fluid_pressure_field)
        
        # Compute pressure forces
        pressure_forces = -np.pi * radii[:, np.newaxis]**3 * pressure_gradients
        
        return pressure_forces
    
    def _compute_pressure_gradients(self, positions: np.ndarray,
                                  pressure: np.ndarray) -> np.ndarray:
        """Compute pressure gradients at particle positions.
        
        Args:
            positions: Particle positions
            pressure: Full 3D pressure field
        
        Returns:
            Array of pressure gradients
        """
        # Create interpolation function for pressure
        x = np.arange(pressure.shape[0])
        y = np.arange(pressure.shape[1])
        z = np.arange(pressure.shape[2])
        interp_pressure = interpolate.RegularGridInterpolator(
            (x, y, z), pressure,
            method='linear', bounds_error=False, fill_value=0.0)
        
        dx = 1.0
        dy = 1.0
        dz = 1.0
        gradients = np.zeros_like(positions)
        p_orig = interp_pressure(positions)
        # For each direction, shift positions and interpolate
        for i, delta in enumerate([(dx,0,0), (0,dy,0), (0,0,dz)]):
            shifted = positions + np.array(delta)
            p_shifted = interp_pressure(shifted)
            gradients[:, i] = (p_shifted - p_orig) / [dx, dy, dz][i]
        return gradients
    
    def _compute_lift_forces(self, positions: np.ndarray,
                           velocities: np.ndarray,
                           fluid_props: Dict,
                           radii: np.ndarray,
                           fluid_velocity_field=None) -> np.ndarray:
        """Compute lift forces on particles.
        
        Args:
            positions: Particle positions
            velocities: Particle velocities
            fluid_props: Interpolated fluid properties
            radii: Particle radii
            fluid_velocity_field: Full 3D velocity field
        
        Returns:
            Array of lift forces
        """
        if fluid_velocity_field is None:
            raise ValueError("Full 3D velocity field must be provided for vorticity calculation.")
        vorticity_field = self._compute_vorticity(fluid_velocity_field)
        # Interpolate vorticity at particle positions
        x = np.arange(vorticity_field.shape[0])
        y = np.arange(vorticity_field.shape[1])
        z = np.arange(vorticity_field.shape[2])
        interp_vorticity = interpolate.RegularGridInterpolator(
            (x, y, z), vorticity_field,
            method='linear', bounds_error=False, fill_value=0.0)
        vorticity_at_particles = interp_vorticity(positions)
        
        # Get relative velocities
        rel_velocities = velocities - fluid_props['velocity']
        rel_vel_magnitudes = np.linalg.norm(rel_velocities, axis=1)
        
        # Compute lift forces (Saffman lift)
        lift_forces = 0.5 * fluid_props['density'][:, np.newaxis] * rel_vel_magnitudes[:, np.newaxis] * np.cross(rel_velocities, vorticity_at_particles)
        return lift_forces
    
    def _compute_vorticity(self, velocity: np.ndarray) -> np.ndarray:
        """Compute vorticity field.
        
        Args:
            velocity: Velocity field
            
        Returns:
            Vorticity field
        """
        # Compute velocity gradients
        du_dx = np.gradient(velocity[..., 0], axis=0)
        du_dy = np.gradient(velocity[..., 0], axis=1)
        du_dz = np.gradient(velocity[..., 0], axis=2)
        
        dv_dx = np.gradient(velocity[..., 1], axis=0)
        dv_dy = np.gradient(velocity[..., 1], axis=1)
        dv_dz = np.gradient(velocity[..., 1], axis=2)
        
        dw_dx = np.gradient(velocity[..., 2], axis=0)
        dw_dy = np.gradient(velocity[..., 2], axis=1)
        dw_dz = np.gradient(velocity[..., 2], axis=2)
        
        # Compute vorticity components
        vorticity = np.zeros_like(velocity)
        vorticity[..., 0] = dw_dy - dv_dz
        vorticity[..., 1] = du_dz - dw_dx
        vorticity[..., 2] = dv_dx - du_dy
        
        return vorticity
    
    def _compute_virtual_mass_forces(self, positions: np.ndarray,
                                   velocities: np.ndarray,
                                   fluid_props: Dict,
                                   radii: np.ndarray) -> np.ndarray:
        """Compute virtual mass forces on particles.
        
        Args:
            positions: Particle positions
            velocities: Particle velocities
            fluid_props: Interpolated fluid properties
            radii: Particle radii
            
        Returns:
            Array of virtual mass forces
        """
        # Compute accelerations
        fluid_acc = self._compute_fluid_acceleration(fluid_props['velocity'])
        particle_acc = self._compute_particle_acceleration(velocities)
        # Broadcast density for vector operations
        density = fluid_props['density'][:, np.newaxis]
        # Compute virtual mass forces
        virtual_mass_forces = 0.5 * density * \
                             (fluid_acc - particle_acc)
        return virtual_mass_forces
    
    def _compute_fluid_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """Compute fluid acceleration field.
        
        Args:
            velocity: Velocity field
            
        Returns:
            Acceleration field
        """
        # Compute velocity gradients
        du_dt = np.gradient(velocity[..., 0], axis=0)
        dv_dt = np.gradient(velocity[..., 1], axis=0)
        dw_dt = np.gradient(velocity[..., 2], axis=0)
        
        # Compute acceleration
        acceleration = np.zeros_like(velocity)
        acceleration[..., 0] = du_dt
        acceleration[..., 1] = dv_dt
        acceleration[..., 2] = dw_dt
        
        return acceleration
    
    def _compute_particle_acceleration(self, velocities: np.ndarray) -> np.ndarray:
        """Compute particle acceleration.
        
        Args:
            velocities: Particle velocities
            
        Returns:
            Particle acceleration
        """
        # Compute velocity differences
        velocity_diff = np.diff(velocities, axis=0)
        
        # Pad with zeros to match input size
        acceleration = np.pad(velocity_diff, ((0, 1), (0, 0)), 'constant')
        
        return acceleration
    
    def update_particle_bonds(self, particles: Dict,
                            forces: Dict) -> Dict:
        """Update particle bonds based on coupling forces.
        
        Args:
            particles: Dictionary containing particle data
            forces: Dictionary containing coupling forces
            
        Returns:
            Updated particle data
        """
        updated_particles = particles.copy()
        
        # Get total forces
        total_forces = np.zeros_like(forces['drag'])
        for force_type, force in forces.items():
            total_forces += force
        
        # Update bond strengths
        if 'bond_strength' in particles:
            bond_strength = particles['bond_strength']
            force_magnitudes = np.linalg.norm(total_forces, axis=1)
            
            # Compute bond degradation
            degradation = self._compute_bond_degradation(
                force_magnitudes, bond_strength)
            
            # Update bond strengths
            updated_particles['bond_strength'] = bond_strength * (1 - degradation)
        
        return updated_particles
    
    def _compute_bond_degradation(self, force_magnitudes: np.ndarray,
                                bond_strength: np.ndarray) -> np.ndarray:
        """Compute bond degradation based on forces.
        
        Args:
            force_magnitudes: Magnitude of forces on particles
            bond_strength: Current bond strengths
            
        Returns:
            Bond degradation factors
        """
        # Get critical force
        critical_force = self.config['coupling'].get('critical_force', 1.0)
        
        # Compute degradation
        degradation = np.zeros_like(force_magnitudes)
        mask = force_magnitudes > critical_force
        degradation[mask] = (force_magnitudes[mask] - critical_force) / \
                          (force_magnitudes[mask] + critical_force)
        
        return degradation
    
    def save_coupling_results(self, results: Dict) -> None:
        """Save coupling results to file.
        
        Args:
            results: Dictionary containing coupling results
        """
        # Save force data
        np.savez(
            self.output_dir / 'force_data.npz',
            drag=results['drag'],
            pressure=results['pressure'],
            lift=results['lift'],
            virtual_mass=results['virtual_mass']
        )
        
        # Save statistics
        with open(self.output_dir / 'coupling_stats.json', 'w') as f:
            json.dump(results['statistics'], f, indent=4)
        
        self.logger.info("Saved coupling results") 