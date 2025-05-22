"""
Computational Fluid Dynamics (CFD) solver for fluid flow simulation.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class CFDSolver:
    def __init__(self, config: Dict):
        """Initialize the CFD solver with configuration parameters."""
        # Validate required configuration parameters
        required_params = {
            'cfd': ['fluid_density', 'fluid_viscosity'],
            'simulation': ['domain_size', 'time_step']
        }
        
        for section, params in required_params.items():
            if section not in config:
                raise ValueError(f"Missing configuration section: {section}")
            for param in params:
                if param not in config[section]:
                    raise ValueError(f"Missing parameter {param} in {section} section")
        
        # Initialize CFD parameters
        self.config = config
        self.fluid_density = config['cfd']['fluid_density']
        self.fluid_viscosity = config['cfd']['fluid_viscosity']
        self.pressure_gradient = np.array(config['cfd'].get('pressure_gradient', [0.0, 0.0, 0.0]))
        
        # Set default boundary conditions if not provided
        default_boundary_conditions = {
            'x': {'left': 'no-slip', 'right': 'no-slip'},
            'y': {'bottom': 'no-slip', 'top': 'no-slip'},
            'z': {'front': 'no-slip', 'back': 'no-slip'}
        }
        self.boundary_conditions = config['cfd'].get('boundary_conditions', default_boundary_conditions)
        
        # Domain parameters
        self.domain_size = np.array(config['simulation']['domain_size'])
        self.time_step = config['simulation']['time_step']
        
        # Grid parameters
        self.grid_resolution = np.array(config['simulation'].get('grid_resolution', [50, 50, 50]))
        self.dx = self.domain_size / self.grid_resolution
        
        # Initialize fluid fields
        self.velocity_field = None
        self.pressure_field = None
        self.vorticity_field = None
        
        # Initialize geometry
        self.tunnel_geometry = None
        
        logger.info("CFD solver initialized")

    def initialize_fields(self):
        """Initialize fluid velocity and pressure fields."""
        logger.info("Initializing fluid fields")
        # Initialize velocity field with zeros
        self.velocity_field = np.zeros((*self.grid_resolution, 3))
        
        # Initialize pressure field with hydrostatic pressure
        x = np.linspace(0, self.domain_size[0], self.grid_resolution[0])
        y = np.linspace(0, self.domain_size[1], self.grid_resolution[1])
        z = np.linspace(0, self.domain_size[2], self.grid_resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # Use 'ij' indexing to match array dimensions
        
        # Hydrostatic pressure: p = ρgh
        self.pressure_field = self.fluid_density * 9.81 * (self.domain_size[1] - Y)
        
        # Initialize vorticity field
        self.vorticity_field = np.zeros((*self.grid_resolution, 3))
        
        logger.info("Fluid fields initialized")

    def set_tunnel_geometry(self, tunnel_params: Dict):
        """Set tunnel geometry parameters."""
        self.tunnel_geometry = {
            'length': tunnel_params.get('length', self.domain_size[0]),
            'diameter': tunnel_params.get('diameter', min(self.domain_size[1:])),
            'center': np.array(tunnel_params.get('center', self.domain_size / 2)),
            'roughness': tunnel_params.get('roughness', 0.001)  # Default roughness in meters
        }
        logger.info("Tunnel geometry set")

    def compute_fluid_forces(self, particle_positions: List[np.ndarray]) -> List[np.ndarray]:
        """Compute fluid forces on particles using interpolated field values."""
        forces = []
        for position in particle_positions:
            # Interpolate velocity and pressure at particle position
            velocity = self._interpolate_field(position, self.velocity_field)
            pressure_gradient = self._compute_pressure_gradient(position)
            vorticity = self._interpolate_field(position, self.vorticity_field)
            
            # Compute forces
            drag_force = self._compute_drag_force(velocity, particle_positions)
            pressure_force = self._compute_pressure_force(pressure_gradient)
            lift_force = self._compute_lift_force(velocity, vorticity)
            
            # Total force
            total_force = drag_force + pressure_force + lift_force
            forces.append(total_force)
        
        return forces

    def update_fluid_state(self):
        """Update fluid velocity and pressure fields using Navier-Stokes equations."""
        # Compute velocity gradients
        velocity_gradients = self._compute_velocity_gradients()
        
        # Update velocity field using Navier-Stokes equations
        self.velocity_field = self._solve_momentum_equation(velocity_gradients)
        
        # Update pressure field using Poisson equation
        self.pressure_field = self._solve_pressure_poisson()
        
        # Update vorticity field
        self.vorticity_field = self._compute_vorticity()

    def _interpolate_field(self, position: np.ndarray, field: np.ndarray) -> np.ndarray:
        """Interpolate field values at particle position using trilinear interpolation."""
        # Ensure position is within domain bounds
        position = np.clip(position, np.zeros(3), self.domain_size)
        
        # Convert position to grid indices
        indices = position / self.dx
        
        # Get surrounding grid points with boundary checking
        i0 = np.clip(np.floor(indices[0]).astype(int), 0, self.grid_resolution[0]-2)
        j0 = np.clip(np.floor(indices[1]).astype(int), 0, self.grid_resolution[1]-2)
        k0 = np.clip(np.floor(indices[2]).astype(int), 0, self.grid_resolution[2]-2)
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # Compute interpolation weights
        wx = (indices[0] - i0) / (i1 - i0) if i1 > i0 else 0
        wy = (indices[1] - j0) / (j1 - j0) if j1 > j0 else 0
        wz = (indices[2] - k0) / (k1 - k0) if k1 > k0 else 0
        
        # Trilinear interpolation
        c000 = field[i0, j0, k0]
        c001 = field[i0, j0, k1]
        c010 = field[i0, j1, k0]
        c011 = field[i0, j1, k1]
        c100 = field[i1, j0, k0]
        c101 = field[i1, j0, k1]
        c110 = field[i1, j1, k0]
        c111 = field[i1, j1, k1]
        
        c00 = c000 * (1 - wz) + c001 * wz
        c01 = c010 * (1 - wz) + c011 * wz
        c10 = c100 * (1 - wz) + c101 * wz
        c11 = c110 * (1 - wz) + c111 * wz
        
        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy
        
        return c0 * (1 - wx) + c1 * wx

    def _compute_velocity_gradients(self) -> np.ndarray:
        """Compute velocity gradients using central differences with proper boundary handling."""
        gradients = np.zeros((*self.grid_resolution, 3, 3))
        
        # Use forward/backward differences at boundaries and central differences in interior
        for i in range(self.grid_resolution[0]):
            for j in range(self.grid_resolution[1]):
                for k in range(self.grid_resolution[2]):
                    # x-direction
                    if i == 0:
                        # Forward difference at left boundary
                        du_dx = (self.velocity_field[i+1,j,k] - self.velocity_field[i,j,k]) / self.dx[0]
                    elif i == self.grid_resolution[0]-1:
                        # Backward difference at right boundary
                        du_dx = (self.velocity_field[i,j,k] - self.velocity_field[i-1,j,k]) / self.dx[0]
                    else:
                        # Central difference in interior
                        du_dx = (self.velocity_field[i+1,j,k] - self.velocity_field[i-1,j,k]) / (2 * self.dx[0])
                    
                    # y-direction
                    if j == 0:
                        # Forward difference at bottom boundary
                        du_dy = (self.velocity_field[i,j+1,k] - self.velocity_field[i,j,k]) / self.dx[1]
                    elif j == self.grid_resolution[1]-1:
                        # Backward difference at top boundary
                        du_dy = (self.velocity_field[i,j,k] - self.velocity_field[i,j-1,k]) / self.dx[1]
                    else:
                        # Central difference in interior
                        du_dy = (self.velocity_field[i,j+1,k] - self.velocity_field[i,j-1,k]) / (2 * self.dx[1])
                    
                    # z-direction
                    if k == 0:
                        # Forward difference at front boundary
                        du_dz = (self.velocity_field[i,j,k+1] - self.velocity_field[i,j,k]) / self.dx[2]
                    elif k == self.grid_resolution[2]-1:
                        # Backward difference at back boundary
                        du_dz = (self.velocity_field[i,j,k] - self.velocity_field[i,j,k-1]) / self.dx[2]
                    else:
                        # Central difference in interior
                        du_dz = (self.velocity_field[i,j,k+1] - self.velocity_field[i,j,k-1]) / (2 * self.dx[2])
                    
                    gradients[i,j,k] = np.array([du_dx, du_dy, du_dz])
        
        return gradients

    def _solve_momentum_equation(self, velocity_gradients: np.ndarray) -> np.ndarray:
        """Solve momentum equation using explicit time integration."""
        # Compute convective term
        convective = np.einsum('ijklm,ijkm->ijkl', velocity_gradients, self.velocity_field)
        
        # Compute viscous term
        laplacian = self._compute_laplacian(self.velocity_field)
        viscous = self.fluid_viscosity * laplacian
        
        # Compute pressure gradient
        pressure_gradient = self._compute_pressure_gradient_field()
        
        # Update velocity
        new_velocity = self.velocity_field + self.time_step * (
            -convective - pressure_gradient/self.fluid_density + viscous
        )
        
        return new_velocity

    def _solve_pressure_poisson(self) -> np.ndarray:
        """Solve pressure Poisson equation using SOR method."""
        # Initialize pressure field
        pressure = self.pressure_field.copy()
        
        # SOR parameters
        omega = 1.5  # Relaxation parameter
        max_iter = 1000
        tolerance = 1e-6
        
        for _ in range(max_iter):
            pressure_old = pressure.copy()
            
            # Update pressure using SOR
            for i in range(1, self.grid_resolution[0]-1):
                for j in range(1, self.grid_resolution[1]-1):
                    for k in range(1, self.grid_resolution[2]-1):
                        # Compute divergence of velocity
                        div_u = (
                            (self.velocity_field[i+1,j,k,0] - self.velocity_field[i-1,j,k,0]) / (2 * self.dx[0]) +
                            (self.velocity_field[i,j+1,k,1] - self.velocity_field[i,j-1,k,1]) / (2 * self.dx[1]) +
                            (self.velocity_field[i,j,k+1,2] - self.velocity_field[i,j,k-1,2]) / (2 * self.dx[2])
                        )
                        
                        # Update pressure
                        pressure[i,j,k] = (1 - omega) * pressure[i,j,k] + omega * (
                            (pressure[i+1,j,k] + pressure[i-1,j,k]) / self.dx[0]**2 +
                            (pressure[i,j+1,k] + pressure[i,j-1,k]) / self.dx[1]**2 +
                            (pressure[i,j,k+1] + pressure[i,j,k-1]) / self.dx[2]**2 -
                            self.fluid_density * div_u / self.time_step
                        ) / (
                            2 / self.dx[0]**2 + 2 / self.dx[1]**2 + 2 / self.dx[2]**2
                        )
            
            # Check convergence
            if np.max(np.abs(pressure - pressure_old)) < tolerance:
                break
        
        return pressure

    def _compute_vorticity(self) -> np.ndarray:
        """Compute vorticity field from velocity field."""
        vorticity = np.zeros((*self.grid_resolution, 3))
        
        for i in range(1, self.grid_resolution[0]-1):
            for j in range(1, self.grid_resolution[1]-1):
                for k in range(1, self.grid_resolution[2]-1):
                    # Compute velocity gradients
                    du_dy = (self.velocity_field[i,j+1,k,0] - self.velocity_field[i,j-1,k,0]) / (2 * self.dx[1])
                    du_dz = (self.velocity_field[i,j,k+1,0] - self.velocity_field[i,j,k-1,0]) / (2 * self.dx[2])
                    dv_dx = (self.velocity_field[i+1,j,k,1] - self.velocity_field[i-1,j,k,1]) / (2 * self.dx[0])
                    dv_dz = (self.velocity_field[i,j,k+1,1] - self.velocity_field[i,j,k-1,1]) / (2 * self.dx[2])
                    dw_dx = (self.velocity_field[i+1,j,k,2] - self.velocity_field[i-1,j,k,2]) / (2 * self.dx[0])
                    dw_dy = (self.velocity_field[i,j+1,k,2] - self.velocity_field[i,j-1,k,2]) / (2 * self.dx[1])
                    
                    # Compute vorticity components
                    vorticity[i,j,k] = np.array([
                        dw_dy - dv_dz,
                        du_dz - dw_dx,
                        dv_dx - du_dy
                    ])
        
        return vorticity

    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute Laplacian of a field using central differences."""
        laplacian = np.zeros_like(field)
        
        for i in range(1, self.grid_resolution[0]-1):
            for j in range(1, self.grid_resolution[1]-1):
                for k in range(1, self.grid_resolution[2]-1):
                    laplacian[i,j,k] = (
                        (field[i+1,j,k] - 2*field[i,j,k] + field[i-1,j,k]) / self.dx[0]**2 +
                        (field[i,j+1,k] - 2*field[i,j,k] + field[i,j-1,k]) / self.dx[1]**2 +
                        (field[i,j,k+1] - 2*field[i,j,k] + field[i,j,k-1]) / self.dx[2]**2
                    )
        
        return laplacian

    def _compute_pressure_gradient_field(self) -> np.ndarray:
        """Compute pressure gradient field using central differences."""
        gradient = np.zeros((*self.grid_resolution, 3))
        
        # Use forward/backward differences at boundaries and central differences in interior
        for i in range(self.grid_resolution[0]):
            for j in range(self.grid_resolution[1]):
                for k in range(self.grid_resolution[2]):
                    # x-direction
                    if i == 0:
                        # Forward difference at left boundary
                        dx = (self.pressure_field[i+1,j,k] - self.pressure_field[i,j,k]) / self.dx[0]
                    elif i == self.grid_resolution[0]-1:
                        # Backward difference at right boundary
                        dx = (self.pressure_field[i,j,k] - self.pressure_field[i-1,j,k]) / self.dx[0]
                    else:
                        # Central difference in interior
                        dx = (self.pressure_field[i+1,j,k] - self.pressure_field[i-1,j,k]) / (2 * self.dx[0])
                    
                    # y-direction
                    if j == 0:
                        # Forward difference at bottom boundary
                        dy = (self.pressure_field[i,j+1,k] - self.pressure_field[i,j,k]) / self.dx[1]
                    elif j == self.grid_resolution[1]-1:
                        # Backward difference at top boundary
                        dy = (self.pressure_field[i,j,k] - self.pressure_field[i,j-1,k]) / self.dx[1]
                    else:
                        # Central difference in interior
                        dy = (self.pressure_field[i,j+1,k] - self.pressure_field[i,j-1,k]) / (2 * self.dx[1])
                    
                    # z-direction
                    if k == 0:
                        # Forward difference at front boundary
                        dz = (self.pressure_field[i,j,k+1] - self.pressure_field[i,j,k]) / self.dx[2]
                    elif k == self.grid_resolution[2]-1:
                        # Backward difference at back boundary
                        dz = (self.pressure_field[i,j,k] - self.pressure_field[i,j,k-1]) / self.dx[2]
                    else:
                        # Central difference in interior
                        dz = (self.pressure_field[i,j,k+1] - self.pressure_field[i,j,k-1]) / (2 * self.dx[2])
                    
                    gradient[i,j,k] = np.array([dx, dy, dz])
        
        return gradient

    def get_fluid_data(self) -> Dict:
        """Get current fluid field data."""
        return {
            'velocity_field': self.velocity_field,
            'pressure_field': self.pressure_field,
            'vorticity_field': self.vorticity_field,
            'domain_size': self.domain_size,
            'grid_resolution': self.grid_resolution
        }

    def set_boundary_conditions(self, boundary_conditions: Dict):
        """Set boundary conditions for the fluid simulation."""
        self.boundary_conditions = boundary_conditions
        logger.info("Boundary conditions updated") 