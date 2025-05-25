"""
Parallel solver module for engineering scale CFD-DEM simulations.
This module implements parallel processing support and adaptive mesh refinement.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from mpi4py import MPI
import numba
from numba import prange
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class ParallelSolver:
    def __init__(self, config: Dict):
        """Initialize the parallel solver."""
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Initialize domain decomposition
        self._initialize_domain_decomposition()
        
        # Initialize adaptive mesh refinement
        self._initialize_amr()
        
        logger.info(f"Parallel solver initialized on rank {self.rank}")

    def _initialize_domain_decomposition(self):
        """Initialize domain decomposition for parallel processing."""
        # Get domain size and grid resolution
        domain_size = np.array(self.config['simulation']['domain_size'])
        grid_resolution = np.array(self.config['simulation']['grid_resolution'])
        
        # Compute optimal domain decomposition
        self.decomposition = self._compute_optimal_decomposition(
            domain_size, grid_resolution, self.size
        )
        
        # Set up local domain for this rank
        self.local_domain = self._setup_local_domain()
        
        logger.info(f"Domain decomposition initialized on rank {self.rank}")

    def _compute_optimal_decomposition(self, domain_size: np.ndarray,
                                     grid_resolution: np.ndarray,
                                     num_ranks: int) -> np.ndarray:
        """Compute optimal domain decomposition."""
        # Compute aspect ratios
        aspect_ratios = domain_size / np.min(domain_size)
        
        # Initialize decomposition
        decomposition = np.ones(3, dtype=int)
        
        # Distribute ranks based on aspect ratios
        remaining_ranks = num_ranks
        while remaining_ranks > 1:
            # Find dimension with largest aspect ratio
            dim = np.argmax(aspect_ratios)
            
            # Split this dimension
            decomposition[dim] *= 2
            aspect_ratios[dim] /= 2
            remaining_ranks //= 2
        
        return decomposition

    def _setup_local_domain(self) -> Dict:
        """Set up local domain for this rank."""
        # Get global domain size and grid resolution
        domain_size = np.array(self.config['simulation']['domain_size'])
        grid_resolution = np.array(self.config['simulation']['grid_resolution'])
        
        # Compute local domain size
        local_size = domain_size / self.decomposition
        
        # Compute local grid resolution
        local_resolution = grid_resolution / self.decomposition
        
        # Compute local domain bounds
        local_bounds = self._compute_local_bounds()
        
        return {
            'size': local_size,
            'resolution': local_resolution,
            'bounds': local_bounds
        }

    def _compute_local_bounds(self) -> Dict:
        """Compute local domain bounds for this rank."""
        # Get global domain size
        domain_size = np.array(self.config['simulation']['domain_size'])
        
        # Compute rank coordinates in decomposition
        rank_coords = np.unravel_index(self.rank, self.decomposition)
        
        # Compute local bounds
        bounds = {}
        for dim in range(3):
            local_size = domain_size[dim] / self.decomposition[dim]
            bounds[f'min_{dim}'] = rank_coords[dim] * local_size
            bounds[f'max_{dim}'] = (rank_coords[dim] + 1) * local_size
        
        return bounds

    def _initialize_amr(self):
        """Initialize adaptive mesh refinement."""
        self.amr_config = {
            'max_level': self.config['parallel'].get('max_amr_level', 3),
            'refinement_threshold': self.config['parallel'].get('refinement_threshold', 0.1),
            'coarsening_threshold': self.config['parallel'].get('coarsening_threshold', 0.01),
            'buffer_cells': self.config['parallel'].get('amr_buffer_cells', 2)
        }
        
        # Initialize AMR grid
        self.amr_grid = self._create_amr_grid()
        
        logger.info(f"AMR initialized on rank {self.rank}")

    def _create_amr_grid(self) -> Dict:
        """Create initial AMR grid."""
        grid = {
            'level': 0,
            'cells': self.local_domain['resolution'],
            'parent': None,
            'children': []
        }
        
        return grid

    @numba.jit(parallel=True)
    def _compute_fluid_forces_parallel(self, positions: np.ndarray,
                                     velocities: np.ndarray,
                                     pressure: np.ndarray) -> np.ndarray:
        """Compute fluid forces in parallel using Numba."""
        n_particles = len(positions)
        forces = np.zeros((n_particles, 3))
        
        for i in prange(n_particles):
            # Interpolate fluid properties at particle position
            velocity = self._interpolate_field(positions[i], velocities)
            pressure_grad = self._compute_pressure_gradient(positions[i], pressure)
            
            # Compute forces
            drag_force = self._compute_drag_force(velocity, positions[i])
            pressure_force = self._compute_pressure_force(pressure_grad)
            lift_force = self._compute_lift_force(velocity, positions[i])
            
            # Total force
            forces[i] = drag_force + pressure_force + lift_force
        
        return forces

    def _interpolate_field(self, position: np.ndarray, field: np.ndarray) -> np.ndarray:
        """Interpolate field values at particle position."""
        # Convert position to local grid coordinates
        local_pos = (position - self.local_domain['bounds']['min']) / \
                   self.local_domain['size'] * self.local_domain['resolution']
        
        # Get surrounding grid points
        i0, j0, k0 = np.floor(local_pos).astype(int)
        i1, j1, k1 = np.ceil(local_pos).astype(int)
        
        # Compute interpolation weights
        wx = (local_pos[0] - i0) / (i1 - i0) if i1 > i0 else 0
        wy = (local_pos[1] - j0) / (j1 - j0) if j1 > j0 else 0
        wz = (local_pos[2] - k0) / (k1 - k0) if k1 > k0 else 0
        
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

    def _compute_pressure_gradient(self, position: np.ndarray,
                                 pressure: np.ndarray) -> np.ndarray:
        """Compute pressure gradient at particle position."""
        # Convert position to local grid coordinates
        local_pos = (position - self.local_domain['bounds']['min']) / \
                   self.local_domain['size'] * self.local_domain['resolution']
        
        # Get surrounding grid points
        i0, j0, k0 = np.floor(local_pos).astype(int)
        i1, j1, k1 = np.ceil(local_pos).astype(int)
        
        # Compute gradients using central differences
        dx = self.local_domain['size'][0] / self.local_domain['resolution'][0]
        dy = self.local_domain['size'][1] / self.local_domain['resolution'][1]
        dz = self.local_domain['size'][2] / self.local_domain['resolution'][2]
        
        dp_dx = (pressure[i1, j0, k0] - pressure[i0, j0, k0]) / dx
        dp_dy = (pressure[i0, j1, k0] - pressure[i0, j0, k0]) / dy
        dp_dz = (pressure[i0, j0, k1] - pressure[i0, j0, k0]) / dz
        
        return np.array([dp_dx, dp_dy, dp_dz])

    def _compute_drag_force(self, velocity: np.ndarray,
                           position: np.ndarray) -> np.ndarray:
        """Compute drag force on particle."""
        # Get particle properties
        particle_radius = self.config['dem']['particle_radius']
        fluid_density = self.config['cfd']['fluid_density']
        fluid_viscosity = self.config['cfd']['fluid_viscosity']
        
        # Compute Reynolds number
        velocity_magnitude = np.linalg.norm(velocity)
        reynolds = 2 * particle_radius * velocity_magnitude * fluid_density / fluid_viscosity
        
        # Compute drag coefficient
        if reynolds < 1:
            # Stokes flow
            drag_coef = 24 / reynolds
        elif reynolds < 1000:
            # Intermediate flow
            drag_coef = 24 / reynolds * (1 + 0.15 * reynolds**0.687)
        else:
            # Turbulent flow
            drag_coef = 0.44
        
        # Compute drag force
        drag_force = 0.5 * fluid_density * velocity_magnitude**2 * \
                    np.pi * particle_radius**2 * drag_coef
        
        return -drag_force * velocity / velocity_magnitude

    def _compute_pressure_force(self, pressure_grad: np.ndarray) -> np.ndarray:
        """Compute pressure force on particle."""
        particle_radius = self.config['dem']['particle_radius']
        return -np.pi * particle_radius**2 * pressure_grad

    def _compute_lift_force(self, velocity: np.ndarray,
                           position: np.ndarray) -> np.ndarray:
        """Compute lift force on particle."""
        # Get particle properties
        particle_radius = self.config['dem']['particle_radius']
        fluid_density = self.config['cfd']['fluid_density']
        
        # Compute velocity gradient
        velocity_grad = self._compute_velocity_gradient(position)
        
        # Compute vorticity
        vorticity = np.array([
            velocity_grad[2, 1] - velocity_grad[1, 2],
            velocity_grad[0, 2] - velocity_grad[2, 0],
            velocity_grad[1, 0] - velocity_grad[0, 1]
        ])
        
        # Compute lift force (Saffman lift)
        velocity_magnitude = np.linalg.norm(velocity)
        lift_coef = 1.61 * particle_radius**2 * np.sqrt(
            fluid_density * np.linalg.norm(vorticity) / fluid_density
        )
        
        return lift_coef * np.cross(vorticity, velocity)

    def _compute_velocity_gradient(self, position: np.ndarray) -> np.ndarray:
        """Compute velocity gradient at particle position."""
        # Convert position to local grid coordinates
        local_pos = (position - self.local_domain['bounds']['min']) / \
                   self.local_domain['size'] * self.local_domain['resolution']
        
        # Get surrounding grid points
        i0, j0, k0 = np.floor(local_pos).astype(int)
        i1, j1, k1 = np.ceil(local_pos).astype(int)
        
        # Compute gradients using central differences
        dx = self.local_domain['size'][0] / self.local_domain['resolution'][0]
        dy = self.local_domain['size'][1] / self.local_domain['resolution'][1]
        dz = self.local_domain['size'][2] / self.local_domain['resolution'][2]
        
        # Get velocity field
        velocity_field = self.get_velocity_field()
        
        # Compute gradients
        grad = np.zeros((3, 3))
        for i in range(3):
            grad[i, 0] = (velocity_field[i1, j0, k0, i] - velocity_field[i0, j0, k0, i]) / dx
            grad[i, 1] = (velocity_field[i0, j1, k0, i] - velocity_field[i0, j0, k0, i]) / dy
            grad[i, 2] = (velocity_field[i0, j0, k1, i] - velocity_field[i0, j0, k0, i]) / dz
        
        return grad

    def get_velocity_field(self) -> np.ndarray:
        """Get local velocity field."""
        # This should be implemented to get the actual velocity field
        # For now, return a placeholder
        return np.zeros((*self.local_domain['resolution'], 3))

    def update_amr(self, error_indicators: np.ndarray):
        """Update adaptive mesh refinement based on error indicators."""
        if self.amr_grid['level'] >= self.amr_config['max_level']:
            return
        
        # Check if refinement is needed
        if np.max(error_indicators) > self.amr_config['refinement_threshold']:
            self._refine_grid()
        elif np.max(error_indicators) < self.amr_config['coarsening_threshold']:
            self._coarsen_grid()

    def _refine_grid(self):
        """Refine the AMR grid."""
        # Create new child grid
        child_grid = {
            'level': self.amr_grid['level'] + 1,
            'cells': self.amr_grid['cells'] * 2,
            'parent': self.amr_grid,
            'children': []
        }
        
        # Add child to parent
        self.amr_grid['children'].append(child_grid)
        
        # Update current grid
        self.amr_grid = child_grid
        
        logger.info(f"Grid refined to level {self.amr_grid['level']} on rank {self.rank}")

    def _coarsen_grid(self):
        """Coarsen the AMR grid."""
        if self.amr_grid['parent'] is None:
            return
        
        # Update current grid to parent
        self.amr_grid = self.amr_grid['parent']
        
        logger.info(f"Grid coarsened to level {self.amr_grid['level']} on rank {self.rank}")

    def exchange_boundary_data(self, field: np.ndarray):
        """Exchange boundary data between ranks."""
        # Get local domain size
        local_size = self.local_domain['resolution']
        
        # Create buffers for boundary data
        send_buffers = {}
        recv_buffers = {}
        
        # Set up communication for each dimension
        for dim in range(3):
            # Get neighbor ranks
            neighbor_ranks = self._get_neighbor_ranks(dim)
            
            # Create buffers for this dimension
            for direction in ['min', 'max']:
                if neighbor_ranks[direction] is not None:
                    # Create send buffer
                    if direction == 'min':
                        send_buffers[f'{dim}_min'] = field[0, :, :].copy()
                    else:
                        send_buffers[f'{dim}_max'] = field[-1, :, :].copy()
                    
                    # Create receive buffer
                    recv_buffers[f'{dim}_{direction}'] = np.zeros_like(
                        send_buffers[f'{dim}_{direction}']
                    )
        
        # Exchange data
        for dim in range(3):
            for direction in ['min', 'max']:
                if neighbor_ranks[direction] is not None:
                    # Send data
                    self.comm.Send(
                        send_buffers[f'{dim}_{direction}'],
                        dest=neighbor_ranks[direction]
                    )
                    
                    # Receive data
                    self.comm.Recv(
                        recv_buffers[f'{dim}_{direction}'],
                        source=neighbor_ranks[direction]
                    )
        
        # Update field with received data
        for dim in range(3):
            for direction in ['min', 'max']:
                if neighbor_ranks[direction] is not None:
                    if direction == 'min':
                        field[0, :, :] = recv_buffers[f'{dim}_min']
                    else:
                        field[-1, :, :] = recv_buffers[f'{dim}_max']

    def _get_neighbor_ranks(self, dim: int) -> Dict[str, Optional[int]]:
        """Get neighbor ranks for a given dimension."""
        # Get rank coordinates in decomposition
        rank_coords = np.unravel_index(self.rank, self.decomposition)
        
        # Compute neighbor coordinates
        neighbors = {}
        for direction in ['min', 'max']:
            neighbor_coords = rank_coords.copy()
            if direction == 'min':
                neighbor_coords[dim] -= 1
            else:
                neighbor_coords[dim] += 1
            
            # Check if neighbor exists
            if 0 <= neighbor_coords[dim] < self.decomposition[dim]:
                neighbors[direction] = np.ravel_multi_index(
                    neighbor_coords, self.decomposition
                )
            else:
                neighbors[direction] = None
        
        return neighbors

    def plot_domain_decomposition(self):
        """Plot domain decomposition."""
        if self.rank != 0:
            return
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot domain boundaries
        domain_size = np.array(self.config['simulation']['domain_size'])
        ax.plot([0, domain_size[0]], [0, 0], [0, 0], 'k-')
        ax.plot([0, domain_size[0]], [0, 0], [0, domain_size[2]], 'k-')
        ax.plot([0, domain_size[0]], [0, domain_size[1]], [0, 0], 'k-')
        ax.plot([0, domain_size[0]], [0, domain_size[1]], [0, domain_size[2]], 'k-')
        
        # Plot subdomain boundaries
        for i in range(self.decomposition[0]):
            for j in range(self.decomposition[1]):
                for k in range(self.decomposition[2]):
                    rank = np.ravel_multi_index(
                        (i, j, k), self.decomposition
                    )
                    
                    # Compute subdomain bounds
                    x_min = i * domain_size[0] / self.decomposition[0]
                    x_max = (i + 1) * domain_size[0] / self.decomposition[0]
                    y_min = j * domain_size[1] / self.decomposition[1]
                    y_max = (j + 1) * domain_size[1] / self.decomposition[1]
                    z_min = k * domain_size[2] / self.decomposition[2]
                    z_max = (k + 1) * domain_size[2] / self.decomposition[2]
                    
                    # Plot subdomain
                    ax.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], 'b-')
                    ax.plot([x_min, x_max], [y_min, y_min], [z_max, z_max], 'b-')
                    ax.plot([x_min, x_max], [y_max, y_max], [z_min, z_min], 'b-')
                    ax.plot([x_min, x_max], [y_max, y_max], [z_max, z_max], 'b-')
                    
                    # Add rank label
                    ax.text(
                        (x_min + x_max) / 2,
                        (y_min + y_max) / 2,
                        (z_min + z_max) / 2,
                        str(rank)
                    )
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Save figure
        output_dir = Path(self.config['output']['directory']) / 'parallel'
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'domain_decomposition.png')
        plt.close()
        
        logger.info("Domain decomposition plot saved") 