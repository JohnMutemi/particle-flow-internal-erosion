"""
Initialization stage for tunnel water inrush simulation.
"""

import numpy as np


class InitializationStage:
    def __init__(self, config, case_params):
        self.config = config
        self.case_params = case_params

    def setup_initial_conditions(self, domain_size, particle_radius):
        # Create particle positions
        positions = self._create_particle_positions(
            domain_size, particle_radius)
        velocities = np.zeros_like(positions)
        bond_health = np.ones(len(positions))
        self._apply_tunnel_geometry(positions, bond_health)
        pressure_field = self._create_pressure_field(domain_size)
        velocity_field = self._create_velocity_field(domain_size)
        return {
            'positions': positions,
            'velocities': velocities,
            'bond_health': bond_health,
            'pressure_field': pressure_field,
            'velocity_field': velocity_field
        }

    def _create_particle_positions(self, domain_size, particle_radius):
        volume = domain_size[0] * domain_size[1] * domain_size[2]
        particle_volume = 4/3 * np.pi * particle_radius**3
        num_particles = int(volume / particle_volume * 0.6)
        positions = np.random.rand(num_particles, 3)
        positions[:, 0] *= domain_size[0]
        positions[:, 1] *= domain_size[1]
        positions[:, 2] *= domain_size[2]
        return positions

    def _apply_tunnel_geometry(self, positions, bond_health):
        tunnel_center = self.case_params['tunnel']['center']
        tunnel_radius = self.case_params['tunnel']['diameter'] / 2
        distances = np.linalg.norm(positions - tunnel_center, axis=1)
        mask = distances > tunnel_radius
        positions[:] = positions[mask]
        bond_health[:] = bond_health[mask]
        surface_distance = 2 * tunnel_radius
        surface_mask = distances < surface_distance
        bond_health[surface_mask] *= 0.8

    def _create_pressure_field(self, domain_size):
        # Placeholder for pressure field creation
        return np.zeros(domain_size)

    def _create_velocity_field(self, domain_size):
        # Placeholder for velocity field creation
        return np.zeros(domain_size)
