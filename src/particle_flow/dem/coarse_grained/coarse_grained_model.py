"""
Coarse-grained model for DEM simulations.
This module implements a coarse-grained model for particle interactions.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class CoarseGrainedModel:
    def __init__(self, config: Dict):
        """Initialize the coarse-grained model."""
        self.config = config
        self.particle_radius = config['dem']['particle_radius']
        self.bond_strength = config['dem'].get('bond_strength', 1e6)
        self.bond_health = 1.0  # Initial bond health
        
        logger.info("Coarse-grained model initialized")

    def compute_bond_forces(self, particle_positions: np.ndarray,
                          particle_velocities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forces between bonded particles."""
        num_particles = len(particle_positions)
        forces = np.zeros((num_particles, 3))
        torques = np.zeros((num_particles, 3))
        
        # Compute forces between neighboring particles
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                # Compute distance and direction
                r_ij = particle_positions[j] - particle_positions[i]
                distance = np.linalg.norm(r_ij)
                direction = r_ij / distance if distance > 0 else np.zeros(3)
                
                # Check if particles are bonded
                if distance < 2.1 * self.particle_radius:  # Slight overlap allowed
                    # Compute spring force
                    spring_force = self.bond_strength * self.bond_health * \
                                 (2 * self.particle_radius - distance) * direction
                    
                    # Add forces to particles
                    forces[i] += spring_force
                    forces[j] -= spring_force
                    
                    # Compute torques
                    torques[i] += np.cross(r_ij/2, spring_force)
                    torques[j] += np.cross(-r_ij/2, spring_force)
        
        return forces, torques

    def update_bond_health(self, erosion_rate: float, time_step: float):
        """Update bond health based on erosion rate."""
        # Decrease bond health based on erosion rate
        self.bond_health -= erosion_rate * time_step
        self.bond_health = max(0.0, min(1.0, self.bond_health))
        
        return self.bond_health

    def get_bond_health(self) -> float:
        """Get current bond health."""
        return self.bond_health

    def initialize_coarse_particles(self, num_particles: int):
        """Initialize coarse-grained particles in the domain."""
        logger.info(f"Initializing {num_particles} coarse-grained particles")
        self.particles = []
        for _ in range(num_particles):
            # Random position within the domain
            position = np.random.rand(3) * self.config['simulation']['domain_size']
            velocity = np.zeros(3)
            radius = self.config['dem']['particle_radius']
            mass = self.config['dem']['particle_density'] * (4/3) * np.pi * radius**3
            
            particle = {
                'position': position,
                'velocity': velocity,
                'radius': radius,
                'mass': mass,
                'bonds': []
            }
            self.particles.append(particle) 