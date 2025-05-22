"""
Coarse-grained model for large-scale CFD-DEM simulations.

This model uses scaling laws to map fine-scale geotechnical parameters (density, specific gravity, water content, Cu, Cc, clay content, permeability, etc.) to coarse-grained parameters for efficient engineering-scale simulation.

References:
- Liu et al. (2025), KSCE J. Civ. Eng.
- User's personal geotechnical parameters (see config)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from scipy import stats

class CoarseGrainedModel:
    def __init__(self, config: Dict):
        """Initialize the coarse-grained model.
        
        Args:
            config: Configuration dictionary containing model parameters, including geotechnical values and scaling factors.
        """
        self.config = config
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Compute scaling factors (see _compute_scaling_factors)
        self.scaling_factors = self._compute_scaling_factors()
        
        # Load calibrated parameters (from fine-scale calibration, e.g., triaxial test)
        self.calibrated_params = self._load_calibrated_params()
        
        # Initialize output directory
        self.output_dir = Path(config['output']['directory']) / 'coarse_grained'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Initialized coarse-grained model with scaling laws and geotechnical parameters")
    
    def _compute_scaling_factors(self) -> Dict:
        """Compute scaling factors for coarse-graining.
        
        Returns:
            Dictionary of scaling factors
        """
        # Get fine-scale parameters from config (geotechnical values)
        fine_scale = {
            'particle_radius': self.config['dem']['particle_radius'],
            'bond_strength': self.config['dem']['bond_strength'],
            'fluid_viscosity': self.config['cfd']['fluid_viscosity'],
            'fluid_density': self.config['cfd']['fluid_density']
        }
        # Scaling factors are set in config['coarse_grained']
        scaling_factors = {
            'length': self.config['coarse_grained'].get('length_scale', 10.0),
            'time': self.config['coarse_grained'].get('time_scale', 5.0),
            'force': self.config['coarse_grained'].get('force_scale', 8.0)
        }
        # Derived scaling factors
        scaling_factors['area'] = scaling_factors['length']**2
        scaling_factors['volume'] = scaling_factors['length']**3
        scaling_factors['velocity'] = scaling_factors['length'] / scaling_factors['time']
        scaling_factors['acceleration'] = scaling_factors['length'] / scaling_factors['time']**2
        return scaling_factors
    
    def _load_calibrated_params(self) -> Dict:
        """Load calibrated parameters from file (e.g., from fine-scale triaxial test).
        
        Returns:
            Dictionary of calibrated parameters
        """
        try:
            param_file = Path(self.config['coarse_grained']['calibration_file'])
            if param_file.exists():
                with open(param_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load calibrated parameters: {e}")
        return {}
    
    def map_fine_to_coarse(self, fine_scale_data: Dict) -> Dict:
        """Map fine-scale data to coarse-scale using scaling laws.
        
        Args:
            fine_scale_data: Dictionary containing fine-scale data (positions, velocities, forces, fluid fields, etc.)
        Returns:
            Dictionary containing coarse-scale data
        """
        coarse_data = {}
        # Map particle positions
        if 'positions' in fine_scale_data:
            coarse_data['positions'] = self._map_positions(
                fine_scale_data['positions'])
        # Map velocities
        if 'velocities' in fine_scale_data:
            coarse_data['velocities'] = self._map_velocities(
                fine_scale_data['velocities'])
        # Map forces
        if 'forces' in fine_scale_data:
            coarse_data['forces'] = self._map_forces(
                fine_scale_data['forces'])
        # Map fluid fields
        if 'fluid_fields' in fine_scale_data:
            coarse_data['fluid_fields'] = self._map_fluid_fields(
                fine_scale_data['fluid_fields'])
        return coarse_data
    
    def _map_positions(self, positions: np.ndarray) -> np.ndarray:
        """Map particle positions to coarse scale (scaling and clustering).
        
        Args:
            positions: Fine-scale particle positions
        Returns:
            Coarse-scale particle positions
        """
        # Scale positions by length scaling factor
        scaled_positions = positions / self.scaling_factors['length']
        # Cluster particles (coarse-graining)
        clusters = self._cluster_particles(scaled_positions)
        # Compute cluster centers
        cluster_centers = np.array([
            np.mean(scaled_positions[cluster], axis=0)
            for cluster in clusters
        ])
        return cluster_centers
    
    def _map_velocities(self, velocities: np.ndarray) -> np.ndarray:
        """Map particle velocities to coarse scale (scaling and clustering).
        
        Args:
            velocities: Fine-scale particle velocities
        Returns:
            Coarse-scale particle velocities
        """
        # Scale velocities by velocity scaling factor
        scaled_velocities = velocities / self.scaling_factors['velocity']
        # Cluster particles
        clusters = self._cluster_particles(scaled_velocities)
        # Compute cluster average velocities
        cluster_velocities = np.array([
            np.mean(scaled_velocities[cluster], axis=0)
            for cluster in clusters
        ])
        return cluster_velocities
    
    def _map_forces(self, forces: np.ndarray) -> np.ndarray:
        """Map particle forces to coarse scale (scaling and clustering).
        
        Args:
            forces: Fine-scale particle forces
        Returns:
            Coarse-scale particle forces
        """
        # Scale forces by force scaling factor
        scaled_forces = forces / self.scaling_factors['force']
        # Cluster particles
        clusters = self._cluster_particles(scaled_forces)
        # Compute cluster total forces
        cluster_forces = np.array([
            np.sum(scaled_forces[cluster], axis=0)
            for cluster in clusters
        ])
        return cluster_forces
    
    def _map_fluid_fields(self, fluid_fields: Dict) -> Dict:
        """Map fluid fields to coarse scale (scaling).
        
        Args:
            fluid_fields: Dictionary of fine-scale fluid fields
        Returns:
            Dictionary of coarse-scale fluid fields
        """
        coarse_fields = {}
        for field_name, field_data in fluid_fields.items():
            # Scale field data
            if field_name == 'velocity':
                scaled_data = field_data / self.scaling_factors['velocity']
            elif field_name == 'pressure':
                scaled_data = field_data / self.scaling_factors['force']
            else:
                scaled_data = field_data
            coarse_fields[field_name] = scaled_data
        return coarse_fields
    
    def _cluster_particles(self, data: np.ndarray) -> List[np.ndarray]:
        """Cluster particles based on spatial proximity.
        
        Args:
            data: Particle data (positions, velocities, etc.)
            
        Returns:
            List of particle indices for each cluster
        """
        # Compute pairwise distances
        distances = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                distances[i, j] = distances[j, i] = np.linalg.norm(
                    data[i] - data[j])
        
        # Cluster particles using hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        Z = linkage(distances)
        clusters = fcluster(Z, self.scaling_factors['length'], criterion='distance')
        
        # Group particles by cluster
        unique_clusters = np.unique(clusters)
        return [np.where(clusters == c)[0] for c in unique_clusters]
    
    def _coarsen_grid(self, field_data: np.ndarray) -> np.ndarray:
        """Coarsen grid data.
        
        Args:
            field_data: Fine-scale grid data
            
        Returns:
            Coarse-scale grid data
        """
        # Handle both scalar (3D) and vector (4D) fields
        if field_data.ndim == 3:
            nx, ny, nz = field_data.shape
            nvec = None
        elif field_data.ndim == 4:
            nx, ny, nz, nvec = field_data.shape
        else:
            raise ValueError("field_data must be 3D or 4D array")
        
        # Compute coarse grid dimensions
        scale = int(self.scaling_factors['length'])
        nx_coarse = nx // scale
        ny_coarse = ny // scale
        nz_coarse = nz // scale
        
        if field_data.ndim == 3:
            coarse_data = np.zeros((nx_coarse, ny_coarse, nz_coarse))
        else:
            coarse_data = np.zeros((nx_coarse, ny_coarse, nz_coarse, nvec))
        
        # Average fine grid data to coarse grid
        for i in range(nx_coarse):
            for j in range(ny_coarse):
                for k in range(nz_coarse):
                    i_fine = slice(i * scale, (i + 1) * scale)
                    j_fine = slice(j * scale, (j + 1) * scale)
                    k_fine = slice(k * scale, (k + 1) * scale)
                    if field_data.ndim == 3:
                        coarse_data[i, j, k] = np.mean(
                            field_data[i_fine, j_fine, k_fine])
                    else:
                        for v in range(nvec):
                            coarse_data[i, j, k, v] = np.mean(
                                field_data[i_fine, j_fine, k_fine, v])
        
        return coarse_data
    
    def validate_coarse_grained(self, fine_results: Dict,
                              coarse_results: Dict) -> Dict:
        """Validate coarse-grained results against fine-scale results.
        
        Args:
            fine_results: Fine-scale simulation results
            coarse_results: Coarse-scale simulation results
            
        Returns:
            Dictionary of validation metrics
        """
        validation = {}
        
        # Compare particle statistics
        validation['particle_stats'] = self._compare_particle_stats(
            fine_results, coarse_results)
        
        # Compare fluid statistics
        validation['fluid_stats'] = self._compare_fluid_stats(
            fine_results, coarse_results)
        
        # Compare erosion statistics
        validation['erosion_stats'] = self._compare_erosion_stats(
            fine_results, coarse_results)
        
        return validation
    
    def _compare_particle_stats(self, fine_results: Dict,
                              coarse_results: Dict) -> Dict:
        """Compare particle statistics between fine and coarse scales.
        
        Args:
            fine_results: Fine-scale results
            coarse_results: Coarse-scale results
            
        Returns:
            Dictionary of comparison metrics
        """
        stats = {}
        
        # Compare position distributions
        fine_pos = fine_results['positions']
        coarse_pos = coarse_results['positions']
        
        stats['position_correlation'] = np.corrcoef(
            fine_pos.flatten(), coarse_pos.flatten())[0, 1]
        
        # Compare velocity distributions
        fine_vel = fine_results['velocities']
        coarse_vel = coarse_results['velocities']
        
        stats['velocity_correlation'] = np.corrcoef(
            fine_vel.flatten(), coarse_vel.flatten())[0, 1]
        
        # Compare force distributions
        fine_force = fine_results['forces']
        coarse_force = coarse_results['forces']
        
        stats['force_correlation'] = np.corrcoef(
            fine_force.flatten(), coarse_force.flatten())[0, 1]
        
        return stats
    
    def _compare_fluid_stats(self, fine_results: Dict,
                           coarse_results: Dict) -> Dict:
        """Compare fluid statistics between fine and coarse scales.
        
        Args:
            fine_results: Fine-scale results
            coarse_results: Coarse-scale results
            
        Returns:
            Dictionary of comparison metrics
        """
        stats = {}
        
        # Compare velocity fields
        fine_vel = fine_results['fluid_fields']['velocity']
        coarse_vel = coarse_results['fluid_fields']['velocity']
        
        stats['velocity_correlation'] = np.corrcoef(
            fine_vel.flatten(), coarse_vel.flatten())[0, 1]
        
        # Compare pressure fields
        fine_pressure = fine_results['fluid_fields']['pressure']
        coarse_pressure = coarse_results['fluid_fields']['pressure']
        
        stats['pressure_correlation'] = np.corrcoef(
            fine_pressure.flatten(), coarse_pressure.flatten())[0, 1]
        
        return stats
    
    def _compare_erosion_stats(self, fine_results: Dict,
                             coarse_results: Dict) -> Dict:
        """Compare erosion statistics between fine and coarse scales.
        
        Args:
            fine_results: Fine-scale results
            coarse_results: Coarse-scale results
            
        Returns:
            Dictionary of comparison metrics
        """
        stats = {}
        
        # Compare erosion rates
        fine_rate = fine_results['erosion_rate']
        coarse_rate = coarse_results['erosion_rate']
        
        stats['erosion_rate_correlation'] = np.corrcoef(
            fine_rate, coarse_rate)[0, 1]
        
        # Compare total eroded volume
        fine_volume = fine_results['eroded_volume']
        coarse_volume = coarse_results['eroded_volume']
        
        stats['volume_error'] = np.abs(fine_volume - coarse_volume) / fine_volume
        
        return stats
    
    def save_results(self, results: Dict) -> None:
        """Save coarse-grained results to file.
        
        Args:
            results: Dictionary containing simulation results
        """
        # Save particle data
        np.savez(
            self.output_dir / 'particle_data.npz',
            positions=results['positions'],
            velocities=results['velocities'],
            forces=results['forces']
        )
        
        # Save fluid data
        np.savez(
            self.output_dir / 'fluid_data.npz',
            **results['fluid_fields']
        )
        
        # Save statistics
        with open(self.output_dir / 'statistics.json', 'w') as f:
            json.dump(results['statistics'], f, indent=4)
        
        self.logger.info("Saved coarse-grained results") 