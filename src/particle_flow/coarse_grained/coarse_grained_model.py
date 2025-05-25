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
from scipy.cluster.hierarchy import linkage, fcluster

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
        
        # Load geotechnical parameters
        self.geotech_params = self._load_geotech_params()
        
        # Compute scaling factors (see _compute_scaling_factors)
        self.scaling_factors = self._compute_scaling_factors()
        
        # Load calibrated parameters (from fine-scale calibration, e.g., triaxial test)
        self.calibrated_params = self._load_calibrated_params()
        
        # Initialize output directory
        self.output_dir = Path(config['output']['directory']) / 'coarse_grained'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Initialized coarse-grained model with scaling laws and geotechnical parameters")
    
    def _load_geotech_params(self) -> Dict:
        """Load geotechnical parameters from config."""
        return {
            'clay_content': self.config['geotechnical']['clay_content'],
            'water_content': self.config['geotechnical']['water_content'],
            'Cu': self.config['geotechnical']['Cu'],
            'Cc': self.config['geotechnical']['Cc'],
            'cohesion': self.config['geotechnical']['cohesion'],
            'permeability': self.config['geotechnical']['permeability'],
            'density': self.config['geotechnical']['density'],
            'specific_gravity': self.config['geotechnical']['specific_gravity'],
            'porosity': self.config['geotechnical']['porosity'],
            'void_ratio': self.config['geotechnical']['void_ratio']
        }
    
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
        
        # Base scaling factors from config
        scaling_factors = {
            'length': self.config['coarse_grained'].get('length_scale', 10.0),
            'time': self.config['coarse_grained'].get('time_scale', 5.0),
            'force': self.config['coarse_grained'].get('force_scale', 8.0)
        }
        
        # Adjust scaling factors based on geotechnical parameters
        clay_factor = 1.0 + 0.1 * (self.geotech_params['clay_content'] / 20.0)
        water_factor = 1.0 - 0.2 * (self.geotech_params['water_content'] / 20.0)
        cu_factor = 1.0 + 0.1 * (self.geotech_params['Cu'] / 10.0)
        
        # Apply geotechnical adjustments
        scaling_factors['length'] *= clay_factor
        scaling_factors['time'] *= water_factor
        scaling_factors['force'] *= cu_factor
        
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
        
        # Map particle positions with geotechnical adjustments
        if 'positions' in fine_scale_data:
            coarse_data['positions'] = self._map_positions(
                fine_scale_data['positions'])
        
        # Map velocities with geotechnical adjustments
        if 'velocities' in fine_scale_data:
            coarse_data['velocities'] = self._map_velocities(
                fine_scale_data['velocities'])
        
        # Map forces with geotechnical adjustments
        if 'forces' in fine_scale_data:
            coarse_data['forces'] = self._map_forces(
                fine_scale_data['forces'])
        
        # Map fluid fields with geotechnical adjustments
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
        
        # Adjust for clay content effect on clustering
        clay_factor = 1.0 + 0.2 * (self.geotech_params['clay_content'] / 20.0)
        clustering_distance = self.scaling_factors['length'] * clay_factor
        
        # Cluster particles
        clusters = self._cluster_particles(scaled_positions, clustering_distance)
        
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
        
        # Adjust for water content effect on velocity mapping
        water_factor = 1.0 - 0.3 * (self.geotech_params['water_content'] / 20.0)
        scaled_velocities *= water_factor
        
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
        
        # Adjust for Cu effect on force mapping
        cu_factor = 1.0 + 0.2 * (self.geotech_params['Cu'] / 10.0)
        scaled_forces *= cu_factor
        
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
                # Adjust for permeability effect
                perm_factor = self.geotech_params['permeability'] / 1e-6  # Normalize to 1e-6 m/s
                scaled_data *= perm_factor
            elif field_name == 'pressure':
                scaled_data = field_data / self.scaling_factors['force']
                # Adjust for porosity effect
                porosity_factor = self.geotech_params['porosity'] / 0.3  # Normalize to 0.3
                scaled_data *= porosity_factor
            else:
                scaled_data = field_data
            
            coarse_fields[field_name] = scaled_data
        
        return coarse_fields
    
    def _cluster_particles(self, data: np.ndarray, distance_threshold: Optional[float] = None) -> List[np.ndarray]:
        """Cluster particles based on spatial proximity.
        
        Args:
            data: Particle data (positions, velocities, etc.)
            distance_threshold: Optional distance threshold for clustering
            
        Returns:
            List of particle indices for each cluster
        """
        if distance_threshold is None:
            distance_threshold = self.scaling_factors['length']
        
        # Compute pairwise distances
        distances = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                distances[i, j] = distances[j, i] = np.linalg.norm(
                    data[i] - data[j])
        
        # Cluster particles using hierarchical clustering
        Z = linkage(distances)
        clusters = fcluster(Z, distance_threshold, criterion='distance')
        
        # Group particles by cluster
        unique_clusters = np.unique(clusters)
        return [np.where(clusters == c)[0] for c in unique_clusters]
    
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
        
        # Add geotechnical parameter effects
        validation['geotech_effects'] = self._analyze_geotech_effects(
            fine_results, coarse_results)
        
        return validation
    
    def _analyze_geotech_effects(self, fine_results: Dict,
                               coarse_results: Dict) -> Dict:
        """Analyze effects of geotechnical parameters on coarse-graining.
        
        Args:
            fine_results: Fine-scale results
            coarse_results: Coarse-scale results
            
        Returns:
            Dictionary of geotechnical effects
        """
        effects = {}
        
        # Analyze clay content effect
        effects['clay_content'] = {
            'clustering_quality': self._compute_clustering_quality(
                fine_results, coarse_results),
            'force_scaling': self._compute_force_scaling_accuracy(
                fine_results, coarse_results)
        }
        
        # Analyze water content effect
        effects['water_content'] = {
            'velocity_scaling': self._compute_velocity_scaling_accuracy(
                fine_results, coarse_results),
            'permeability_effect': self._compute_permeability_effect(
                fine_results, coarse_results)
        }
        
        # Analyze Cu effect
        effects['Cu'] = {
            'particle_distribution': self._compute_particle_distribution(
                fine_results, coarse_results),
            'force_distribution': self._compute_force_distribution(
                fine_results, coarse_results)
        }
        
        return effects
    
    def _compute_clustering_quality(self, fine_results: Dict,
                                  coarse_results: Dict) -> float:
        """Compute quality of particle clustering."""
        fine_pos = fine_results['positions']
        coarse_pos = coarse_results['positions']
        
        # Compute average distance between fine and coarse particles
        distances = []
        for coarse_pos in coarse_pos:
            min_dist = np.min(np.linalg.norm(fine_pos - coarse_pos, axis=1))
            distances.append(min_dist)
        
        return np.mean(distances)
    
    def _compute_force_scaling_accuracy(self, fine_results: Dict,
                                      coarse_results: Dict) -> float:
        """Compute accuracy of force scaling."""
        fine_forces = fine_results['forces']
        coarse_forces = coarse_results['forces']
        
        # Compute correlation between fine and coarse forces
        return np.corrcoef(fine_forces.flatten(), coarse_forces.flatten())[0, 1]
    
    def _compute_velocity_scaling_accuracy(self, fine_results: Dict,
                                         coarse_results: Dict) -> float:
        """Compute accuracy of velocity scaling."""
        fine_vel = fine_results['velocities']
        coarse_vel = coarse_results['velocities']
        
        # Compute correlation between fine and coarse velocities
        return np.corrcoef(fine_vel.flatten(), coarse_vel.flatten())[0, 1]
    
    def _compute_permeability_effect(self, fine_results: Dict,
                                   coarse_results: Dict) -> float:
        """Compute effect of permeability on fluid flow."""
        fine_fluid = fine_results['fluid_fields']['velocity']
        coarse_fluid = coarse_results['fluid_fields']['velocity']
        
        # Compute correlation between fine and coarse fluid velocities
        return np.corrcoef(fine_fluid.flatten(), coarse_fluid.flatten())[0, 1]
    
    def _compute_particle_distribution(self, fine_results: Dict,
                                     coarse_results: Dict) -> float:
        """Compute accuracy of particle distribution mapping."""
        fine_pos = fine_results['positions']
        coarse_pos = coarse_results['positions']
        
        # Compute Kolmogorov-Smirnov test statistic
        ks_stat, _ = stats.ks_2samp(fine_pos.flatten(), coarse_pos.flatten())
        return 1.0 - ks_stat  # Convert to similarity measure
    
    def _compute_force_distribution(self, fine_results: Dict,
                                  coarse_results: Dict) -> float:
        """Compute accuracy of force distribution mapping."""
        fine_forces = fine_results['forces']
        coarse_forces = coarse_results['forces']
        
        # Compute Kolmogorov-Smirnov test statistic
        ks_stat, _ = stats.ks_2samp(fine_forces.flatten(), coarse_forces.flatten())
        return 1.0 - ks_stat  # Convert to similarity measure
    
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
        
        # Save geotechnical effects
        if 'geotech_effects' in results:
            with open(self.output_dir / 'geotech_effects.json', 'w') as f:
                json.dump(results['geotech_effects'], f, indent=4)
        
        self.logger.info("Saved coarse-grained results") 