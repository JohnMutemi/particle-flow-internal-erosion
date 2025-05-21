"""
Validation module for CFD-DEM simulation.
This module implements various validation mechanisms including experimental data comparison,
statistical analysis, and parameter sensitivity studies.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationManager:
    def __init__(self, config: Dict):
        """Initialize the validation manager."""
        self.config = config
        self.validation_data = {}
        self.statistics = {}
        self.sensitivity_results = {}
        
        # Create output directory for validation results
        self.output_dir = Path(config['output']['directory']) / 'validation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Validation manager initialized")

    def load_experimental_data(self, data_file: str):
        """Load experimental data for validation."""
        try:
            data = np.load(data_file)
            self.validation_data['experimental'] = {
                'time': data['time'],
                'erosion_rate': data['erosion_rate'],
                'pressure': data['pressure'],
                'velocity': data['velocity']
            }
            logger.info(f"Loaded experimental data from {data_file}")
        except Exception as e:
            logger.error(f"Failed to load experimental data: {e}")
            raise

    def compare_with_experimental(self, simulation_results: Dict) -> Dict:
        """Compare simulation results with experimental data."""
        if 'experimental' not in self.validation_data:
            raise ValueError("No experimental data loaded")
        
        comparison = {}
        
        # Compare erosion rates
        exp_erosion = self.validation_data['experimental']['erosion_rate']
        sim_erosion = simulation_results['erosion_stats']
        
        # Compute statistics
        comparison['erosion'] = {
            'mean_error': np.mean(np.abs(exp_erosion - sim_erosion)),
            'max_error': np.max(np.abs(exp_erosion - sim_erosion)),
            'correlation': np.corrcoef(exp_erosion, sim_erosion)[0, 1],
            'rms_error': np.sqrt(np.mean((exp_erosion - sim_erosion)**2))
        }
        
        # Compare pressure distributions
        exp_pressure = self.validation_data['experimental']['pressure']
        sim_pressure = simulation_results['fluid_data']['pressure_field']
        
        comparison['pressure'] = {
            'mean_error': np.mean(np.abs(exp_pressure - sim_pressure)),
            'max_error': np.max(np.abs(exp_pressure - sim_pressure)),
            'correlation': np.corrcoef(exp_pressure.flatten(), sim_pressure.flatten())[0, 1],
            'rms_error': np.sqrt(np.mean((exp_pressure - sim_pressure)**2))
        }
        
        # Compare velocity profiles
        exp_velocity = self.validation_data['experimental']['velocity']
        sim_velocity = simulation_results['fluid_data']['velocity_field']
        
        comparison['velocity'] = {
            'mean_error': np.mean(np.abs(exp_velocity - sim_velocity)),
            'max_error': np.max(np.abs(exp_velocity - sim_velocity)),
            'correlation': np.corrcoef(exp_velocity.flatten(), sim_velocity.flatten())[0, 1],
            'rms_error': np.sqrt(np.mean((exp_velocity - sim_velocity)**2))
        }
        
        self.statistics['experimental_comparison'] = comparison
        logger.info("Completed experimental data comparison")
        
        return comparison

    def perform_statistical_analysis(self, simulation_results: Dict):
        """Perform statistical analysis of simulation results."""
        # Analyze erosion statistics
        erosion_stats = simulation_results['erosion_stats']
        self.statistics['erosion'] = {
            'mean': np.mean(erosion_stats),
            'std': np.std(erosion_stats),
            'min': np.min(erosion_stats),
            'max': np.max(erosion_stats),
            'skewness': stats.skew(erosion_stats),
            'kurtosis': stats.kurtosis(erosion_stats)
        }
        
        # Analyze particle distribution
        particle_positions = np.array(simulation_results['particle_data'])
        self.statistics['particles'] = {
            'mean_position': np.mean(particle_positions, axis=0),
            'std_position': np.std(particle_positions, axis=0),
            'spatial_correlation': self._compute_spatial_correlation(particle_positions)
        }
        
        # Analyze fluid statistics
        fluid_data = simulation_results['fluid_data']
        self.statistics['fluid'] = {
            'mean_velocity': np.mean(fluid_data['velocity_field']),
            'std_velocity': np.std(fluid_data['velocity_field']),
            'mean_pressure': np.mean(fluid_data['pressure_field']),
            'std_pressure': np.std(fluid_data['pressure_field'])
        }
        
        logger.info("Completed statistical analysis")

    def _compute_spatial_correlation(self, positions: np.ndarray) -> np.ndarray:
        """Compute spatial correlation of particle positions."""
        n_particles = positions.shape[1]
        correlation = np.zeros((n_particles, n_particles))
        
        for i in range(n_particles):
            for j in range(n_particles):
                correlation[i, j] = np.corrcoef(
                    positions[:, i, 0],
                    positions[:, j, 0]
                )[0, 1]
        
        return correlation

    def perform_sensitivity_analysis(self, base_config: Dict,
                                  parameters: List[str],
                                  ranges: List[Tuple[float, float]],
                                  num_samples: int = 10):
        """Perform sensitivity analysis for specified parameters."""
        results = {}
        
        for param, (min_val, max_val) in zip(parameters, ranges):
            param_results = []
            
            # Generate parameter values
            values = np.linspace(min_val, max_val, num_samples)
            
            for value in values:
                # Create modified configuration
                config = base_config.copy()
                self._set_parameter_value(config, param, value)
                
                # Run simulation with modified configuration
                # Note: This requires integration with the simulation framework
                simulation_results = self._run_simulation(config)
                
                # Store results
                param_results.append({
                    'value': value,
                    'erosion_rate': np.mean(simulation_results['erosion_stats']),
                    'pressure': np.mean(simulation_results['fluid_data']['pressure_field']),
                    'velocity': np.mean(simulation_results['fluid_data']['velocity_field'])
                })
            
            results[param] = param_results
        
        self.sensitivity_results = results
        logger.info("Completed sensitivity analysis")
        
        return results

    def _set_parameter_value(self, config: Dict, parameter: str, value: float):
        """Set parameter value in configuration dictionary."""
        # Split parameter path
        parts = parameter.split('.')
        
        # Navigate to correct location in config
        current = config
        for part in parts[:-1]:
            current = current[part]
        
        # Set value
        current[parts[-1]] = value

    def _run_simulation(self, config: Dict) -> Dict:
        """Run simulation with given configuration."""
        # This method should be implemented to integrate with the simulation framework
        # For now, it's a placeholder
        raise NotImplementedError("Simulation integration not implemented")

    def plot_validation_results(self):
        """Plot validation results."""
        # Create figure for experimental comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot erosion rate comparison
        if 'experimental_comparison' in self.statistics:
            exp_data = self.validation_data['experimental']
            axes[0, 0].plot(exp_data['time'], exp_data['erosion_rate'],
                           'b-', label='Experimental')
            axes[0, 0].set_title('Erosion Rate Comparison')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Erosion Rate')
            axes[0, 0].legend()
        
        # Plot pressure distribution
        if 'pressure' in self.statistics:
            axes[0, 1].hist(self.statistics['pressure']['mean_pressure'],
                           bins=30, alpha=0.5, label='Simulation')
            axes[0, 1].set_title('Pressure Distribution')
            axes[0, 1].set_xlabel('Pressure')
            axes[0, 1].set_ylabel('Frequency')
        
        # Plot velocity profile
        if 'velocity' in self.statistics:
            axes[1, 0].plot(self.statistics['velocity']['mean_velocity'],
                           label='Mean Velocity')
            axes[1, 0].set_title('Velocity Profile')
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('Velocity')
        
        # Plot sensitivity analysis results
        if self.sensitivity_results:
            for param, results in self.sensitivity_results.items():
                values = [r['value'] for r in results]
                erosion_rates = [r['erosion_rate'] for r in results]
                axes[1, 1].plot(values, erosion_rates, label=param)
            
            axes[1, 1].set_title('Parameter Sensitivity')
            axes[1, 1].set_xlabel('Parameter Value')
            axes[1, 1].set_ylabel('Erosion Rate')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'validation_results.png')
        plt.close()
        
        logger.info("Validation results plotted and saved")

    def save_validation_results(self):
        """Save validation results to file."""
        results = {
            'statistics': self.statistics,
            'sensitivity_results': self.sensitivity_results
        }
        
        np.savez(
            self.output_dir / 'validation_results.npz',
            **results
        )
        
        logger.info(f"Validation results saved to {self.output_dir}") 