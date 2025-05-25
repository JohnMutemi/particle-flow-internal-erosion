"""
Validation manager for CFD-DEM simulations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy import stats

class ValidationManager:
    def __init__(self, config: Dict):
        """Initialize the validation manager.
        
        Args:
            config: Configuration dictionary containing validation parameters
        """
        self.config = config
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load validation parameters
        self.validation_params = self._load_validation_params()
        
        # Initialize output directory
        self.output_dir = Path(config['output']['directory']) / 'validation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Initialized validation manager")
    
    def _load_validation_params(self) -> Dict:
        """Load validation parameters from configuration.
        
        Returns:
            Dictionary of validation parameters
        """
        params = {
            'experimental_data_file': self.config['validation'].get('experimental_data_file', ''),
            'variables_to_compare': self.config['validation'].get('variables_to_compare', []),
            'statistics': self.config['validation'].get('statistics', {}),
            'sensitivity_analysis': self.config['validation'].get('sensitivity_analysis', {})
        }
        
        return params
    
    def load_experimental_data(self) -> Dict:
        """Load experimental data from file.
        
        Returns:
            Dictionary containing experimental data
        """
        try:
            data_file = Path(self.validation_params['experimental_data_file'])
            if data_file.exists():
                data = np.load(data_file)
                self.logger.info("Loaded experimental data")
                return dict(data)
        except Exception as e:
            self.logger.error(f"Error loading experimental data: {e}")
        
        return {}
    
    def compare_with_experimental(self, simulation_results: Dict,
                                experimental_data: Dict) -> Dict:
        """Compare simulation results with experimental data.
        
        Args:
            simulation_results: Dictionary containing simulation results
            experimental_data: Dictionary containing experimental data
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison = {}
        
        # Compare each variable
        for variable in self.validation_params['variables_to_compare']:
            if variable in simulation_results and variable in experimental_data:
                comparison[variable] = self._compare_variable(
                    simulation_results[variable],
                    experimental_data[variable])
        
        return comparison
    
    def _compare_variable(self, sim_data: np.ndarray,
                         exp_data: np.ndarray) -> Dict:
        """Compare a single variable between simulation and experiment.
        
        Args:
            sim_data: Simulation data
            exp_data: Experimental data
            
        Returns:
            Dictionary of comparison metrics
        """
        metrics = {}
        
        # Compute mean error
        metrics['mean_error'] = np.mean(np.abs(sim_data - exp_data))
        
        # Compute maximum error
        metrics['max_error'] = np.max(np.abs(sim_data - exp_data))
        
        # Compute correlation
        metrics['correlation'] = np.corrcoef(sim_data.flatten(),
                                           exp_data.flatten())[0, 1]
        
        # Compute RMS error
        metrics['rms_error'] = np.sqrt(np.mean((sim_data - exp_data)**2))
        
        return metrics
    
    def perform_statistical_analysis(self, simulation_results: Dict) -> Dict:
        """Perform statistical analysis on simulation results.
        
        Args:
            simulation_results: Dictionary containing simulation results
            
        Returns:
            Dictionary containing statistical analysis results
        """
        statistics = {}
        
        # Analyze erosion statistics
        if 'erosion_rate' in simulation_results:
            statistics['erosion'] = self._analyze_erosion_statistics(
                simulation_results['erosion_rate'])
        
        # Analyze particle statistics
        if 'particle_data' in simulation_results:
            statistics['particles'] = self._analyze_particle_statistics(
                simulation_results['particle_data'])
        
        # Analyze fluid statistics
        if 'fluid_fields' in simulation_results:
            statistics['fluid'] = self._analyze_fluid_statistics(
                simulation_results['fluid_fields'])
        
        return statistics
    
    def _analyze_erosion_statistics(self, erosion_rate: np.ndarray) -> Dict:
        """Analyze erosion rate statistics.
        
        Args:
            erosion_rate: Array of erosion rates
            
        Returns:
            Dictionary of erosion statistics
        """
        result = {}
        
        # Basic statistics
        result['mean'] = np.mean(erosion_rate)
        result['std'] = np.std(erosion_rate)
        result['max'] = np.max(erosion_rate)
        result['min'] = np.min(erosion_rate)
        
        # Distribution statistics
        result['skewness'] = stats.skew(erosion_rate)
        result['kurtosis'] = stats.kurtosis(erosion_rate)
        
        return result
    
    def _analyze_particle_statistics(self, particle_data: Dict) -> Dict:
        """Analyze particle statistics.
        
        Args:
            particle_data: Dictionary containing particle data
            
        Returns:
            Dictionary of particle statistics
        """
        result = {}
        
        # Analyze positions
        if 'positions' in particle_data:
            positions = particle_data['positions']
            result['position'] = {
                'mean': np.mean(positions, axis=0),
                'std': np.std(positions, axis=0),
                'distribution': self._analyze_distribution(positions)
            }
        
        # Analyze velocities
        if 'velocities' in particle_data:
            velocities = particle_data['velocities']
            result['velocity'] = {
                'mean': np.mean(velocities, axis=0),
                'std': np.std(velocities, axis=0),
                'distribution': self._analyze_distribution(velocities)
            }
        
        # Analyze forces
        if 'forces' in particle_data:
            forces = particle_data['forces']
            result['force'] = {
                'mean': np.mean(forces, axis=0),
                'std': np.std(forces, axis=0),
                'distribution': self._analyze_distribution(forces)
            }
        
        return result
    
    def _analyze_fluid_statistics(self, fluid_fields: Dict) -> Dict:
        """Analyze fluid field statistics.
        
        Args:
            fluid_fields: Dictionary containing fluid field data
            
        Returns:
            Dictionary of fluid statistics
        """
        result = {}
        
        # Analyze velocity field
        if 'velocity' in fluid_fields:
            velocity = fluid_fields['velocity']
            result['velocity'] = {
                'mean': np.mean(velocity, axis=(0, 1, 2)),
                'std': np.std(velocity, axis=(0, 1, 2)),
                'max': np.max(velocity, axis=(0, 1, 2)),
                'min': np.min(velocity, axis=(0, 1, 2))
            }
        
        # Analyze pressure field
        if 'pressure' in fluid_fields:
            pressure = fluid_fields['pressure']
            result['pressure'] = {
                'mean': np.mean(pressure),
                'std': np.std(pressure),
                'max': np.max(pressure),
                'min': np.min(pressure)
            }
        
        return result
    
    def _analyze_distribution(self, data: np.ndarray) -> Dict:
        """Analyze data distribution.
        
        Args:
            data: Array of data
            
        Returns:
            Dictionary of distribution statistics
        """
        dist = {}
        
        # Compute histogram
        hist, bins = np.histogram(data.flatten(), bins=50, density=True)
        dist['histogram'] = {'values': hist, 'bins': bins}
        
        # Fit normal distribution
        mean = np.mean(data)
        std = np.std(data)
        dist['normal_fit'] = {
            'mean': mean,
            'std': std
        }
        
        # Compute percentiles
        dist['percentiles'] = {
            '25': np.percentile(data, 25),
            '50': np.percentile(data, 50),
            '75': np.percentile(data, 75)
        }
        
        return dist
    
    def perform_sensitivity_analysis(self, simulation_results: Dict,
                                   parameters: List[str]) -> Dict:
        """Perform sensitivity analysis on simulation results.
        
        Args:
            simulation_results: Dictionary containing simulation results
            parameters: List of parameters to analyze
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        sensitivity = {}
        
        # Get parameter ranges
        param_ranges = self.validation_params['sensitivity_analysis'].get(
            'parameter_ranges', {})
        
        # Analyze each parameter
        for param in parameters:
            if param in param_ranges:
                sensitivity[param] = self._analyze_parameter_sensitivity(
                    simulation_results, param, param_ranges[param])
        
        return sensitivity
    
    def _analyze_parameter_sensitivity(self, results: Dict,
                                     parameter: str,
                                     param_range: Tuple[float, float]) -> Dict:
        """Analyze sensitivity of results to a parameter.
        
        Args:
            results: Dictionary containing simulation results
            parameter: Parameter to analyze
            param_range: Range of parameter values
            
        Returns:
            Dictionary of sensitivity analysis results
        """
        sensitivity = {}
        
        # Generate parameter values
        param_values = np.linspace(param_range[0], param_range[1], 10)
        
        # Compute results for each parameter value
        results_values = []
        for value in param_values:
            # Update parameter value
            results_copy = results.copy()
            results_copy[parameter] = value
            
            # Compute metrics
            metrics = self._compute_sensitivity_metrics(results_copy)
            results_values.append(metrics)
        
        # Compute sensitivity indices
        sensitivity['main_effect'] = self._compute_main_effect(
            param_values, results_values)
        sensitivity['interaction'] = self._compute_interaction_effects(
            param_values, results_values)
        
        return sensitivity
    
    def _compute_sensitivity_metrics(self, results: Dict) -> Dict:
        """Compute metrics for sensitivity analysis.
        
        Args:
            results: Dictionary containing simulation results
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Compute erosion metrics
        if 'erosion_rate' in results:
            metrics['erosion'] = {
                'mean': np.mean(results['erosion_rate']),
                'max': np.max(results['erosion_rate']),
                'std': np.std(results['erosion_rate'])
            }
        
        # Compute fluid metrics
        if 'fluid_fields' in results:
            metrics['fluid'] = {
                'velocity_mean': np.mean(results['fluid_fields']['velocity']),
                'pressure_mean': np.mean(results['fluid_fields']['pressure'])
            }
        
        return metrics
    
    def _compute_main_effect(self, param_values: np.ndarray,
                           results_values: List[Dict]) -> Dict:
        """Compute main effect of parameter on results.
        
        Args:
            param_values: Array of parameter values
            results_values: List of result dictionaries
            
        Returns:
            Dictionary of main effects
        """
        main_effect = {}
        
        # Compute main effect for each metric
        for metric in results_values[0].keys():
            values = [r[metric] for r in results_values]
            # If the value is a dict, compute for each submetric
            if isinstance(values[0], dict):
                main_effect[metric] = {}
                for submetric in values[0].keys():
                    subvalues = [v[submetric] for v in values]
                    main_effect[metric][submetric] = np.polyfit(param_values, subvalues, 1)[0]
            else:
                main_effect[metric] = np.polyfit(param_values, values, 1)[0]
        
        return main_effect
    
    def _compute_interaction_effects(self, param_values: np.ndarray,
                                  results_values: List[Dict]) -> Dict:
        """Compute interaction effects between parameters.
        
        Args:
            param_values: Array of parameter values
            results_values: List of result dictionaries
            
        Returns:
            Dictionary of interaction effects
        """
        interaction = {}
        
        # Compute interaction for each metric
        for metric in results_values[0].keys():
            values = [r[metric] for r in results_values]
            # If the value is a dict, compute for each submetric
            if isinstance(values[0], dict):
                interaction[metric] = {}
                for submetric in values[0].keys():
                    subvalues = [v[submetric] for v in values]
                    interaction[metric][submetric] = np.polyfit(param_values, subvalues, 2)[0]
            else:
                interaction[metric] = np.polyfit(param_values, values, 2)[0]
        
        return interaction
    
    def plot_validation_results(self, comparison: Dict,
                              statistics: Dict,
                              sensitivity: Dict) -> None:
        """Plot validation results.
        
        Args:
            comparison: Dictionary containing comparison metrics
            statistics: Dictionary containing statistical analysis results
            sensitivity: Dictionary containing sensitivity analysis results
        """
        # Plot comparison results
        self._plot_comparison_results(comparison)
        
        # Plot statistical analysis
        self._plot_statistical_analysis(statistics)
        
        # Plot sensitivity analysis
        self._plot_sensitivity_analysis(sensitivity)
    
    def _plot_comparison_results(self, comparison: Dict) -> None:
        """Plot comparison results.
        
        Args:
            comparison: Dictionary containing comparison metrics
        """
        for variable, metrics in comparison.items():
            plt.figure(figsize=(10, 6))
            
            # Plot metrics
            plt.bar(metrics.keys(), metrics.values())
            plt.title(f'Comparison Metrics for {variable}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(self.output_dir / f'comparison_{variable}.png')
            plt.close()
    
    def _plot_statistical_analysis(self, statistics: Dict) -> None:
        """Plot statistical analysis results.
        
        Args:
            statistics: Dictionary containing statistical analysis results
        """
        import numbers
        for category, stats in statistics.items():
            plt.figure(figsize=(10, 6))
            # Flatten stats dict
            flat_stats = {}
            for k, v in stats.items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        if isinstance(subv, (int, float, np.generic)):
                            flat_stats[f"{k}_{subk}"] = subv
                elif isinstance(v, (int, float, np.generic)):
                    flat_stats[k] = v
            # Plot statistics
            plt.bar(flat_stats.keys(), flat_stats.values())
            plt.title(f'Statistics for {category}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            # Save plot
            plt.savefig(self.output_dir / f'statistics_{category}.png')
            plt.close()
    
    def _plot_sensitivity_analysis(self, sensitivity: Dict) -> None:
        """Plot sensitivity analysis results.
        
        Args:
            sensitivity: Dictionary containing sensitivity analysis results
        """
        import numbers
        for parameter, results in sensitivity.items():
            plt.figure(figsize=(10, 6))
            # Flatten main_effect dict
            flat_main = {}
            for k, v in results['main_effect'].items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        if isinstance(subv, (int, float, np.generic)):
                            flat_main[f"{k}_{subk}"] = subv
                elif isinstance(v, (int, float, np.generic)):
                    flat_main[k] = v
            # Plot main effects
            plt.bar(flat_main.keys(), flat_main.values())
            plt.title(f'Main Effects for {parameter}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            # Save plot
            plt.savefig(self.output_dir / f'sensitivity_{parameter}.png')
            plt.close()
    
    def save_validation_results(self, comparison: Dict,
                              statistics: Dict,
                              sensitivity: Dict) -> None:
        """Save validation results to file.
        
        Args:
            comparison: Dictionary containing comparison metrics
            statistics: Dictionary containing statistical analysis results
            sensitivity: Dictionary containing sensitivity analysis results
        """
        # Save comparison results
        with open(self.output_dir / 'comparison_results.json', 'w') as f:
            json.dump(comparison, f, indent=4)
        
        # Save statistical analysis
        with open(self.output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(statistics, f, indent=4)
        
        # Save sensitivity analysis
        with open(self.output_dir / 'sensitivity_analysis.json', 'w') as f:
            json.dump(sensitivity, f, indent=4)
        
        self.logger.info("Saved validation results") 