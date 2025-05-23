"""
Validation module for CFD-DEM simulation.
This module implements various validation mechanisms including experimental data comparison,
statistical analysis, and parameter sensitivity studies.

Geotechnical parameters (density, specific gravity, water content, Cu, Cc, clay content, permeability, etc.)
are used for validation, calibration, and sensitivity analysis. These parameters are loaded from the configuration
and experimental data files, and are referenced throughout the validation process.

References:
- Liu et al. (2025), KSCE J. Civ. Eng.
- User's personal geotechnical parameters (see config)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import yaml
import json
import h5py
import openpyxl
from datetime import datetime

logger = logging.getLogger(__name__)

class ValidationManager:
    def __init__(self, config: Dict):
        """Initialize the validation manager.
        
        Args:
            config: Configuration dictionary containing validation parameters, including geotechnical values for calibration and comparison.
        """
        self.config = config
        self.validation_data = {}
        self.statistics = {}
        self.sensitivity_results = {}
        
        # Create output directory for validation results
        self.output_dir = Path(config['output']['directory']) / 'validation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Validation manager initialized with geotechnical parameters")

    def load_experimental_data(self, data_file: str, data_type: str = 'triaxial'):
        """Load experimental data for validation.
        
        Args:
            data_file: Path to experimental data file (CSV, NPZ, YAML, Excel, JSON, or HDF5)
            data_type: Type of experimental data ('triaxial' or 'seepage')
        """
        try:
            file_path = Path(data_file)
            if not file_path.exists():
                raise FileNotFoundError(f"Experimental data file not found: {data_file}")

            # Load data based on file extension
            if file_path.suffix == '.npz':
                data = np.load(data_file)
                self._process_numpy_data(data)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(data_file)
                self._process_dataframe(df)
            elif file_path.suffix in ['.yaml', '.yml']:
                with open(data_file, 'r') as f:
                    data = yaml.safe_load(f)
                self._process_yaml_data(data)
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(data_file)
                self._process_dataframe(df)
            elif file_path.suffix == '.json':
                with open(data_file, 'r') as f:
                    data = json.load(f)
                self._process_json_data(data)
            elif file_path.suffix in ['.h5', '.hdf5']:
                with h5py.File(data_file, 'r') as f:
                    self._process_hdf5_data(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Load geotechnical parameters if available
            params_file = file_path.parent / f"{data_type}_parameters.yaml"
            if params_file.exists():
                with open(params_file, 'r') as f:
                    self.validation_data['geotechnical_params'] = yaml.safe_load(f)

            logger.info(f"Loaded experimental data from {data_file}")
            logger.info(f"Data type: {data_type}")
            logger.info(f"Available measurements: {list(self.validation_data['experimental'].keys())}")

        except Exception as e:
            logger.error(f"Failed to load experimental data: {e}")
            raise

    def _process_numpy_data(self, data: np.ndarray):
        """Process data from numpy file."""
        self.validation_data['experimental'] = {
            'time': data['time'],
            'erosion_rate': data['erosion_rate'],
            'pressure': data['pressure'],
            'velocity': data['velocity']
        }

    def _process_dataframe(self, df: pd.DataFrame):
        """Process data from pandas DataFrame."""
        required_columns = ['time', 'erosion_rate', 'pressure', 'velocity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.validation_data['experimental'] = {
            'time': df['time'].values,
            'erosion_rate': df['erosion_rate'].values,
            'pressure': df['pressure'].values,
            'velocity': df['velocity'].values
        }
        
        # Add any additional columns as metadata
        additional_columns = [col for col in df.columns if col not in required_columns]
        if additional_columns:
            self.validation_data['metadata'] = {
                col: df[col].values for col in additional_columns
            }

    def _process_yaml_data(self, data: Dict):
        """Process data from YAML file."""
        self.validation_data['experimental'] = {
            'time': np.array(data['time']),
            'erosion_rate': np.array(data['erosion_rate']),
            'pressure': np.array(data['pressure']),
            'velocity': np.array(data['velocity'])
        }

    def _process_json_data(self, data: Dict):
        """Process data from JSON file."""
        self.validation_data['experimental'] = {
            'time': np.array(data['time']),
            'erosion_rate': np.array(data['erosion_rate']),
            'pressure': np.array(data['pressure']),
            'velocity': np.array(data['velocity'])
        }

    def _process_hdf5_data(self, data: h5py.File):
        """Process data from HDF5 file."""
        self.validation_data['experimental'] = {
            'time': data['time'][:],
            'erosion_rate': data['erosion_rate'][:],
            'pressure': data['pressure'][:],
            'velocity': data['velocity'][:]
        }

    def compare_with_experimental(self, simulation_results: Dict) -> Dict:
        """Compare simulation results with experimental data using geotechnical parameters for calibration.
        
        Args:
            simulation_results: Dictionary containing simulation outputs (erosion, pressure, velocity, etc.)
        Returns:
            Dictionary of comparison statistics
        """
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
        logger.info("Completed experimental data comparison using geotechnical parameters")
        
        return comparison

    def perform_statistical_analysis(self, simulation_results: Dict):
        """Perform statistical analysis of simulation results, referencing geotechnical parameters for interpretation.
        
        Args:
            simulation_results: Dictionary containing simulation outputs
        """
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
        
        logger.info("Completed statistical analysis with geotechnical context")

    def _compute_spatial_correlation(self, positions: np.ndarray) -> np.ndarray:
        """Compute spatial correlation of particle positions.
        
        Args:
            positions: Array of particle positions, shape (N_particles, 3)
        Returns:
            Correlation matrix
        """
        # Compute correlation between the x-coordinates of all particles
        correlation = np.corrcoef(positions[:, 0])
        return correlation

    def perform_sensitivity_analysis(self, base_config: Dict,
                                  parameters: List[str],
                                  ranges: List[Tuple[float, float]],
                                  num_samples: int = 10):
        """Perform sensitivity analysis for specified parameters (e.g., permeability, clay content, etc.).
        
        Args:
            base_config: Base configuration dictionary
            parameters: List of parameter names (dot notation)
            ranges: List of (min, max) tuples for each parameter
            num_samples: Number of samples per parameter
        Returns:
            Dictionary of sensitivity results
        """
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
        logger.info("Completed sensitivity analysis for geotechnical parameters")
        
        return results

    def _set_parameter_value(self, config: Dict, parameter: str, value: float):
        """Set parameter value in configuration dictionary (dot notation).
        
        Args:
            config: Configuration dictionary
            parameter: Parameter name (dot notation)
            value: Value to set
        """
        # Split parameter path
        parts = parameter.split('.')
        
        # Navigate to correct location in config
        current = config
        for part in parts[:-1]:
            current = current[part]
        
        # Set value
        current[parts[-1]] = value

    def _run_simulation(self, config: Dict) -> Dict:
        """Run simulation with given configuration (placeholder for integration).
        
        Args:
            config: Configuration dictionary
        Returns:
            Dictionary of simulation results
        """
        # This method should be implemented to integrate with the simulation framework
        # For now, it's a placeholder
        raise NotImplementedError("Simulation integration not implemented")

    def plot_validation_results(self):
        """Plot validation results (erosion, pressure, velocity, etc.).
        
        Uses geotechnical parameters for labeling and interpretation.
        """
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

    def calibrate_parameter(self, param_path: str, bounds: tuple, simulation_func, exp_data_key='erosion_rate'):
        """Calibrate a model parameter to minimize error with experimental data.
        Args:
            param_path (str): Dot notation path to parameter (e.g., 'dem.bond_strength')
            bounds (tuple): (min, max) bounds for the parameter
            simulation_func (callable): Function that takes config and returns simulation results dict
            exp_data_key (str): Key for experimental data to compare (default: 'erosion_rate')
        Returns:
            dict: Calibration result with optimal parameter and error
        """
        exp_data = self.validation_data['experimental'][exp_data_key]
        config = self.config.copy()

        def objective(x):
            # Set parameter in config
            self._set_parameter_value(config, param_path, x[0])
            # Run simulation
            sim_results = simulation_func(config)
            sim_data = sim_results['erosion_stats']
            # Interpolate if needed
            if sim_data.shape != exp_data.shape:
                from scipy.interpolate import interp1d
                sim_time = np.linspace(0, 0.5, sim_data.shape[0])
                exp_time = np.linspace(0, 0.5, exp_data.shape[0])
                interp = interp1d(sim_time, sim_data)
                sim_data = interp(exp_time)
            # Compute mean squared error
            return np.mean((exp_data - sim_data) ** 2)

        result = minimize(objective, x0=[np.mean(bounds)], bounds=[bounds], method='L-BFGS-B')
        best_param = result.x[0]
        best_error = result.fun
        return {'best_param': best_param, 'best_error': best_error, 'success': result.success}

    def validate_erosion_rate(self, simulation_results: Dict,
                            experimental_data: Dict) -> Dict:
        """Validate erosion rate against experimental data."""
        # Extract data
        sim_time = np.array(simulation_results['time'])
        sim_eroded = np.array(simulation_results['eroded_particles'])
        exp_time = np.array(experimental_data['time'])
        exp_eroded = np.array(experimental_data['eroded_particles'])
        
        # Interpolate simulation results to match experimental time points
        sim_eroded_interp = np.interp(exp_time, sim_time, sim_eroded)
        
        # Compute error metrics
        mse = np.mean((sim_eroded_interp - exp_eroded)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(sim_eroded_interp - exp_eroded))
        r2 = stats.pearsonr(sim_eroded_interp, exp_eroded)[0]**2
        
        # Perform statistical test
        t_stat, p_value = stats.ttest_ind(sim_eroded_interp, exp_eroded)
        
        # Create validation plot
        plt.figure(figsize=(10, 6))
        plt.plot(sim_time, sim_eroded, 'b-', label='Simulation')
        plt.plot(exp_time, exp_eroded, 'r--', label='Experimental')
        plt.xlabel('Time (s)')
        plt.ylabel('Number of Eroded Particles')
        plt.title('Erosion Rate Validation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'erosion_rate_validation.png')
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_valid': p_value > (1 - self.config['validation']['confidence_level'])
        }
    
    def validate_fluid_forces(self, simulation_results: Dict,
                            analytical_solution: Dict) -> Dict:
        """Validate fluid forces against analytical solution."""
        # Extract data
        sim_forces = np.array(simulation_results['fluid_forces'])
        ana_forces = np.array(analytical_solution['forces'])
        
        # Compute error metrics
        force_magnitudes_sim = np.linalg.norm(sim_forces, axis=-1)
        force_magnitudes_ana = np.linalg.norm(ana_forces, axis=-1)
        
        mse = np.mean((force_magnitudes_sim - force_magnitudes_ana)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(force_magnitudes_sim - force_magnitudes_ana))
        r2 = stats.pearsonr(force_magnitudes_sim, force_magnitudes_ana)[0]**2
        
        # Create validation plot
        plt.figure(figsize=(10, 6))
        plt.hist(force_magnitudes_sim, bins=50, alpha=0.5, label='Simulation')
        plt.hist(force_magnitudes_ana, bins=50, alpha=0.5, label='Analytical')
        plt.xlabel('Force Magnitude (N)')
        plt.ylabel('Frequency')
        plt.title('Fluid Force Distribution Validation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'fluid_force_validation.png')
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'is_valid': rmse < self.config['validation']['tolerance']
        }
    
    def validate_bond_degradation(self, simulation_results: Dict,
                                experimental_data: Dict) -> Dict:
        """Validate bond degradation against experimental data."""
        # Extract data
        sim_time = np.array(simulation_results['time'])
        sim_health = np.array(simulation_results['bond_health'])
        exp_time = np.array(experimental_data['time'])
        exp_health = np.array(experimental_data['bond_health'])
        
        # Interpolate simulation results to match experimental time points
        sim_health_interp = np.interp(exp_time, sim_time, sim_health)
        
        # Compute error metrics
        mse = np.mean((sim_health_interp - exp_health)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(sim_health_interp - exp_health))
        r2 = stats.pearsonr(sim_health_interp, exp_health)[0]**2
        
        # Create validation plot
        plt.figure(figsize=(10, 6))
        plt.plot(sim_time, sim_health, 'b-', label='Simulation')
        plt.plot(exp_time, exp_health, 'r--', label='Experimental')
        plt.xlabel('Time (s)')
        plt.ylabel('Bond Health')
        plt.title('Bond Degradation Validation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'bond_degradation_validation.png')
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'is_valid': rmse < self.config['validation']['tolerance']
        }
    
    def validate_conservation_laws(self, simulation_results: Dict) -> Dict:
        """Validate conservation of mass, momentum, and energy."""
        # Extract data
        time = np.array(simulation_results['time'])
        positions = np.array(simulation_results['particle_positions'])
        velocities = np.array(simulation_results['particle_velocities'])
        forces = np.array(simulation_results['fluid_forces'])
        
        # Compute total mass (assuming constant particle mass)
        total_mass = np.sum(self.config['dem']['particle_mass'])
        
        # Compute total momentum
        momentum = np.sum(velocities * self.config['dem']['particle_mass'][:, np.newaxis], axis=0)
        
        # Compute total energy (kinetic + potential)
        kinetic_energy = 0.5 * np.sum(self.config['dem']['particle_mass'] * 
                                    np.linalg.norm(velocities, axis=-1)**2)
        potential_energy = np.sum(self.config['dem']['particle_mass'] * 
                                self.config['simulation']['gravity'][2] * 
                                positions[:, 2])
        total_energy = kinetic_energy + potential_energy
        
        # Compute conservation errors
        mass_error = np.abs(total_mass - self.config['dem']['initial_mass'])
        momentum_error = np.linalg.norm(momentum - self.config['dem']['initial_momentum'])
        energy_error = np.abs(total_energy - self.config['dem']['initial_energy'])
        
        # Create validation plots
        plt.figure(figsize=(15, 5))
        
        # Mass conservation
        plt.subplot(131)
        plt.plot(time, np.ones_like(time) * total_mass, 'b-')
        plt.axhline(y=self.config['dem']['initial_mass'], color='r', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Total Mass (kg)')
        plt.title('Mass Conservation')
        plt.grid(True, alpha=0.3)
        
        # Momentum conservation
        plt.subplot(132)
        plt.plot(time, np.linalg.norm(momentum, axis=-1), 'b-')
        plt.axhline(y=np.linalg.norm(self.config['dem']['initial_momentum']), 
                   color='r', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Total Momentum (kg⋅m/s)')
        plt.title('Momentum Conservation')
        plt.grid(True, alpha=0.3)
        
        # Energy conservation
        plt.subplot(133)
        plt.plot(time, total_energy, 'b-')
        plt.axhline(y=self.config['dem']['initial_energy'], color='r', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Total Energy (J)')
        plt.title('Energy Conservation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'conservation_laws_validation.png')
        plt.close()
        
        return {
            'mass_error': mass_error,
            'momentum_error': momentum_error,
            'energy_error': energy_error,
            'is_valid': (mass_error < self.config['validation']['tolerance'] and
                        momentum_error < self.config['validation']['tolerance'] and
                        energy_error < self.config['validation']['tolerance'])
        }
    
    def generate_validation_report(self, validation_results: Dict):
        """Generate a comprehensive validation report."""
        report = {
            'timestamp': str(np.datetime64('now')),
            'validation_parameters': self.config['validation'],
            'results': validation_results
        }
        
        # Save report
        with open(self.output_dir / 'validation_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # Generate summary plot
        plt.figure(figsize=(10, 6))
        
        metrics = ['mse', 'rmse', 'mae', 'r2']
        categories = ['erosion_rate', 'fluid_forces', 'bond_degradation']
        
        x = np.arange(len(categories))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [validation_results[cat][metric] for cat in categories]
            plt.bar(x + i*width, values, width, label=metric.upper())
        
        plt.xlabel('Validation Category')
        plt.ylabel('Error Metric')
        plt.title('Validation Summary')
        plt.xticks(x + width, categories)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / 'validation_summary.png')
        plt.close()
        
        logger.info("Validation report generated successfully") 