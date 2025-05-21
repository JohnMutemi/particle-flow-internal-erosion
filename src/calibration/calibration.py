"""
Calibration module for CFD-DEM simulation.
This module implements parameter calibration routines for the coarse-grained model,
including optimization algorithms and sensitivity analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from scipy.optimize import minimize, differential_evolution
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class CalibrationManager:
    def __init__(self, config: Dict):
        """Initialize the calibration manager."""
        self.config = config
        self.calibration_results = {}
        self.optimization_history = []
        
        # Create output directory for calibration results
        self.output_dir = Path(config['output']['directory']) / 'calibration'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load calibration parameters
        self.parameters = self._load_calibration_parameters()
        
        logger.info("Calibration manager initialized")

    def _load_calibration_parameters(self) -> Dict:
        """Load calibration parameters from configuration."""
        params = {}
        
        # Load parameter bounds and initial values
        for param_name, param_config in self.config['calibration']['parameters'].items():
            params[param_name] = {
                'bounds': tuple(param_config['bounds']),
                'initial_value': param_config.get('initial_value'),
                'distribution': param_config.get('distribution', 'uniform'),
                'scale': param_config.get('scale', 1.0)
            }
        
        return params

    def define_objective_function(self, experimental_data: Dict) -> Callable:
        """Define the objective function for calibration."""
        def objective_function(params: np.ndarray) -> float:
            # Convert parameter array to dictionary
            param_dict = self._array_to_param_dict(params)
            
            # Run simulation with current parameters
            simulation_results = self._run_simulation(param_dict)
            
            # Compute error metrics
            error = 0.0
            
            # Compare erosion rates
            if 'erosion_rate' in experimental_data:
                exp_erosion = experimental_data['erosion_rate']
                sim_erosion = simulation_results['erosion_stats']
                error += np.mean((exp_erosion - sim_erosion)**2)
            
            # Compare pressure distributions
            if 'pressure' in experimental_data:
                exp_pressure = experimental_data['pressure']
                sim_pressure = simulation_results['fluid_data']['pressure_field']
                error += np.mean((exp_pressure - sim_pressure)**2)
            
            # Compare velocity profiles
            if 'velocity' in experimental_data:
                exp_velocity = experimental_data['velocity']
                sim_velocity = simulation_results['fluid_data']['velocity_field']
                error += np.mean((exp_velocity - sim_velocity)**2)
            
            return error
        
        return objective_function

    def _array_to_param_dict(self, params: np.ndarray) -> Dict:
        """Convert parameter array to dictionary."""
        param_dict = {}
        for i, (param_name, param_config) in enumerate(self.parameters.items()):
            param_dict[param_name] = params[i] * param_config['scale']
        return param_dict

    def _param_dict_to_array(self, param_dict: Dict) -> np.ndarray:
        """Convert parameter dictionary to array."""
        params = []
        for param_name in self.parameters:
            params.append(param_dict[param_name] / self.parameters[param_name]['scale'])
        return np.array(params)

    def calibrate_parameters(self, experimental_data: Dict,
                           method: str = 'differential_evolution',
                           **kwargs) -> Dict:
        """Calibrate model parameters using specified optimization method."""
        # Define objective function
        objective = self.define_objective_function(experimental_data)
        
        # Get parameter bounds
        bounds = [param['bounds'] for param in self.parameters.values()]
        
        # Get initial values
        x0 = self._param_dict_to_array({
            name: param['initial_value'] or np.mean(param['bounds'])
            for name, param in self.parameters.items()
        })
        
        # Run optimization
        if method == 'differential_evolution':
            result = differential_evolution(
                objective,
                bounds=bounds,
                **kwargs
            )
        elif method == 'minimize':
            result = minimize(
                objective,
                x0=x0,
                bounds=bounds,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Store results
        self.calibration_results = {
            'method': method,
            'success': result.success,
            'message': result.message,
            'optimal_parameters': self._array_to_param_dict(result.x),
            'optimal_value': result.fun,
            'n_iterations': result.nit,
            'n_evaluations': result.nfev
        }
        
        logger.info(f"Calibration completed using {method}")
        logger.info(f"Optimal parameters: {self.calibration_results['optimal_parameters']}")
        logger.info(f"Optimal value: {self.calibration_results['optimal_value']}")
        
        return self.calibration_results

    def perform_bayesian_calibration(self, experimental_data: Dict,
                                   n_samples: int = 1000,
                                   **kwargs) -> Dict:
        """Perform Bayesian calibration of model parameters."""
        # Define likelihood function
        def likelihood(params: np.ndarray) -> float:
            # Convert parameters to dictionary
            param_dict = self._array_to_param_dict(params)
            
            # Run simulation
            simulation_results = self._run_simulation(param_dict)
            
            # Compute log-likelihood
            log_likelihood = 0.0
            
            # Compare erosion rates
            if 'erosion_rate' in experimental_data:
                exp_erosion = experimental_data['erosion_rate']
                sim_erosion = simulation_results['erosion_stats']
                log_likelihood += np.sum(norm.logpdf(
                    exp_erosion,
                    loc=sim_erosion,
                    scale=kwargs.get('noise_scale', 1.0)
                ))
            
            # Compare pressure distributions
            if 'pressure' in experimental_data:
                exp_pressure = experimental_data['pressure']
                sim_pressure = simulation_results['fluid_data']['pressure_field']
                log_likelihood += np.sum(norm.logpdf(
                    exp_pressure,
                    loc=sim_pressure,
                    scale=kwargs.get('noise_scale', 1.0)
                ))
            
            # Compare velocity profiles
            if 'velocity' in experimental_data:
                exp_velocity = experimental_data['velocity']
                sim_velocity = simulation_results['fluid_data']['velocity_field']
                log_likelihood += np.sum(norm.logpdf(
                    exp_velocity,
                    loc=sim_velocity,
                    scale=kwargs.get('noise_scale', 1.0)
                ))
            
            return log_likelihood
        
        # Define prior distributions
        priors = {}
        for param_name, param_config in self.parameters.items():
            if param_config['distribution'] == 'uniform':
                priors[param_name] = uniform(
                    loc=param_config['bounds'][0],
                    scale=param_config['bounds'][1] - param_config['bounds'][0]
                )
            elif param_config['distribution'] == 'normal':
                priors[param_name] = norm(
                    loc=param_config['initial_value'],
                    scale=param_config['scale']
                )
        
        # Perform MCMC sampling
        # Note: This is a simplified implementation
        # In practice, you would use a proper MCMC library like emcee or PyMC3
        samples = []
        current_params = self._param_dict_to_array({
            name: param['initial_value'] or np.mean(param['bounds'])
            for name, param in self.parameters.items()
        })
        current_log_likelihood = likelihood(current_params)
        
        for _ in range(n_samples):
            # Propose new parameters
            proposed_params = current_params + np.random.normal(0, 0.1, size=len(current_params))
            
            # Compute log-likelihood
            proposed_log_likelihood = likelihood(proposed_params)
            
            # Accept/reject
            if proposed_log_likelihood > current_log_likelihood:
                current_params = proposed_params
                current_log_likelihood = proposed_log_likelihood
            
            samples.append(current_params)
        
        # Store results
        self.calibration_results = {
            'method': 'bayesian',
            'samples': samples,
            'posterior_mean': np.mean(samples, axis=0),
            'posterior_std': np.std(samples, axis=0)
        }
        
        logger.info("Bayesian calibration completed")
        
        return self.calibration_results

    def _run_simulation(self, param_dict: Dict) -> Dict:
        """Run simulation with given parameters."""
        # This method should be implemented to integrate with the simulation framework
        # For now, it's a placeholder
        raise NotImplementedError("Simulation integration not implemented")

    def plot_calibration_results(self):
        """Plot calibration results."""
        if not self.calibration_results:
            logger.warning("No calibration results to plot")
            return
        
        # Create figure for parameter distributions
        n_params = len(self.parameters)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 4*n_params))
        
        if n_params == 1:
            axes = [axes]
        
        # Plot parameter distributions
        for i, (param_name, param_config) in enumerate(self.parameters.items()):
            if self.calibration_results['method'] == 'bayesian':
                # Plot posterior distribution
                samples = self.calibration_results['samples']
                axes[i].hist(samples[:, i], bins=30, density=True, alpha=0.5)
                axes[i].axvline(self.calibration_results['posterior_mean'][i],
                              color='r', linestyle='--', label='Posterior Mean')
            else:
                # Plot optimal value
                optimal_value = self.calibration_results['optimal_parameters'][param_name]
                axes[i].axvline(optimal_value, color='r', linestyle='--',
                              label='Optimal Value')
            
            # Plot prior distribution
            x = np.linspace(*param_config['bounds'], 100)
            if param_config['distribution'] == 'uniform':
                y = uniform.pdf(x, *param_config['bounds'])
            elif param_config['distribution'] == 'normal':
                y = norm.pdf(x, param_config['initial_value'], param_config['scale'])
            axes[i].plot(x, y, 'k-', label='Prior')
            
            axes[i].set_title(f'Parameter: {param_name}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration_results.png')
        plt.close()
        
        logger.info("Calibration results plotted and saved")

    def save_calibration_results(self):
        """Save calibration results to file."""
        if not self.calibration_results:
            logger.warning("No calibration results to save")
            return
        
        # Save results as JSON
        results_file = self.output_dir / 'calibration_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.calibration_results, f, indent=4)
        
        logger.info(f"Calibration results saved to {results_file}")

    def get_optimal_parameters(self) -> Dict:
        """Get optimal parameters from calibration results."""
        if not self.calibration_results:
            raise ValueError("No calibration results available")
        
        if self.calibration_results['method'] == 'bayesian':
            return self._array_to_param_dict(self.calibration_results['posterior_mean'])
        else:
            return self.calibration_results['optimal_parameters'] 