"""
Experiment manager module for CFD-DEM simulation validation.
This module handles experimental data collection, processing, and analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

logger = logging.getLogger(__name__)

class ExperimentManager:
    def __init__(self, config: Dict):
        """Initialize the experiment manager."""
        self.config = config
        self.experimental_data = {}
        self.analysis_results = {}
        
        # Create output directory for experimental results
        self.output_dir = Path(config['output']['directory']) / 'experimental'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data collection parameters
        self.sampling_rate = config['experimental'].get('sampling_rate', 100)  # Hz
        self.measurement_points = self._setup_measurement_points()
        
        logger.info("Experiment manager initialized")

    def _setup_measurement_points(self) -> List[Dict]:
        """Set up measurement points for data collection."""
        points = []
        
        # Get tunnel dimensions
        tunnel_length = self.config['case_study']['tunnel_length']
        tunnel_diameter = self.config['case_study']['tunnel_diameter']
        
        # Set up pressure measurement points
        pressure_points = self.config['experimental'].get('pressure_points', 5)
        for i in range(pressure_points):
            x = tunnel_length * (i + 1) / (pressure_points + 1)
            points.append({
                'type': 'pressure',
                'position': np.array([x, 0.0, 0.0]),
                'sensor_id': f'P{i+1}'
            })
        
        # Set up velocity measurement points
        velocity_points = self.config['experimental'].get('velocity_points', 3)
        for i in range(velocity_points):
            x = tunnel_length * (i + 1) / (velocity_points + 1)
            points.append({
                'type': 'velocity',
                'position': np.array([x, 0.0, 0.0]),
                'sensor_id': f'V{i+1}'
            })
        
        # Set up erosion measurement points
        erosion_points = self.config['experimental'].get('erosion_points', 4)
        for i in range(erosion_points):
            x = tunnel_length * (i + 1) / (erosion_points + 1)
            points.append({
                'type': 'erosion',
                'position': np.array([x, 0.0, 0.0]),
                'sensor_id': f'E{i+1}'
            })
        
        return points

    def collect_data(self, duration: float) -> Dict:
        """Collect experimental data for specified duration."""
        logger.info(f"Starting data collection for {duration} seconds")
        
        # Initialize data storage
        num_samples = int(duration * self.sampling_rate)
        data = {
            'time': np.linspace(0, duration, num_samples),
            'pressure': {},
            'velocity': {},
            'erosion': {}
        }
        
        # Collect data from each measurement point
        for point in self.measurement_points:
            sensor_id = point['sensor_id']
            if point['type'] == 'pressure':
                data['pressure'][sensor_id] = self._collect_pressure_data(
                    point['position'], num_samples
                )
            elif point['type'] == 'velocity':
                data['velocity'][sensor_id] = self._collect_velocity_data(
                    point['position'], num_samples
                )
            elif point['type'] == 'erosion':
                data['erosion'][sensor_id] = self._collect_erosion_data(
                    point['position'], num_samples
                )
        
        self.experimental_data = data
        logger.info("Data collection completed")
        
        return data

    def _collect_pressure_data(self, position: np.ndarray, num_samples: int) -> np.ndarray:
        """Collect pressure data from sensor at specified position."""
        # This is a placeholder for actual sensor data collection
        # In practice, this would interface with pressure sensors
        base_pressure = self.config['case_study']['water_pressure']
        noise = np.random.normal(0, base_pressure * 0.01, num_samples)
        return base_pressure + noise

    def _collect_velocity_data(self, position: np.ndarray, num_samples: int) -> np.ndarray:
        """Collect velocity data from sensor at specified position."""
        # This is a placeholder for actual sensor data collection
        # In practice, this would interface with velocity sensors
        base_velocity = 1.0  # m/s
        noise = np.random.normal(0, base_velocity * 0.05, num_samples)
        return base_velocity + noise

    def _collect_erosion_data(self, position: np.ndarray, num_samples: int) -> np.ndarray:
        """Collect erosion data from sensor at specified position."""
        # This is a placeholder for actual sensor data collection
        # In practice, this would interface with erosion sensors
        erosion_rate = 0.001  # m/s
        noise = np.random.normal(0, erosion_rate * 0.1, num_samples)
        return erosion_rate + noise

    def analyze_data(self) -> Dict:
        """Analyze collected experimental data."""
        if not self.experimental_data:
            raise ValueError("No experimental data available for analysis")
        
        analysis = {}
        
        # Analyze pressure data
        pressure_data = self.experimental_data['pressure']
        analysis['pressure'] = {
            'mean': {sensor: np.mean(data) for sensor, data in pressure_data.items()},
            'std': {sensor: np.std(data) for sensor, data in pressure_data.items()},
            'max': {sensor: np.max(data) for sensor, data in pressure_data.items()},
            'min': {sensor: np.min(data) for sensor, data in pressure_data.items()}
        }
        
        # Analyze velocity data
        velocity_data = self.experimental_data['velocity']
        analysis['velocity'] = {
            'mean': {sensor: np.mean(data) for sensor, data in velocity_data.items()},
            'std': {sensor: np.std(data) for sensor, data in velocity_data.items()},
            'max': {sensor: np.max(data) for sensor, data in velocity_data.items()},
            'min': {sensor: np.min(data) for sensor, data in velocity_data.items()}
        }
        
        # Analyze erosion data
        erosion_data = self.experimental_data['erosion']
        analysis['erosion'] = {
            'mean': {sensor: np.mean(data) for sensor, data in erosion_data.items()},
            'std': {sensor: np.std(data) for sensor, data in erosion_data.items()},
            'max': {sensor: np.max(data) for sensor, data in erosion_data.items()},
            'min': {sensor: np.min(data) for sensor, data in erosion_data.items()}
        }
        
        # Compute correlations between measurements
        analysis['correlations'] = self._compute_correlations()
        
        self.analysis_results = analysis
        logger.info("Data analysis completed")
        
        return analysis

    def _compute_correlations(self) -> Dict:
        """Compute correlations between different measurements."""
        correlations = {}
        
        # Compute pressure-velocity correlations
        for p_sensor, p_data in self.experimental_data['pressure'].items():
            for v_sensor, v_data in self.experimental_data['velocity'].items():
                key = f'pressure_{p_sensor}_velocity_{v_sensor}'
                correlations[key] = np.corrcoef(p_data, v_data)[0, 1]
        
        # Compute pressure-erosion correlations
        for p_sensor, p_data in self.experimental_data['pressure'].items():
            for e_sensor, e_data in self.experimental_data['erosion'].items():
                key = f'pressure_{p_sensor}_erosion_{e_sensor}'
                correlations[key] = np.corrcoef(p_data, e_data)[0, 1]
        
        # Compute velocity-erosion correlations
        for v_sensor, v_data in self.experimental_data['velocity'].items():
            for e_sensor, e_data in self.experimental_data['erosion'].items():
                key = f'velocity_{v_sensor}_erosion_{e_sensor}'
                correlations[key] = np.corrcoef(v_data, e_data)[0, 1]
        
        return correlations

    def plot_results(self):
        """Plot experimental results."""
        if not self.experimental_data:
            logger.warning("No experimental data to plot")
            return
        
        # Create figure for time series
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot pressure time series
        for sensor, data in self.experimental_data['pressure'].items():
            axes[0].plot(self.experimental_data['time'], data, label=sensor)
        axes[0].set_title('Pressure Measurements')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Pressure (Pa)')
        axes[0].legend()
        
        # Plot velocity time series
        for sensor, data in self.experimental_data['velocity'].items():
            axes[1].plot(self.experimental_data['time'], data, label=sensor)
        axes[1].set_title('Velocity Measurements')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Velocity (m/s)')
        axes[1].legend()
        
        # Plot erosion time series
        for sensor, data in self.experimental_data['erosion'].items():
            axes[2].plot(self.experimental_data['time'], data, label=sensor)
        axes[2].set_title('Erosion Measurements')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Erosion Rate (m/s)')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'experimental_results.png')
        plt.close()
        
        # Create correlation heatmap
        if self.analysis_results:
            correlations = self.analysis_results['correlations']
            corr_matrix = pd.DataFrame.from_dict(correlations, orient='index')
            plt.figure(figsize=(10, 8))
            plt.imshow(corr_matrix, cmap='coolwarm')
            plt.colorbar()
            plt.title('Measurement Correlations')
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
            plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_heatmap.png')
            plt.close()
        
        logger.info("Experimental results plotted and saved")

    def save_results(self):
        """Save experimental results to file."""
        if not self.experimental_data:
            logger.warning("No experimental data to save")
            return
        
        # Save raw data
        np.savez(
            self.output_dir / 'experimental_data.npz',
            time=self.experimental_data['time'],
            pressure=self.experimental_data['pressure'],
            velocity=self.experimental_data['velocity'],
            erosion=self.experimental_data['erosion']
        )
        
        # Save analysis results
        if self.analysis_results:
            with open(self.output_dir / 'analysis_results.json', 'w') as f:
                json.dump(self.analysis_results, f, indent=4)
        
        logger.info(f"Experimental results saved to {self.output_dir}")

    def compare_with_simulation(self, simulation_results: Dict) -> Dict:
        """Compare experimental results with simulation results."""
        if not self.experimental_data:
            raise ValueError("No experimental data available for comparison")
        
        comparison = {}
        
        # Compare pressure data
        for sensor, exp_data in self.experimental_data['pressure'].items():
            sim_data = simulation_results['fluid_data']['pressure_field']
            comparison[f'pressure_{sensor}'] = {
                'mean_error': np.mean(np.abs(exp_data - sim_data)),
                'max_error': np.max(np.abs(exp_data - sim_data)),
                'correlation': np.corrcoef(exp_data, sim_data)[0, 1],
                'rms_error': np.sqrt(np.mean((exp_data - sim_data)**2))
            }
        
        # Compare velocity data
        for sensor, exp_data in self.experimental_data['velocity'].items():
            sim_data = simulation_results['fluid_data']['velocity_field']
            comparison[f'velocity_{sensor}'] = {
                'mean_error': np.mean(np.abs(exp_data - sim_data)),
                'max_error': np.max(np.abs(exp_data - sim_data)),
                'correlation': np.corrcoef(exp_data, sim_data)[0, 1],
                'rms_error': np.sqrt(np.mean((exp_data - sim_data)**2))
            }
        
        # Compare erosion data
        for sensor, exp_data in self.experimental_data['erosion'].items():
            sim_data = simulation_results['erosion_stats']
            comparison[f'erosion_{sensor}'] = {
                'mean_error': np.mean(np.abs(exp_data - sim_data)),
                'max_error': np.max(np.abs(exp_data - sim_data)),
                'correlation': np.corrcoef(exp_data, sim_data)[0, 1],
                'rms_error': np.sqrt(np.mean((exp_data - sim_data)**2))
            }
        
        logger.info("Comparison with simulation completed")
        
        return comparison 