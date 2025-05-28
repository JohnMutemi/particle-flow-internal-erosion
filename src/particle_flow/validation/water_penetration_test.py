"""
Water Penetration Damage Test implementation following Wang Meixia Chapter 4.
This test simulates sudden water penetration in soil samples and analyzes the damage process.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import logging


class WaterPenetrationTest:
    def __init__(self, config_path: str):
        """Initialize the water penetration test.

        Args:
            config_path (str): Path to the test configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self._initialize_test_parameters()

    def _load_config(self, config_path: str) -> Dict:
        """Load test configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_test_parameters(self):
        """Initialize test parameters from configuration."""
        self.sample_params = self.config['sample']
        self.test_conditions = self.config['test_conditions']
        self.material_params = self.config['material']
        self.particle_params = self.config.get('particles', {
            'size_distribution': {
                'mean': 0.002,
                'std': 0.0005,
                'min_size': 0.001,
                'max_size': 0.005
            },
            'number': 10000,
            'density': 2650.0,
            'friction_coefficient': 0.5,
            'restitution_coefficient': 0.3
        })

        # Calculate derived parameters
        self.sample_volume = np.pi * \
            (self.sample_params['diameter']/2)**2 * \
            self.sample_params['height']
        self.initial_porosity = self.material_params.get('porosity', 0.3)
        self.initial_water_content = self.material_params.get(
            'water_content', 0.15)

        # Advanced saturation model selection
        self.saturation_model = self.material_params.get('saturation_model', 'linear')

        # Initialize saturation state with enhanced tracking
        self.saturation_state = {
            'current_water_content': self.initial_water_content,
            'saturation_degree': 0.0,
            'is_saturated': False,
            'saturation_history': [],
            'pressure_history': [],
            'flow_history': [],
            'erosion_history': []
        }

    def prepare_sample(self) -> Dict:
        """Prepare the sample according to Wang Meixia Chapter 4 procedures.

        This includes:
        1. Initial water content control
        2. Sample saturation process with advanced models
        3. Initial pressure application
        4. Pre-test stability check

        Returns:
            Dict: Initial sample state
        """
        self.logger.info("Preparing sample for water penetration test")

        # Step 1: Initial water content control with enhanced monitoring
        target_water_content = self.test_conditions.get(
            'target_water_content', 0.15)
        if abs(self.initial_water_content - target_water_content) > 0.01:
            self.logger.warning(f"Initial water content ({self.initial_water_content:.3f}) "
                                f"differs from target ({target_water_content:.3f})")
            self._adjust_water_content(target_water_content)

        # Step 2: Advanced sample saturation process
        saturation_pressure = self.test_conditions.get(
            'saturation_pressure', 50e3)  # 50 kPa
        saturation_time = self.test_conditions.get(
            'saturation_time', 3600)  # 1 hour

        # Enhanced saturation process with detailed monitoring
        saturation_results = self._enhanced_saturation_process(
            saturation_pressure, saturation_time)

        # Step 3: Initial pressure application with stability check
        initial_pressure = self.test_conditions['initial_water_pressure']
        stability_check = self._apply_initial_pressure_with_stability(
            initial_pressure)

        # Step 4: Pre-test stability verification
        if not stability_check['is_stable']:
            self.logger.warning("Sample stability check failed: " +
                                stability_check['reason'])

        return {
            'water_content': self.saturation_state['current_water_content'],
            'saturation_degree': self.saturation_state['saturation_degree'],
            'initial_pressure': initial_pressure,
            'is_ready': self.saturation_state['is_saturated'] and stability_check['is_stable'],
            'stability_check': stability_check,
            'saturation_results': saturation_results
        }

    def _adjust_water_content(self, target_content: float):
        """Adjust sample water content to target value."""
        current_content = self.saturation_state['current_water_content']
        adjustment_rate = self.material_params.get(
            'water_adjustment_rate', 0.01)

        while abs(current_content - target_content) > 0.001:
            if current_content < target_content:
                current_content += adjustment_rate
            else:
                current_content -= adjustment_rate
            self.saturation_state['current_water_content'] = current_content

    def _enhanced_saturation_process(self, pressure: float, duration: float) -> Dict:
        """Enhanced sample saturation process with detailed monitoring.

        Args:
            pressure (float): Saturation pressure in Pa
            duration (float): Saturation duration in seconds

        Returns:
            Dict: Detailed saturation results
        """
        self.logger.info(
            f"Starting enhanced sample saturation at {pressure/1000:.1f} kPa")

        # Enhanced saturation parameters
        max_water_content = self.material_params.get('max_water_content', 0.25)
        saturation_rate = self._calculate_saturation_rate(pressure)
        time_steps = np.linspace(0, duration, 100)

        saturation_results = {
            'time': time_steps,
            'water_content': [],
            'saturation_degree': [],
            'pressure': [],
            'flow_rate': [],
            'stability_index': []
        }

        for t in time_steps:
            # Calculate enhanced saturation progress
            progress = self._calculate_saturation_progress(
                t, duration, pressure)

            # Update water content with enhanced model
            target_content = self._calculate_target_water_content(
                progress, max_water_content)
            self.saturation_state['current_water_content'] = target_content

            # Update saturation degree with enhanced model
            saturation_degree = self._calculate_saturation_degree(
                target_content, pressure)
            self.saturation_state['saturation_degree'] = saturation_degree

            # Calculate and store stability index
            stability = self._calculate_stability_index(
                target_content, pressure)

            # Store history
            self.saturation_state['saturation_history'].append(
                saturation_degree)
            self.saturation_state['pressure_history'].append(pressure)

            # Update results
            saturation_results['water_content'].append(target_content)
            saturation_results['saturation_degree'].append(saturation_degree)
            saturation_results['pressure'].append(pressure)
            saturation_results['stability_index'].append(stability)

            # Check for saturation completion
            if saturation_degree >= 0.95 and stability > 0.8:
                self.saturation_state['is_saturated'] = True
                break

        return saturation_results

    def _calculate_saturation_rate(self, pressure: float) -> float:
        """Calculate saturation rate based on pressure and material properties."""
        base_rate = self.material_params.get('saturation_rate', 1e-4)
        pressure_factor = 1 + (pressure / 1e6)  # Pressure effect
        porosity_factor = self.initial_porosity / 0.3  # Porosity effect
        return base_rate * pressure_factor * porosity_factor

    def _calculate_saturation_progress(self, time: float, duration: float, pressure: float) -> float:
        """Calculate saturation progress with enhanced model."""
        base_progress = min(1.0, time / duration)
        pressure_factor = 1 + (pressure / 1e6) * 0.1
        return base_progress * pressure_factor

    def _calculate_target_water_content(self, progress: float, max_content: float) -> float:
        """Calculate target water content with enhanced model."""
        return self.initial_water_content + (max_content - self.initial_water_content) * progress

    def _calculate_saturation_degree(self, water_content: float, pressure: float) -> float:
        """Calculate saturation degree with support for advanced models."""
        max_content = self.material_params.get('max_water_content', 0.25)
        if self.saturation_model == 'van_genuchten':
            # van Genuchten model: S = [1 + (alpha * |psi|)^n]^(-m)
            alpha = self.material_params.get('alpha', 0.08)
            n = self.material_params.get('n', 1.6)
            m = self.material_params.get('m', 1 - 1/n)
            psi = -pressure / 1000.0  # Convert Pa to kPa and negative for suction
            S = (1 + (alpha * abs(psi))**n) ** (-m)
            return S
        else:
            # Default linear model
            pressure_factor = 1 + (pressure / 1e6) * 0.05
            return (water_content / max_content) * pressure_factor

    def _calculate_stability_index(self, water_content: float, pressure: float) -> float:
        """Calculate stability index for saturation process."""
        content_stability = 1 - \
            abs(water_content - self.initial_water_content) / \
            self.initial_water_content
        pressure_stability = 1 - (pressure / 1e6) * 0.1
        return (content_stability + pressure_stability) / 2

    def _apply_initial_pressure_with_stability(self, pressure: float) -> Dict:
        """Apply initial pressure with enhanced stability check."""
        if not self.saturation_state['is_saturated']:
            return {'is_stable': False, 'reason': 'Sample not fully saturated'}

        # Enhanced pressure application parameters
        pressure_ramp_time = self.test_conditions.get('pressure_ramp_time', 60)
        pressure_steps = np.linspace(0, pressure, 20)
        stability_threshold = 0.8

        stability_results = {
            'is_stable': True,
            'reason': '',
            'stability_history': []
        }

        for p in pressure_steps:
            # Update sample state with current pressure
            self._update_sample_state(p)

            # Calculate stability metrics
            displacement_stability = self._calculate_displacement_stability(p)
            erosion_stability = self._calculate_erosion_stability(p)
            overall_stability = (displacement_stability +
                                 erosion_stability) / 2

            stability_results['stability_history'].append(overall_stability)

            # Check for stability issues
            if overall_stability < stability_threshold:
                stability_results['is_stable'] = False
                stability_results['reason'] = f'Stability threshold not met: {overall_stability:.2f}'
                break

            # Check for early failure
            if self._check_early_failure(p):
                stability_results['is_stable'] = False
                stability_results['reason'] = 'Early failure detected'
                break

        return stability_results

    def _calculate_displacement_stability(self, pressure: float) -> float:
        """Calculate displacement stability metric."""
        max_allowed_displacement = self.sample_params['height'] * 0.05
        current_displacement = (
            pressure * self.sample_params['height']) / self.material_params['youngs_modulus']
        return 1 - (current_displacement / max_allowed_displacement)

    def _calculate_erosion_stability(self, pressure: float) -> float:
        """Calculate erosion stability metric."""
        critical_shear_stress = self.material_params['critical_shear_stress']
        current_shear_stress = pressure * self.material_params['permeability']
        return 1 - (current_shear_stress / critical_shear_stress)

    def _update_sample_state(self, pressure: float):
        """Update sample state with enhanced models."""
        # Update material properties based on pressure
        self.material_params['permeability'] *= (1 - pressure / 1e6)
        self.material_params['cohesion'] *= (1 + pressure / 1e6)

        # Update saturation state
        self.saturation_state['pressure_history'].append(pressure)

        # Calculate and store flow rate
        flow_rate = self._calculate_flow_rate(pressure)
        self.saturation_state['flow_history'].append(flow_rate)

        # Calculate and store erosion rate
        erosion_rate = self._calculate_erosion_rate(
            {'initial_pressure': pressure, 'flow_rate': flow_rate})
        self.saturation_state['erosion_history'].append(erosion_rate)

    def _calculate_flow_rate(self, pressure: float) -> float:
        """Calculate flow rate with enhanced model."""
        k = self.material_params['permeability']
        i = pressure / self.sample_params['height']
        A = np.pi * (self.sample_params['diameter']/2)**2
        return k * i * A

    def run_test(self, duration: float, time_step: float) -> Dict:
        """Run the water penetration test with enhanced monitoring.

        Args:
            duration (float): Test duration in seconds
            time_step (float): Time step for simulation

        Returns:
            Dict: Enhanced test results including detailed monitoring data
        """
        self.logger.info("Starting water penetration test")

        # Initialize enhanced results storage
        time_steps = np.arange(0, duration, time_step)
        results = {
            'time': time_steps,
            'pressure': [],
            'flow_rate': [],
            'erosion_rate': [],
            'particle_outflow': [],
            'porosity': [],
            'displacement': [],
            'saturation_degree': [],
            'stability_index': [],
            'shear_stress': [],
            'water_content': []
        }

        # Run test simulation with enhanced monitoring
        current_conditions = self.setup_test_conditions()
        for t in time_steps:
            # Update conditions based on current state
            current_conditions = self._update_conditions(current_conditions, t)

            # Calculate enhanced metrics
            shear_stress = self._calculate_shear_stress(current_conditions)
            stability_index = self._calculate_stability_index(
                self.saturation_state['current_water_content'],
                current_conditions['initial_pressure']
            )

            # Record enhanced results
            results['pressure'].append(current_conditions['initial_pressure'])
            results['flow_rate'].append(current_conditions['flow_rate'])
            results['erosion_rate'].append(
                self._calculate_erosion_rate(current_conditions))
            results['particle_outflow'].append(
                self._calculate_particle_outflow(current_conditions))
            results['porosity'].append(
                self._calculate_porosity(current_conditions))
            results['displacement'].append(
                self._calculate_displacement(current_conditions))
            results['saturation_degree'].append(
                self.saturation_state['saturation_degree'])
            results['stability_index'].append(stability_index)
            results['shear_stress'].append(shear_stress)
            results['water_content'].append(
                self.saturation_state['current_water_content'])

        self.logger.info("Water penetration test completed")
        return results

    def _calculate_shear_stress(self, conditions: Dict) -> float:
        """Calculate shear stress with enhanced model."""
        pressure = conditions['initial_pressure']
        flow_rate = conditions['flow_rate']
        area = np.pi * (self.sample_params['diameter']/2)**2
        return (pressure * flow_rate) / area

    def generate_test_report(self, results: Dict) -> Dict:
        """Generate a comprehensive test report with enhanced analysis.

        Args:
            results (Dict): Test results from run_test()

        Returns:
            Dict: Enhanced test report including detailed analysis
        """
        return {
            'test_parameters': {
                'sample': self.sample_params,
                'material': self.material_params,
                'test_conditions': self.test_conditions
            },
            'results': {
                'max_pressure': max(results['pressure']),
                'max_flow_rate': max(results['flow_rate']),
                'max_erosion_rate': max(results['erosion_rate']),
                'total_particle_outflow': sum(results['particle_outflow']),
                'final_porosity': results['porosity'][-1],
                'max_displacement': max(results['displacement']),
                'max_shear_stress': max(results['shear_stress']),
                'final_stability_index': results['stability_index'][-1],
                'final_saturation_degree': results['saturation_degree'][-1]
            },
            'analysis': {
                'erosion_progression': self._analyze_erosion_progression(results),
                'failure_criteria': self._check_failure_criteria(results),
                'stability_analysis': self._analyze_stability(results),
                'saturation_analysis': self._analyze_saturation(results),
                'recommendations': self._generate_recommendations(results)
            }
        }

    def _analyze_stability(self, results: Dict) -> Dict:
        """Analyze stability throughout the test."""
        stability_index = results['stability_index']
        time = results['time']

        return {
            'min_stability': min(stability_index),
            'avg_stability': np.mean(stability_index),
            'stability_trend': np.polyfit(time, stability_index, 1)[0],
            'critical_points': self._find_critical_stability_points(stability_index, time)
        }

    def _analyze_saturation(self, results: Dict) -> Dict:
        """Analyze saturation behavior throughout the test."""
        saturation_degree = results['saturation_degree']
        water_content = results['water_content']

        return {
            'final_saturation': saturation_degree[-1],
            'saturation_rate': np.mean(np.diff(saturation_degree)),
            'water_content_stability': np.std(water_content),
            'saturation_efficiency': self._calculate_saturation_efficiency(saturation_degree)
        }

    def _find_critical_stability_points(self, stability_index: List[float], time: List[float]) -> List[Dict]:
        """Find critical points in stability history."""
        critical_points = []
        threshold = 0.8

        for i in range(1, len(stability_index)-1):
            if stability_index[i] < threshold and stability_index[i-1] >= threshold:
                critical_points.append({
                    'time': time[i],
                    'stability': stability_index[i],
                    'type': 'instability_start'
                })
            elif stability_index[i] >= threshold and stability_index[i-1] < threshold:
                critical_points.append({
                    'time': time[i],
                    'stability': stability_index[i],
                    'type': 'stability_recovery'
                })

        return critical_points

    def _calculate_saturation_efficiency(self, saturation_degree: List[float]) -> float:
        """Calculate saturation efficiency metric."""
        target_saturation = 0.95
        final_saturation = saturation_degree[-1]
        time_to_saturation = len(saturation_degree)

        if final_saturation >= target_saturation:
            return 1.0
        else:
            return final_saturation / target_saturation

    def setup_test_conditions(self) -> Dict:
        """Set up the test conditions for water penetration.

        Returns:
            Dict: Test conditions including pressure, flow rate, and boundary conditions
        """
        return {
            'initial_pressure': self.test_conditions['initial_water_pressure'],
            'pressure_gradient': self._calculate_pressure_gradient(),
            'flow_rate': self._calculate_initial_flow_rate(),
            'boundary_conditions': self._setup_boundary_conditions()
        }

    def _calculate_pressure_gradient(self) -> np.ndarray:
        """Calculate the pressure gradient for water penetration."""
        # Following Wang Meixia Chapter 4, calculate pressure gradient
        # based on sample height and pressure difference
        height = self.sample_params['height']
        pressure_diff = self.test_conditions['initial_water_pressure']
        return np.array([0.0, -pressure_diff/height, 0.0])

    def _calculate_initial_flow_rate(self) -> float:
        """Calculate initial flow rate based on Darcy's law."""
        k = self.material_params['permeability']
        i = self.test_conditions['initial_water_pressure'] / \
            self.sample_params['height']
        A = np.pi * (self.sample_params['diameter']/2)**2
        return k * i * A

    def _setup_boundary_conditions(self) -> Dict:
        """Set up boundary conditions for the test."""
        return {
            'top': {
                'type': 'pressure',
                'value': self.test_conditions['initial_water_pressure']
            },
            'bottom': {
                'type': 'pressure',
                'value': 0.0
            },
            'sides': {
                'type': 'no_flow'
            }
        }

    def _update_conditions(self, current_conditions: Dict, time: float) -> Dict:
        """Update test conditions based on current state and time.

        Args:
            current_conditions (Dict): Current test conditions
            time (float): Current simulation time

        Returns:
            Dict: Updated test conditions
        """
        # Update pressure based on time
        pressure = current_conditions['initial_pressure'] * np.exp(-0.1 * time)
        # Ensure pressure does not fall below a minimum threshold (e.g., 1 Pa)
        pressure = max(pressure, 1.0)

        # Update flow rate based on new pressure
        flow_rate = self._calculate_initial_flow_rate(
        ) * (pressure / current_conditions['initial_pressure'])

        return {
            'initial_pressure': pressure,
            'flow_rate': flow_rate,
            'pressure_gradient': self._calculate_pressure_gradient(),
            'boundary_conditions': current_conditions['boundary_conditions']
        }

    def _calculate_erosion_rate(self, conditions: Dict) -> float:
        """Calculate erosion rate based on current conditions."""
        # Following Wang Meixia Chapter 4, calculate erosion rate
        # based on flow rate and material properties
        flow_rate = conditions['flow_rate']
        critical_shear_stress = self.material_params.get(
            'critical_shear_stress', 1.0)
        if conditions['initial_pressure'] < 1e-10:
            return 0.0
        return max(0.0, flow_rate * (1 - critical_shear_stress/conditions['initial_pressure']))

    def _calculate_particle_outflow(self, conditions: Dict) -> float:
        """Calculate particle outflow rate."""
        erosion_rate = self._calculate_erosion_rate(conditions)
        return erosion_rate * self.sample_volume * self.initial_porosity

    def _calculate_porosity(self, conditions: Dict) -> float:
        """Calculate current porosity based on erosion."""
        particle_outflow = self._calculate_particle_outflow(conditions)
        return self.initial_porosity + (particle_outflow / self.sample_volume)

    def _calculate_displacement(self, conditions: Dict) -> float:
        """Calculate sample displacement based on current conditions."""
        # Calculate displacement based on pressure and material properties
        pressure = conditions['initial_pressure']
        youngs_modulus = self.material_params.get('youngs_modulus', 1e9)
        return (pressure * self.sample_params['height']) / youngs_modulus

    def _analyze_erosion_progression(self, results: Dict) -> Dict:
        """Analyze the progression of erosion during the test."""
        erosion_rates = results['erosion_rate']
        time = results['time']

        # Find critical points in erosion progression
        max_rate_idx = np.argmax(erosion_rates)
        acceleration = np.gradient(erosion_rates, time)

        return {
            'critical_time': time[max_rate_idx],
            'max_erosion_rate': erosion_rates[max_rate_idx],
            'acceleration_analysis': {
                'max_acceleration': max(acceleration),
                'acceleration_time': time[np.argmax(acceleration)]
            }
        }

    def _check_failure_criteria(self, results: Dict) -> Dict:
        """Check if the test results meet any failure criteria."""
        max_displacement = max(results['displacement'])
        max_erosion_rate = max(results['erosion_rate'])
        final_porosity = results['porosity'][-1]

        # Get critical values from material parameters
        critical_displacement = self.sample_params['height'] * 0.1
        critical_erosion_rate = self.material_params.get(
            'critical_erosion_rate', 1.0)
        critical_porosity = self.material_params.get('critical_porosity', 0.5)
        critical_water_content = self.material_params.get(
            'critical_water_content', 0.2)

        # Check all failure criteria
        failure_criteria = {
            'displacement_failure': max_displacement > critical_displacement,
            'erosion_failure': max_erosion_rate > critical_erosion_rate,
            'porosity_failure': final_porosity > critical_porosity,
            'water_content_failure': self.saturation_state['current_water_content'] > critical_water_content,
            'saturation_failure': not self.saturation_state['is_saturated']
        }

        # Calculate failure severity
        severity = sum(1 for v in failure_criteria.values() if v)
        failure_criteria['severity'] = severity
        failure_criteria['is_critical'] = severity >= 2

        return failure_criteria

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check erosion progression
        erosion_analysis = self._analyze_erosion_progression(results)
        if erosion_analysis['max_erosion_rate'] > 0.5:
            recommendations.append(
                "High erosion rate detected. Consider implementing erosion control measures.")

        # Check displacement
        if max(results['displacement']) > self.sample_params['height'] * 0.05:
            recommendations.append(
                "Significant displacement observed. Review support system design.")

        # Check porosity changes
        if results['porosity'][-1] > self.initial_porosity * 1.5:
            recommendations.append(
                "Large porosity increase detected. Consider grouting or other stabilization measures.")

        return recommendations
