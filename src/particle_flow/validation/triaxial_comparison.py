"""
Module for comparing water penetration test results with triaxial test data.
This module implements the comparison methodology described in Wang Meixia Chapter 4.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import yaml


class TriaxialComparison:
    def __init__(self, config_path: str):
        """Initialize the triaxial comparison module.

        Args:
            config_path (str): Path to the configuration file containing triaxial test data
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self._initialize_comparison_parameters()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _initialize_comparison_parameters(self):
        """Initialize comparison parameters from configuration."""
        self.triaxial_data = self.config.get('triaxial_data', {})
        self.comparison_params = self.config.get('comparison_parameters', {
            'stress_levels': [0.2, 0.4, 0.6, 0.8, 1.0],
            'strain_rates': [1e-6, 1e-5, 1e-4],
            'confining_pressures': [50e3, 100e3, 200e3]  # 50, 100, 200 kPa
        })

    def compare_stress_strain(self, water_penetration_results: Dict) -> Dict:
        """Compare stress-strain behavior between water penetration and triaxial tests.

        Args:
            water_penetration_results (Dict): Results from water penetration test

        Returns:
            Dict: Comparison results including correlation metrics
        """
        self.logger.info("Comparing stress-strain behavior")

        # Extract relevant data
        wp_stress = np.array(water_penetration_results['pressure'])
        sample_height = water_penetration_results.get('test_params', {}).get(
            'sample', {}).get('height', 0.1)  # Default to 0.1m if not found
        wp_strain = np.array(
            water_penetration_results['displacement']) / sample_height

        # Get triaxial test data
        tx_stress = np.array(self.triaxial_data.get('stress', []))
        tx_strain = np.array(self.triaxial_data.get('strain', []))

        # Calculate comparison metrics
        correlation = self._calculate_correlation(wp_stress, tx_stress)
        strain_difference = self._calculate_strain_difference(
            wp_strain, tx_strain)
        strength_ratio = self._calculate_strength_ratio(wp_stress, tx_stress)

        return {
            'correlation_coefficient': correlation,
            'strain_difference': strain_difference,
            'strength_ratio': strength_ratio,
            'comparison_metrics': self._calculate_comparison_metrics(
                wp_stress, wp_strain, tx_stress, tx_strain)
        }

    def compare_erosion_behavior(self, water_penetration_results: Dict) -> Dict:
        """Compare erosion behavior between water penetration and triaxial tests.

        Args:
            water_penetration_results (Dict): Results from water penetration test

        Returns:
            Dict: Comparison results for erosion behavior
        """
        self.logger.info("Comparing erosion behavior")

        # Extract erosion data
        wp_erosion = np.array(water_penetration_results['erosion_rate'])
        wp_porosity = np.array(water_penetration_results['porosity'])

        # Get triaxial test erosion data
        tx_erosion = np.array(self.triaxial_data.get('erosion_rate', []))
        tx_porosity = np.array(self.triaxial_data.get('porosity', []))

        # Calculate erosion comparison metrics
        erosion_correlation = self._calculate_correlation(
            wp_erosion, tx_erosion)
        porosity_correlation = self._calculate_correlation(
            wp_porosity, tx_porosity)

        return {
            'erosion_correlation': erosion_correlation,
            'porosity_correlation': porosity_correlation,
            'erosion_metrics': self._calculate_erosion_metrics(
                wp_erosion, wp_porosity, tx_erosion, tx_porosity)
        }

    def compare_saturation_behavior(self, water_penetration_results: Dict) -> Dict:
        """Compare saturation behavior between water penetration and triaxial tests.

        Args:
            water_penetration_results (Dict): Results from water penetration test

        Returns:
            Dict: Comparison results for saturation behavior
        """
        self.logger.info("Comparing saturation behavior")

        # Extract saturation data
        wp_saturation = np.array(
            water_penetration_results['saturation_degree'])
        wp_water_content = np.array(water_penetration_results['water_content'])

        # Get triaxial test saturation data
        tx_saturation = np.array(
            self.triaxial_data.get('saturation_degree', []))
        tx_water_content = np.array(
            self.triaxial_data.get('water_content', []))

        # Calculate saturation comparison metrics
        saturation_correlation = self._calculate_correlation(
            wp_saturation, tx_saturation)
        water_content_correlation = self._calculate_correlation(
            wp_water_content, tx_water_content)

        return {
            'saturation_correlation': saturation_correlation,
            'water_content_correlation': water_content_correlation,
            'saturation_metrics': self._calculate_saturation_metrics(
                wp_saturation, wp_water_content, tx_saturation, tx_water_content)
        }

    def _calculate_correlation(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate correlation coefficient between two datasets."""
        if len(data1) != len(data2):
            # Resample to match lengths
            data1 = np.interp(np.linspace(0, 1, len(data2)),
                              np.linspace(0, 1, len(data1)), data1)
        return np.corrcoef(data1, data2)[0, 1]

    def _calculate_strain_difference(self, strain1: np.ndarray, strain2: np.ndarray) -> float:
        """Calculate average difference in strain between two tests."""
        if len(strain1) != len(strain2):
            strain1 = np.interp(np.linspace(0, 1, len(strain2)),
                                np.linspace(0, 1, len(strain1)), strain1)
        return np.mean(np.abs(strain1 - strain2))

    def _calculate_strength_ratio(self, stress1: np.ndarray, stress2: np.ndarray) -> float:
        """Calculate ratio of maximum stresses between two tests."""
        return max(stress1) / max(stress2)

    def _calculate_comparison_metrics(self, wp_stress: np.ndarray, wp_strain: np.ndarray,
                                      tx_stress: np.ndarray, tx_strain: np.ndarray) -> Dict:
        """Calculate detailed comparison metrics for stress-strain behavior."""
        # Resample data to match lengths
        if len(wp_stress) != len(tx_stress):
            wp_stress = np.interp(np.linspace(0, 1, len(tx_stress)),
                                  np.linspace(0, 1, len(wp_stress)), wp_stress)
            wp_strain = np.interp(np.linspace(0, 1, len(tx_strain)),
                                  np.linspace(0, 1, len(wp_strain)), wp_strain)

        # Calculate various metrics
        stress_error = np.mean(np.abs(wp_stress - tx_stress))
        strain_error = np.mean(np.abs(wp_strain - tx_strain))
        max_stress_error = np.max(np.abs(wp_stress - tx_stress))
        max_strain_error = np.max(np.abs(wp_strain - tx_strain))

        return {
            'stress_error': stress_error,
            'strain_error': strain_error,
            'max_stress_error': max_stress_error,
            'max_strain_error': max_strain_error,
            'stress_ratio': max(wp_stress) / max(tx_stress),
            'strain_ratio': max(wp_strain) / max(tx_strain)
        }

    def _calculate_erosion_metrics(self, wp_erosion: np.ndarray, wp_porosity: np.ndarray,
                                   tx_erosion: np.ndarray, tx_porosity: np.ndarray) -> Dict:
        """Calculate detailed comparison metrics for erosion behavior."""
        # Resample data to match lengths
        if len(wp_erosion) != len(tx_erosion):
            wp_erosion = np.interp(np.linspace(0, 1, len(tx_erosion)),
                                   np.linspace(0, 1, len(wp_erosion)), wp_erosion)
            wp_porosity = np.interp(np.linspace(0, 1, len(tx_porosity)),
                                    np.linspace(0, 1, len(wp_porosity)), wp_porosity)

        # Calculate erosion metrics
        erosion_error = np.mean(np.abs(wp_erosion - tx_erosion))
        porosity_error = np.mean(np.abs(wp_porosity - tx_porosity))
        max_erosion_error = np.max(np.abs(wp_erosion - tx_erosion))
        max_porosity_error = np.max(np.abs(wp_porosity - tx_porosity))

        return {
            'erosion_error': erosion_error,
            'porosity_error': porosity_error,
            'max_erosion_error': max_erosion_error,
            'max_porosity_error': max_porosity_error,
            'erosion_ratio': max(wp_erosion) / max(tx_erosion),
            'porosity_ratio': max(wp_porosity) / max(tx_porosity)
        }

    def _calculate_saturation_metrics(self, wp_saturation: np.ndarray, wp_water_content: np.ndarray,
                                      tx_saturation: np.ndarray, tx_water_content: np.ndarray) -> Dict:
        """Calculate detailed comparison metrics for saturation behavior."""
        # Resample data to match lengths
        if len(wp_saturation) != len(tx_saturation):
            wp_saturation = np.interp(np.linspace(0, 1, len(tx_saturation)),
                                      np.linspace(0, 1, len(wp_saturation)), wp_saturation)
            wp_water_content = np.interp(np.linspace(0, 1, len(tx_water_content)),
                                         np.linspace(0, 1, len(wp_water_content)), wp_water_content)

        # Calculate saturation metrics
        saturation_error = np.mean(np.abs(wp_saturation - tx_saturation))
        water_content_error = np.mean(
            np.abs(wp_water_content - tx_water_content))
        max_saturation_error = np.max(np.abs(wp_saturation - tx_saturation))
        max_water_content_error = np.max(
            np.abs(wp_water_content - tx_water_content))

        return {
            'saturation_error': saturation_error,
            'water_content_error': water_content_error,
            'max_saturation_error': max_saturation_error,
            'max_water_content_error': max_water_content_error,
            'saturation_ratio': max(wp_saturation) / max(tx_saturation),
            'water_content_ratio': max(wp_water_content) / max(tx_water_content)
        }

    def generate_comparison_report(self, water_penetration_results: Dict) -> Dict:
        """Generate a comprehensive comparison report.

        Args:
            water_penetration_results (Dict): Results from water penetration test

        Returns:
            Dict: Comprehensive comparison report
        """
        self.logger.info("Generating comparison report")

        # Perform all comparisons
        stress_strain_comparison = self.compare_stress_strain(
            water_penetration_results)
        erosion_comparison = self.compare_erosion_behavior(
            water_penetration_results)
        saturation_comparison = self.compare_saturation_behavior(
            water_penetration_results)

        # Calculate overall validation score
        validation_score = self._calculate_validation_score(
            stress_strain_comparison,
            erosion_comparison,
            saturation_comparison
        )

        return {
            'stress_strain_comparison': stress_strain_comparison,
            'erosion_comparison': erosion_comparison,
            'saturation_comparison': saturation_comparison,
            'validation_score': validation_score,
            'recommendations': self._generate_recommendations(
                stress_strain_comparison,
                erosion_comparison,
                saturation_comparison
            )
        }

    def _calculate_validation_score(self, stress_strain_comparison: Dict,
                                    erosion_comparison: Dict,
                                    saturation_comparison: Dict) -> float:
        """Calculate overall validation score based on all comparisons."""
        # Weight factors for different aspects
        weights = {
            'stress_strain': 0.4,
            'erosion': 0.3,
            'saturation': 0.3
        }

        # Calculate individual scores
        stress_strain_score = self._calculate_component_score(
            stress_strain_comparison)
        erosion_score = self._calculate_component_score(erosion_comparison)
        saturation_score = self._calculate_component_score(
            saturation_comparison)

        # Calculate weighted average
        return (weights['stress_strain'] * stress_strain_score +
                weights['erosion'] * erosion_score +
                weights['saturation'] * saturation_score)

    def _calculate_component_score(self, comparison: Dict) -> float:
        """Calculate score for a single comparison component."""
        # Extract relevant metrics
        correlation = comparison.get('correlation_coefficient', 0)
        error = comparison.get('strain_difference', 1)

        # Normalize error to [0, 1] range
        normalized_error = min(error, 1.0)

        # Calculate score (higher correlation and lower error = better score)
        return 0.7 * correlation + 0.3 * (1 - normalized_error)

    def _generate_recommendations(self, stress_strain_comparison: Dict,
                                  erosion_comparison: Dict,
                                  saturation_comparison: Dict) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []

        # Check stress-strain behavior
        if stress_strain_comparison['correlation_coefficient'] < 0.8:
            recommendations.append(
                "Stress-strain behavior shows significant deviation from triaxial test data. "
                "Consider adjusting material parameters or test conditions.")

        # Check erosion behavior
        if erosion_comparison['erosion_correlation'] < 0.8:
            recommendations.append(
                "Erosion behavior differs from triaxial test data. "
                "Review erosion model parameters and boundary conditions.")

        # Check saturation behavior
        if saturation_comparison['saturation_correlation'] < 0.8:
            recommendations.append(
                "Saturation behavior shows discrepancies with triaxial test data. "
                "Verify saturation model and initial conditions.")

        return recommendations
