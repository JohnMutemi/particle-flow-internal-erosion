import numpy as np
import yaml
from pathlib import Path

class TriaxialTestValidator:
    def __init__(self, config_path):
        """Initialize the triaxial test validator.
        
        Args:
            config_path (str): Path to the test parameters YAML file
        """
        self.config = self._load_config(config_path)
        self.sample_volume = self._calculate_sample_volume()
        
    def _load_config(self, config_path):
        """Load test parameters from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _calculate_sample_volume(self):
        """Calculate sample volume in mm³."""
        d = self.config['sample']['diameter']
        h = self.config['sample']['height']
        return np.pi * (d/2)**2 * h
    
    def validate_cohesion(self, simulated_cohesion):
        """Validate the simulated cohesion against experimental data.
        
        Args:
            simulated_cohesion (float): Cohesion from simulation in kPa
            
        Returns:
            tuple: (is_valid, error_percentage)
        """
        target = self.config['material']['cohesion']
        tolerance = self.config['validation']['tolerance']
        
        error = abs(simulated_cohesion - target) / target
        is_valid = error <= tolerance
        
        return is_valid, error * 100
    
    def validate_friction_angle(self, simulated_friction):
        """Validate the simulated friction angle against experimental data.
        
        Args:
            simulated_friction (float): Friction angle from simulation in degrees
            
        Returns:
            tuple: (is_valid, error_percentage)
        """
        target = self.config['material']['friction_angle']
        tolerance = self.config['validation']['tolerance']
        
        error = abs(simulated_friction - target) / target
        is_valid = error <= tolerance
        
        return is_valid, error * 100
    
    def validate_particle_size_distribution(self, particle_sizes):
        """Validate that all particles are within the specified size range.
        
        Args:
            particle_sizes (numpy.ndarray): Array of particle diameters in mm
            
        Returns:
            tuple: (is_valid, max_size, min_size)
        """
        max_allowed = self.config['particles']['max_diameter']
        min_allowed = self.config['particles']['min_diameter']
        
        max_size = np.max(particle_sizes)
        min_size = np.min(particle_sizes)
        
        is_valid = (max_size <= max_allowed) and (min_size >= min_allowed)
        
        return is_valid, max_size, min_size
    
    def generate_validation_report(self, simulation_results):
        """Generate a comprehensive validation report.
        
        Args:
            simulation_results (dict): Dictionary containing simulation results
                with keys: 'cohesion', 'friction_angle', 'particle_sizes'
                
        Returns:
            dict: Validation report
        """
        cohesion_valid, cohesion_error = self.validate_cohesion(
            simulation_results['cohesion']
        )
        
        friction_valid, friction_error = self.validate_friction_angle(
            simulation_results['friction_angle']
        )
        
        size_valid, max_size, min_size = self.validate_particle_size_distribution(
            simulation_results['particle_sizes']
        )
        
        return {
            'cohesion_validation': {
                'is_valid': cohesion_valid,
                'error_percentage': cohesion_error,
                'target': self.config['material']['cohesion'],
                'simulated': simulation_results['cohesion']
            },
            'friction_angle_validation': {
                'is_valid': friction_valid,
                'error_percentage': friction_error,
                'target': self.config['material']['friction_angle'],
                'simulated': simulation_results['friction_angle']
            },
            'particle_size_validation': {
                'is_valid': size_valid,
                'max_size': max_size,
                'min_size': min_size,
                'max_allowed': self.config['particles']['max_diameter'],
                'min_allowed': self.config['particles']['min_diameter']
            },
            'overall_validation': {
                'is_valid': all([cohesion_valid, friction_valid, size_valid]),
                'sample_volume': self.sample_volume
            }
        }

def main():
    # Example usage
    config_path = Path('data/input/triaxial_test/test_parameters.yaml')
    validator = TriaxialTestValidator(config_path)
    
    # Example simulation results
    simulation_results = {
        'cohesion': 17.0,  # kPa
        'friction_angle': 33.5,  # degrees
        'particle_sizes': np.random.uniform(0.5, 2.0, 1000)  # mm
    }
    
    # Generate validation report
    report = validator.generate_validation_report(simulation_results)
    
    # Print report
    print("\nValidation Report:")
    print("-----------------")
    print(f"Cohesion: {report['cohesion_validation']['is_valid']}")
    print(f"Error: {report['cohesion_validation']['error_percentage']:.2f}%")
    print(f"\nFriction Angle: {report['friction_angle_validation']['is_valid']}")
    print(f"Error: {report['friction_angle_validation']['error_percentage']:.2f}%")
    print(f"\nParticle Sizes: {report['particle_size_validation']['is_valid']}")
    print(f"Max Size: {report['particle_size_validation']['max_size']:.2f} mm")
    print(f"Min Size: {report['particle_size_validation']['min_size']:.2f} mm")
    print(f"\nOverall Validation: {report['overall_validation']['is_valid']}")

if __name__ == "__main__":
    main() 