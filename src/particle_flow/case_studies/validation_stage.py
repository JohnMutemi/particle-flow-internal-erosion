"""
Validation stage for tunnel water inrush simulation, including triaxial (three-axis) validation.
"""

from particle_flow.validation.triaxial_validation import TriaxialTestValidator


class ValidationStage:
    def __init__(self, config_path):
        self.triaxial_validator = TriaxialTestValidator(config_path)

    def run_validation(self, simulation_results):
        # Run triaxial validation as part of the validation stage
        report = self.triaxial_validator.generate_validation_report(
            simulation_results)
        return report
