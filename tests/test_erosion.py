"""
Basic import and instantiation tests for DEM-based Internal Erosion Model modules.
"""
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    from src.dem.particle import Particle
    from src.dem.bonds import Bond
    from src.dem.coarse_grained import CoarseGrainedModel
    from src.cfd.fluid import Fluid
    from src.cfd.coupling import Coupling
    from src.erosion.constitutive.parallel_bond import ParallelBondModel
    from src.erosion.constitutive.degradation import BondDegradationModel
    from src.erosion.criteria.grading import GradingCriterion
    from src.erosion.criteria.hydraulic import HydraulicCriterion
    from src.erosion.criteria.stress import StressCriterion
    from src.erosion.process.suffosion import SuffosionProcess
    from src.erosion.process.suffusion import SuffusionProcess
    from src.validation.experimental import ExperimentalValidation
    from src.validation.case_studies import CaseStudyValidation

    assert Particle() is not None
    assert Bond() is not None
    assert CoarseGrainedModel() is not None
    assert Fluid() is not None
    assert Coupling() is not None
    assert ParallelBondModel() is not None
    assert BondDegradationModel() is not None
    assert GradingCriterion() is not None
    assert HydraulicCriterion() is not None
    assert StressCriterion() is not None
    assert SuffosionProcess() is not None
    assert SuffusionProcess() is not None
    assert ExperimentalValidation() is not None
    assert CaseStudyValidation() is not None
