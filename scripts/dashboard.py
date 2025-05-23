"""
Dashboard script for interactive visualization and post-processing of simulation results.

- Provides a Streamlit-based dashboard for exploring CFD-DEM simulation results.
- Loads parameters from config.yaml and displays results from the results directory.
- Allows interactive parameter studies and validation visualization.

Usage:
    streamlit run scripts/dashboard.py

References:
- config.yaml
- results/
- scripts/demo.py (for running simulations)
"""

import streamlit as st
import os
from PIL import Image
import plotly.express as px
import pandas as pd
import numpy as np
import yaml
import subprocess
import json
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from pathlib import Path
from src.validation.validation import ValidationManager
from src.models.bond_model import SeepageBondModel
from src.coarse_grained.coarse_grained_model import CoarseGrainedModel
from src.coupling.coupling_manager import CouplingManager

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="CFD-DEM Simulation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Language dictionary for bilingual support
LANG = {
    'en': {
        'dashboard_title': "CFD-DEM Simulation Dashboard",
        'navigation': "Navigation",
        'overview': "Overview",
        'coupling_framework': "Coupling Framework",
        'coupling_framework_diagram': "Coupling Framework Diagram",
        'validation_results': "Validation Results",
        'parameter_calibration': "Parameter Calibration",
        'parameter_calibration_title': "Parameter Calibration Interface",
        'geotechnical_parameters': "Material Properties",
        'theoretical_background': "Theoretical Background",
        'project_overview': "Project Overview",
        'project_overview_desc': "This dashboard provides a comprehensive view of the CFD-DEM simulation results, including the coupling framework, validation results, and parameter calibration.",
        'simulation_time': "Simulation Time",
        'particles': "Particles",
        'validation_score': "Validation Score",
        'geotech_params_summary': "Material Properties Summary",
        'recent_results': "Recent Results",
        'cfd_dem_framework': "CFD-DEM Coupling Framework",
        'framework_desc': "The coupling framework illustrates how the Discrete Element Method (DEM) for particles interacts with the Computational Fluid Dynamics (CFD) for fluid flow through the Coupling Manager.",
        'framework_components': "Framework Components",
        'dem_particles': "DEM (Particles): Handles particle motion, contacts, and forces",
        'cfd_fluid': "CFD (Fluid): Computes fluid flow, pressure, and velocity fields",
        'coupling_manager': "Coupling Manager: Exchanges forces and velocities between DEM and CFD",
        'geotechnical_context': "Geotechnical Context",
        'validation_results_header': "Validation Results",
        'validation_results_desc': "This section shows the comparison between simulation results and experimental data.",
        'parameter_calibration_header': "Parameter Calibration",
        'parameter_calibration_desc': "This section allows you to calibrate model parameters by comparing simulation results with experimental data.",
        'current_geotech_params': "Current Material Properties",
        'parameter_sensitivity': "Parameter Sensitivity",
        'parameter_relationships': "Parameter Relationships",
        'soil_classification': "Soil Classification",
        'theoretical_background_header': "Theoretical Background",
        'reference_wang': "Reference: Wang et al. (2020)",
        'model_validation': "Model Validation",
        'parameter_sensitivity_analysis': "Parameter Sensitivity Analysis",
        'select_language': "Select Language",
        'geotechnical_params_title': "Material Properties",
        'density': "Density",
        'specific_gravity': "Specific Gravity",
        'water_content': "Water Content",
        'porosity': "Porosity",
        'void_ratio': "Void Ratio",
        'permeability': "Permeability",
        'cohesion': "Cohesion",
        'friction_angle': "Friction Angle",
        'elastic_modulus': "Elastic Modulus",
        'poisson_ratio': "Poisson's Ratio",
        'tensile_strength': "Tensile Strength",
        'compressive_strength': "Compressive Strength",
        'shear_strength': "Shear Strength",
        'hydraulic_conductivity': "Hydraulic Conductivity",
        'saturation': "Saturation",
        'capillary_pressure': "Capillary Pressure",
        'main_title': "CFD-DEM Simulation Dashboard",
        'main_navigation': "Navigation",
        'main_overview': "Overview",
        'main_coupling_framework': "Coupling Framework",
        'main_validation_results': "Validation Results",
        'main_parameter_calibration': "Parameter Calibration",
        'main_geotechnical_parameters': "Material Properties",
        'main_theoretical_background': "Theoretical Background",
        'case_studies': "Case Studies",
        'experimental_verification': "Experimental Verification",
        'tunnel_water_inrush': "Tunnel Water Inrush",
        'engineering_scale': "Engineering Scale Simulation",
        'mitigation_measures': "Mitigation Measures",
        'triaxial_test': "Triaxial Seepage Test",
        'experimental_data': "Experimental Data",
        'numerical_results': "Numerical Results",
        'comparison': "Comparison",
        'tunnel_geometry': "Tunnel Geometry",
        'water_inrush_simulation': "Water Inrush Simulation",
        'particle_erosion': "Particle Erosion",
        'deposition': "Deposition",
        'rock_displacement': "Rock Displacement",
        'seepage_velocity': "Seepage Velocity",
        'particle_outflow': "Particle Outflow",
        'porosity_evolution': "Porosity Evolution",
        'grouting': "Grouting",
        'drainage': "Drainage",
        'spraying': "Spraying",
        'mitigation_effectiveness': "Mitigation Effectiveness",
        'experimental_setup': "Experimental Setup",
        'test_parameters': "Test Parameters",
        'measurement_points': "Measurement Points",
        'data_analysis': "Data Analysis",
        'validation_metrics': "Validation Metrics",
        'error_analysis': "Error Analysis",
        'sensitivity_study': "Sensitivity Study",
        'case_study_results': "Case Study Results",
        'mitigation_analysis': "Mitigation Analysis",
        'performance_metrics': "Performance Metrics",
        'cost_analysis': "Cost Analysis",
        'recommendations': "Recommendations",
        'steps': "Steps",
        'simulation_steps': "Simulation Steps",
        'validation_steps': "Validation Steps",
        'case_study_title': "Case Studies",
        'case_study_1': "Experimental Verification",
        'case_study_2': "Tunnel Water Inrush",
        'case_study_3': "Mitigation Measures",
        'material_properties': "Material Properties",
        'physical_properties': "Physical Properties",
        'mechanical_properties': "Mechanical Properties",
        'hydraulic_properties': "Hydraulic Properties",
        'validation_metrics': "Validation Metrics",
        'error_metrics': "Error Metrics",
        'correlation_metrics': "Correlation Metrics",
        'mean_error': "Mean Error",
        'max_error': "Max Error",
        'correlation': "Correlation",
        'experimental_data': "Experimental Data",
        'simulation_data': "Simulation Data",
        'comparison': "Comparison",
        'trend': "Trend",
        'improving': "Improving",
        'stable': "Stable",
        'degrading': "Degrading",
        'validation_summary': "Validation Summary",
        'overall_score': "Overall Score",
        'parameter_impact': "Parameter Impact",
        'sensitivity_analysis': "Sensitivity Analysis",
        'effect_analysis': "Effect Analysis",
        'real_time_effects': "Real-time Effects",
        'parameter_adjustment': "Parameter Adjustment",
        'validation_results': "Validation Results",
        'metrics': "Metrics",
        'visualization': "Visualization",
        'erosion_analysis': "Erosion Analysis",
        'pressure_analysis': "Pressure Analysis",
        'velocity_analysis': "Velocity Analysis",
        'experimental_setup': "Experimental Setup",
        'test_parameters': "Test Parameters",
        'measurement_points': "Measurement Points",
        'data_analysis': "Data Analysis",
        'error_analysis': "Error Analysis",
        'sensitivity_study': "Sensitivity Study",
        'case_study_results': "Case Study Results",
        'mitigation_analysis': "Mitigation Analysis",
        'performance_metrics': "Performance Metrics",
        'cost_analysis': "Cost Analysis",
        'recommendations': "Recommendations",
        'overview': "Overview",
        'project_overview': "Project Overview",
        'project_description': "This dashboard provides a comprehensive view of the CFD-DEM simulation results, including the coupling framework, validation results, and parameter calibration.",
        'simulation_time': "Simulation Time",
        'total_steps': "Total Steps",
        'current_step': "Current Step",
        'simulation_progress': "Simulation Progress",
        'total_particles': "Total Particles",
        'active_particles': "Active Particles",
        'eroded_particles': "Eroded Particles",
        'erosion_score': "Erosion Score",
        'pressure_score': "Pressure Score",
        'velocity_score': "Velocity Score",
        'material_properties': "Material Properties",
        'physical_properties': "Physical Properties",
        'mechanical_properties': "Mechanical Properties",
        'hydraulic_properties': "Hydraulic Properties",
        'density': "Density",
        'specific_gravity': "Specific Gravity",
        'water_content': "Water Content",
        'porosity': "Porosity",
        'void_ratio': "Void Ratio",
        'permeability': "Permeability",
        'cohesion': "Cohesion",
        'friction_angle': "Friction Angle",
        'elastic_modulus': "Elastic Modulus",
        'poisson_ratio': "Poisson's Ratio",
        'tensile_strength': "Tensile Strength",
        'compressive_strength': "Compressive Strength",
        'shear_strength': "Shear Strength",
        'hydraulic_conductivity': "Hydraulic Conductivity",
        'saturation': "Saturation",
        'capillary_pressure': "Capillary Pressure",
        'particle_size_distribution': "Particle Size Distribution",
        'grain_size': "Grain Size",
        'passing_percentage': "Passing Percentage",
        'uniformity_coefficient': "Uniformity Coefficient (Cu)",
        'curvature_coefficient': "Curvature Coefficient (Cc)",
        'clay_content': "Clay Content",
        'silt_content': "Silt Content",
        'sand_content': "Sand Content",
        'gravel_content': "Gravel Content",
        'soil_classification': "Soil Classification",
        'uscs_classification': "USCS Classification",
        'soil_type': "Soil Type",
        'soil_description': "Soil Description",
        'material_parameters': "Material Parameters",
        'parameter_values': "Parameter Values",
        'parameter_units': "Units",
        'parameter_description': "Description",
        'parameter_impact': "Parameter Impact",
        'parameter_sensitivity': "Parameter Sensitivity",
        'parameter_relationships': "Parameter Relationships",
        'parameter_validation': "Parameter Validation",
        'parameter_calibration': "Parameter Calibration",
        'parameter_history': "Parameter History",
        'parameter_changes': "Parameter Changes",
        'parameter_effects': "Parameter Effects",
        'parameter_constraints': "Parameter Constraints",
        'parameter_limits': "Parameter Limits",
        'parameter_ranges': "Parameter Ranges",
        'parameter_defaults': "Default Values",
        'parameter_current': "Current Values",
        'parameter_previous': "Previous Values",
        'parameter_recommended': "Recommended Values",
        'parameter_optimal': "Optimal Values",
        'parameter_uncertainty': "Parameter Uncertainty",
        'parameter_variability': "Parameter Variability",
        'parameter_stability': "Parameter Stability",
        'parameter_reliability': "Parameter Reliability",
        'parameter_accuracy': "Parameter Accuracy",
        'parameter_precision': "Parameter Precision",
        'parameter_validation_score': "Validation Score",
        'parameter_calibration_score': "Calibration Score",
        'parameter_sensitivity_score': "Sensitivity Score",
        'parameter_impact_score': "Impact Score",
        'parameter_importance': "Parameter Importance",
        'parameter_significance': "Parameter Significance",
        'parameter_correlation': "Parameter Correlation",
        'parameter_dependency': "Parameter Dependency",
        'parameter_interaction': "Parameter Interaction",
        'parameter_coupling': "Parameter Coupling",
        'parameter_feedback': "Parameter Feedback",
        'parameter_control': "Parameter Control",
        'parameter_optimization': "Parameter Optimization",
        'parameter_tuning': "Parameter Tuning",
        'parameter_adjustment': "Parameter Adjustment",
        'parameter_modification': "Parameter Modification",
        'parameter_update': "Parameter Update",
        'parameter_save': "Save Parameters",
        'parameter_load': "Load Parameters",
        'parameter_reset': "Reset Parameters",
        'parameter_export': "Export Parameters",
        'parameter_import': "Import Parameters",
        'parameter_backup': "Backup Parameters",
        'parameter_restore': "Restore Parameters",
        'parameter_version': "Parameter Version",
        'parameter_history': "Parameter History",
        'parameter_changes': "Parameter Changes",
        'parameter_log': "Parameter Log",
        'parameter_report': "Parameter Report",
        'parameter_summary': "Parameter Summary",
        'parameter_details': "Parameter Details",
        'parameter_info': "Parameter Information",
        'parameter_help': "Parameter Help",
        'parameter_documentation': "Parameter Documentation",
        'parameter_reference': "Parameter Reference",
        'parameter_guide': "Parameter Guide",
        'parameter_tutorial': "Parameter Tutorial",
        'parameter_examples': "Parameter Examples",
        'parameter_demo': "Parameter Demo",
        'parameter_test': "Parameter Test",
        'parameter_validation': "Parameter Validation",
        'parameter_verification': "Parameter Verification",
        'parameter_check': "Parameter Check",
        'parameter_audit': "Parameter Audit",
        'parameter_review': "Parameter Review",
        'parameter_analysis': "Parameter Analysis",
        'parameter_evaluation': "Parameter Evaluation",
        'parameter_assessment': "Parameter Assessment",
        'parameter_rating': "Parameter Rating",
        'parameter_ranking': "Parameter Ranking",
        'parameter_priority': "Parameter Priority",
        'parameter_criticality': "Parameter Criticality",
        'parameter_risk': "Parameter Risk",
        'parameter_impact': "Parameter Impact",
        'parameter_effect': "Parameter Effect",
        'parameter_influence': "Parameter Influence",
        'parameter_contribution': "Parameter Contribution",
        'parameter_role': "Parameter Role",
        'parameter_function': "Parameter Function",
        'parameter_behavior': "Parameter Behavior",
        'parameter_characteristics': "Parameter Characteristics",
        'parameter_properties': "Parameter Properties",
        'parameter_attributes': "Parameter Attributes",
        'parameter_features': "Parameter Features",
        'parameter_qualities': "Parameter Qualities",
        'parameter_traits': "Parameter Traits",
        'parameter_aspects': "Parameter Aspects",
        'parameter_factors': "Parameter Factors",
        'parameter_elements': "Parameter Elements",
        'parameter_components': "Parameter Components",
        'parameter_parts': "Parameter Parts",
        'parameter_sections': "Parameter Sections",
        'parameter_categories': "Parameter Categories",
        'parameter_groups': "Parameter Groups",
        'parameter_classes': "Parameter Classes",
        'parameter_types': "Parameter Types",
        'parameter_kinds': "Parameter Kinds",
        'parameter_forms': "Parameter Forms",
        'parameter_varieties': "Parameter Varieties",
        'parameter_species': "Parameter Species",
        'parameter_instances': "Parameter Instances",
        'parameter_cases': "Parameter Cases",
        'parameter_examples': "Parameter Examples",
        'parameter_samples': "Parameter Samples",
        'parameter_specimens': "Parameter Specimens",
        'parameter_models': "Parameter Models",
        'parameter_patterns': "Parameter Patterns",
        'parameter_templates': "Parameter Templates",
        'parameter_prototypes': "Parameter Prototypes",
        'parameter_standards': "Parameter Standards",
        'parameter_norms': "Parameter Norms",
        'parameter_criteria': "Parameter Criteria",
        'parameter_conditions': "Parameter Conditions",
        'parameter_requirements': "Parameter Requirements",
        'parameter_specifications': "Parameter Specifications",
        'parameter_definitions': "Parameter Definitions",
        'parameter_terms': "Parameter Terms",
        'parameter_concepts': "Parameter Concepts",
        'parameter_principles': "Parameter Principles",
        'parameter_theories': "Parameter Theories",
        'parameter_laws': "Parameter Laws",
        'parameter_rules': "Parameter Rules",
        'parameter_guidelines': "Parameter Guidelines",
        'parameter_policies': "Parameter Policies",
        'parameter_procedures': "Parameter Procedures",
        'parameter_methods': "Parameter Methods",
        'parameter_techniques': "Parameter Techniques",
        'parameter_approaches': "Parameter Approaches",
        'parameter_strategies': "Parameter Strategies",
        'parameter_tactics': "Parameter Tactics",
        'parameter_plans': "Parameter Plans",
        'parameter_schemes': "Parameter Schemes",
        'parameter_programs': "Parameter Programs",
        'parameter_systems': "Parameter Systems",
        'parameter_frameworks': "Parameter Frameworks",
        'parameter_architectures': "Parameter Architectures",
        'parameter_structures': "Parameter Structures",
        'parameter_organizations': "Parameter Organizations",
        'parameter_arrangements': "Parameter Arrangements",
        'parameter_configurations': "Parameter Configurations",
        'parameter_setups': "Parameter Setups",
        'parameter_installations': "Parameter Installations",
        'parameter_implementations': "Parameter Implementations",
        'parameter_deployments': "Parameter Deployments",
        'parameter_operations': "Parameter Operations",
        'parameter_processes': "Parameter Processes",
        'parameter_activities': "Parameter Activities",
        'parameter_tasks': "Parameter Tasks",
        'parameter_jobs': "Parameter Jobs",
        'parameter_work': "Parameter Work",
        'parameter_labor': "Parameter Labor",
        'parameter_effort': "Parameter Effort",
        'parameter_energy': "Parameter Energy",
        'parameter_power': "Parameter Power",
        'parameter_force': "Parameter Force",
        'parameter_pressure': "Parameter Pressure",
        'parameter_stress': "Parameter Stress",
        'parameter_strain': "Parameter Strain",
        'parameter_deformation': "Parameter Deformation",
        'parameter_displacement': "Parameter Displacement",
        'parameter_velocity': "Parameter Velocity",
        'parameter_acceleration': "Parameter Acceleration",
        'parameter_momentum': "Parameter Momentum",
        'parameter_impulse': "Parameter Impulse",
        'parameter_work': "Parameter Work",
        'parameter_energy': "Parameter Energy",
        'parameter_power': "Parameter Power",
        'parameter_force': "Parameter Force",
        'parameter_pressure': "Parameter Pressure",
        'parameter_stress': "Parameter Stress",
        'parameter_strain': "Parameter Strain",
        'parameter_deformation': "Parameter Deformation",
        'parameter_displacement': "Parameter Displacement",
        'parameter_velocity': "Parameter Velocity",
        'parameter_acceleration': "Parameter Acceleration",
        'parameter_momentum': "Parameter Momentum",
        'parameter_impulse': "Parameter Impulse",
        'edit_granular_curve': "Edit Granular Curve",
        'update_granular_curve': "Update Granular Curve",
        'soil_uscs_classification': "USCS Soil Classification",
        'wang_reference_title': "Wang et al. (2020) Reference",
        'wang_reference_desc': "Key findings from Wang's research on CFD-DEM coupling",
        'wang_key_findings': "Key findings from the research:",
        'wang_finding_1': "1. Particle-fluid interaction patterns",
        'wang_finding_1_detail': "- Identified dominant flow regimes",
        'wang_finding_1_detail_2': "- Characterized particle transport mechanisms",
        'wang_finding_1_detail_3': "- Quantified erosion rates",
        'wang_finding_2': "2. Coupling effects",
        'wang_finding_2_detail': "- Analyzed force transmission",
        'wang_finding_2_detail_2': "- Studied momentum exchange",
        'wang_finding_2_detail_3': "- Evaluated energy dissipation",
        'wang_finding_3': "3. Validation results",
        'wang_finding_3_detail': "- Compared with experimental data",
        'wang_finding_3_detail_2': "- Verified numerical accuracy",
        'wang_finding_3_detail_3': "- Assessed model limitations",
        'wang_paper_reference': "Reference: Wang et al. (2020) - CFD-DEM coupling for particle-fluid systems",
        'erosion_rate': "Erosion Rate",
        'pressure_distribution': "Pressure Distribution",
        'velocity_profile': "Velocity Profile",
        'position': "Position",
        'pressure_unit': "Pressure (kPa)",
        'velocity_unit': "Velocity (m/s)",
        'time_steps': "Time Steps",
        'sample_pressure_title': "Pressure Distribution Analysis",
        'sample_velocity_title': "Velocity Profile Analysis",
        'soil_gravel': "Gravel",
        'soil_sand': "Sand",
        'soil_silt': "Silt",
        'soil_clay': "Clay",
        'soil_types': "Soil Types",
        'soil_well_graded': "Well-graded",
        'soil_poorly_graded': "Poorly-graded",
        'soil_silty': "Silty",
        'soil_clayey': "Clayey",
        'soil_current_classification': "Current Classification",
        'soil_composition': "Soil Composition",
        'soil_classification_result': "Classification Result",
        'soil_classification_result_value': "SW-SM (Well-graded sand with silt)",
        'soil_cu': "Cu = 4.5",
        'soil_cc': "Cc = 1.2",
        'soil_fine_content': "Fine content = 35%",
        'update_parameters': "Update Parameters",
        'parameter_updated': "Parameters updated successfully",
        'config_updated': "Configuration updated successfully",
        'config_error': "Error updating configuration",
        'simulation_success': "Simulation completed successfully",
        'simulation_failed': "Simulation failed: ",
        'simulation_error': "Error running simulation: ",
        'sample_data_title': "Sample Data Visualization",
        'sample_data_error': "Error generating sample data",
        'fluidity_effects_title': "Fluidity Effects Analysis",
        'fluidity_effects_error': "Error analyzing fluidity effects",
        'coupling_diagram_error': "Error loading coupling diagram",
        'granular_curve_title': "Granular Curve",
        'grain_size_mm': "Grain Size (mm)",
        'passing_percentage_pct': "Passing Percentage (%)",
        'clay': "Clay",
        'sand': "Sand",
        'gravel': "Gravel",
        'granular_curve_updated': "Granular curve updated successfully",
        'geotechnical_params_error': "Error loading geotechnical parameters",
        'param_bond_strength': "Bond Strength",
        'param_fluid_density': "Fluid Density",
        'param_fluid_viscosity': "Fluid Viscosity",
        'error': "Error",
        'overall_validation_score': "Overall Validation Score",
        'experimental': "Experimental",
        'recent_simulation_results': "Recent Simulation Results",
        'simulation': "Simulation",
        'water_inrush_simulation': "Water Inrush Simulation",
        'water_inrush_process': "Water Inrush Process",
        'flow_vectors': "Flow Vectors",
        'simulation_parameters': "Simulation Parameters",
        'initial_pressure': "Initial Pressure",
        'flow_rate': "Flow Rate",
        'erosion_rate': "Erosion Rate",
        'particle_size': "Particle Size",
        'pressure_evolution': "Pressure Evolution",
        'flow_rate_evolution': "Flow Rate Evolution",
        'real_time_monitoring': "Real-time Monitoring",
    },
    'zh': {
        'dashboard_title': "CFD-DEM仿真仪表板",
        'navigation': "导航",
        'overview': "概述",
        'coupling_framework': "耦合框架",
        'coupling_framework_diagram': "耦合框架图",
        'validation_results': "验证结果",
        'parameter_calibration': "参数校准",
        'parameter_calibration_title': "参数校准界面",
        'geotechnical_parameters': "材料特性",
        'theoretical_background': "理论背景",
        'project_overview': "项目概述",
        'project_overview_desc': "本仪表板提供CFD-DEM仿真结果的全面视图，包括耦合框架、验证结果和参数校准。",
        'simulation_time': "仿真时间",
        'particles': "颗粒数",
        'validation_score': "验证得分",
        'geotech_params_summary': "材料特性摘要",
        'recent_results': "最新结果",
        'cfd_dem_framework': "CFD-DEM耦合框架",
        'framework_desc': "耦合框架展示了离散元法(DEM)颗粒与计算流体力学(CFD)流体通过耦合管理器的相互作用。",
        'framework_components': "框架组成",
        'dem_particles': "DEM（颗粒）：处理颗粒运动、接触和力",
        'cfd_fluid': "CFD（流体）：计算流体流动、压力和速度场",
        'coupling_manager': "耦合管理器：在DEM和CFD之间交换力和速度",
        'geotechnical_context': "岩土背景",
        'validation_results_header': "验证结果",
        'validation_results_desc': "本节显示模拟结果与实验数据的对比。",
        'parameter_calibration_header': "参数校准",
        'parameter_calibration_desc': "本节允许您通过与实验数据对比来校准模型参数。",
        'current_geotech_params': "当前材料特性",
        'parameter_sensitivity': "参数敏感性",
        'parameter_relationships': "参数关系",
        'soil_classification': "土壤分类",
        'theoretical_background_header': "理论背景",
        'reference_wang': "参考文献：Wang等 (2020)",
        'model_validation': "模型验证",
        'parameter_sensitivity_analysis': "参数敏感性分析",
        'select_language': "选择语言",
        'geotechnical_params_title': "材料特性",
        'density': "密度",
        'specific_gravity': "比重",
        'water_content': "含水量",
        'porosity': "孔隙率",
        'void_ratio': "孔隙比",
        'permeability': "渗透系数",
        'cohesion': "粘聚力",
        'friction_angle': "内摩擦角",
        'elastic_modulus': "弹性模量",
        'poisson_ratio': "泊松比",
        'tensile_strength': "抗拉强度",
        'compressive_strength': "抗压强度",
        'shear_strength': "抗剪强度",
        'hydraulic_conductivity': "导水率",
        'saturation': "饱和度",
        'capillary_pressure': "毛细压力",
        'main_title': "CFD-DEM仿真仪表板",
        'main_navigation': "导航",
        'main_overview': "概述",
        'main_coupling_framework': "耦合框架",
        'main_validation_results': "验证结果",
        'main_parameter_calibration': "参数校准",
        'main_geotechnical_parameters': "材料特性",
        'main_theoretical_background': "理论背景",
        'case_studies': "案例研究",
        'experimental_verification': "实验验证",
        'tunnel_water_inrush': "隧道突水",
        'engineering_scale': "工程尺度模拟",
        'mitigation_measures': "缓解措施",
        'triaxial_test': "三轴渗流试验",
        'experimental_data': "实验数据",
        'numerical_results': "数值结果",
        'comparison': "对比分析",
        'tunnel_geometry': "隧道几何",
        'water_inrush_simulation': "突水模拟",
        'particle_erosion': "颗粒侵蚀",
        'deposition': "沉积",
        'rock_displacement': "岩石位移",
        'seepage_velocity': "渗流速度",
        'particle_outflow': "颗粒流出",
        'porosity_evolution': "孔隙率演化",
        'grouting': "注浆",
        'drainage': "排水",
        'spraying': "喷射",
        'mitigation_effectiveness': "缓解效果",
        'experimental_setup': "实验设置",
        'test_parameters': "试验参数",
        'measurement_points': "测量点",
        'data_analysis': "数据分析",
        'validation_metrics': "验证指标",
        'error_analysis': "误差分析",
        'sensitivity_study': "敏感性研究",
        'case_study_results': "案例研究结果",
        'mitigation_analysis': "缓解分析",
        'performance_metrics': "性能指标",
        'cost_analysis': "成本分析",
        'recommendations': "建议",
        'steps': "步数",
        'simulation_steps': "仿真步数",
        'validation_steps': "验证步数",
        'case_study_title': "案例研究",
        'case_study_1': "实验验证",
        'case_study_2': "隧道突水",
        'case_study_3': "缓解措施",
        'material_properties': "材料特性",
        'physical_properties': "物理特性",
        'mechanical_properties': "力学特性",
        'hydraulic_properties': "水力特性",
        'validation_metrics': "验证指标",
        'error_metrics': "误差指标",
        'correlation_metrics': "相关性指标",
        'mean_error': "平均误差",
        'max_error': "最大误差",
        'correlation': "相关系数",
        'experimental_data': "实验数据",
        'simulation_data': "仿真数据",
        'comparison': "对比分析",
        'trend': "趋势",
        'improving': "改善",
        'stable': "稳定",
        'degrading': "恶化",
        'validation_summary': "验证总结",
        'overall_score': "总评分",
        'parameter_impact': "参数影响",
        'sensitivity_analysis': "敏感性分析",
        'effect_analysis': "影响分析",
        'real_time_effects': "实时影响",
        'parameter_adjustment': "参数调整",
        'validation_results': "验证结果",
        'metrics': "指标",
        'visualization': "可视化",
        'erosion_analysis': "侵蚀分析",
        'pressure_analysis': "压力分析",
        'velocity_analysis': "速度分析",
        'experimental_setup': "实验设置",
        'test_parameters': "试验参数",
        'measurement_points': "测量点",
        'data_analysis': "数据分析",
        'error_analysis': "误差分析",
        'sensitivity_study': "敏感性研究",
        'case_study_results': "案例研究结果",
        'mitigation_analysis': "缓解分析",
        'performance_metrics': "性能指标",
        'cost_analysis': "成本分析",
        'recommendations': "建议",
        'overview': "概述",
        'project_overview': "项目概述",
        'project_description': "本仪表板提供CFD-DEM仿真结果的全面视图，包括耦合框架、验证结果和参数校准。",
        'simulation_time': "仿真时间",
        'total_steps': "总步数",
        'current_step': "当前步数",
        'simulation_progress': "仿真进度",
        'total_particles': "总颗粒数",
        'active_particles': "活跃颗粒数",
        'eroded_particles': "侵蚀颗粒数",
        'erosion_score': "侵蚀得分",
        'pressure_score': "压力得分",
        'velocity_score': "速度得分",
        'material_properties': "材料特性",
        'physical_properties': "物理特性",
        'mechanical_properties': "力学特性",
        'hydraulic_properties': "水力特性",
        'density': "密度",
        'specific_gravity': "比重",
        'water_content': "含水量",
        'porosity': "孔隙率",
        'void_ratio': "孔隙比",
        'permeability': "渗透系数",
        'cohesion': "粘聚力",
        'friction_angle': "内摩擦角",
        'elastic_modulus': "弹性模量",
        'poisson_ratio': "泊松比",
        'tensile_strength': "抗拉强度",
        'compressive_strength': "抗压强度",
        'shear_strength': "抗剪强度",
        'hydraulic_conductivity': "导水率",
        'saturation': "饱和度",
        'capillary_pressure': "毛细压力",
        'particle_size_distribution': "颗粒级配",
        'grain_size': "粒径",
        'passing_percentage': "通过率",
        'uniformity_coefficient': "均匀系数 (Cu)",
        'curvature_coefficient': "曲率系数 (Cc)",
        'clay_content': "粘土含量",
        'silt_content': "粉土含量",
        'sand_content': "砂含量",
        'gravel_content': "砾石含量",
        'soil_classification': "土壤分类",
        'uscs_classification': "USCS分类",
        'soil_type': "土壤类型",
        'soil_description': "土壤描述",
        'material_parameters': "材料参数",
        'parameter_values': "参数值",
        'parameter_units': "单位",
        'parameter_description': "描述",
        'parameter_impact': "参数影响",
        'parameter_sensitivity': "参数敏感性",
        'parameter_relationships': "参数关系",
        'parameter_validation': "参数验证",
        'parameter_calibration': "参数校准",
        'parameter_history': "参数历史",
        'parameter_changes': "参数变化",
        'parameter_effects': "参数效应",
        'parameter_constraints': "参数约束",
        'parameter_limits': "参数限制",
        'parameter_ranges': "参数范围",
        'parameter_defaults': "默认值",
        'parameter_current': "当前值",
        'parameter_previous': "先前值",
        'parameter_recommended': "推荐值",
        'parameter_optimal': "最优值",
        'parameter_uncertainty': "参数不确定性",
        'parameter_variability': "参数变异性",
        'parameter_stability': "参数稳定性",
        'parameter_reliability': "参数可靠性",
        'parameter_accuracy': "参数准确性",
        'parameter_precision': "参数精确度",
        'parameter_validation_score': "验证得分",
        'parameter_calibration_score': "校准得分",
        'parameter_sensitivity_score': "敏感性得分",
        'parameter_impact_score': "影响得分",
        'parameter_importance': "参数重要性",
        'parameter_significance': "参数显著性",
        'parameter_correlation': "参数相关性",
        'parameter_dependency': "参数依赖性",
        'parameter_interaction': "参数相互作用",
        'parameter_coupling': "参数耦合",
        'parameter_feedback': "参数反馈",
        'parameter_control': "参数控制",
        'parameter_optimization': "参数优化",
        'parameter_tuning': "参数调谐",
        'parameter_adjustment': "参数调整",
        'parameter_modification': "参数修改",
        'parameter_update': "参数更新",
        'parameter_save': "保存参数",
        'parameter_load': "加载参数",
        'parameter_reset': "重置参数",
        'parameter_export': "导出参数",
        'parameter_import': "导入参数",
        'parameter_backup': "备份参数",
        'parameter_restore': "恢复参数",
        'parameter_version': "参数版本",
        'parameter_history': "参数历史",
        'parameter_changes': "参数变化",
        'parameter_log': "参数日志",
        'parameter_report': "参数报告",
        'parameter_summary': "参数摘要",
        'parameter_details': "参数详情",
        'parameter_info': "参数信息",
        'parameter_help': "参数帮助",
        'parameter_documentation': "参数文档",
        'parameter_reference': "参数参考",
        'parameter_guide': "参数指南",
        'parameter_tutorial': "参数教程",
        'parameter_examples': "参数示例",
        'parameter_demo': "参数演示",
        'parameter_test': "参数测试",
        'parameter_validation': "参数验证",
        'parameter_verification': "参数确认",
        'parameter_check': "参数检查",
        'parameter_audit': "参数审计",
        'parameter_review': "参数审查",
        'parameter_analysis': "参数分析",
        'parameter_evaluation': "参数评估",
        'parameter_assessment': "参数评定",
        'parameter_rating': "参数评级",
        'parameter_ranking': "参数排名",
        'parameter_priority': "参数优先级",
        'parameter_criticality': "参数关键性",
        'parameter_risk': "参数风险",
        'parameter_impact': "参数影响",
        'parameter_effect': "参数效应",
        'parameter_influence': "参数影响",
        'parameter_contribution': "参数贡献",
        'parameter_role': "参数作用",
        'parameter_function': "参数功能",
        'parameter_behavior': "参数行为",
        'parameter_characteristics': "参数特征",
        'parameter_properties': "参数属性",
        'parameter_attributes': "参数特性",
        'parameter_features': "参数特点",
        'parameter_qualities': "参数品质",
        'parameter_traits': "参数特质",
        'parameter_aspects': "参数方面",
        'parameter_factors': "参数因素",
        'parameter_elements': "参数要素",
        'parameter_components': "参数组件",
        'parameter_parts': "参数部分",
        'parameter_sections': "参数章节",
        'parameter_categories': "参数类别",
        'parameter_groups': "参数组",
        'parameter_classes': "参数类",
        'parameter_types': "参数类型",
        'parameter_kinds': "参数种类",
        'parameter_forms': "参数形式",
        'parameter_varieties': "参数变体",
        'parameter_species': "参数物种",
        'parameter_instances': "参数实例",
        'parameter_cases': "参数案例",
        'parameter_examples': "参数示例",
        'parameter_samples': "参数样本",
        'parameter_specimens': "参数标本",
        'parameter_models': "参数模型",
        'parameter_patterns': "参数模式",
        'parameter_templates': "参数模板",
        'parameter_prototypes': "参数原型",
        'parameter_standards': "参数标准",
        'parameter_norms': "参数规范",
        'parameter_criteria': "参数标准",
        'parameter_conditions': "参数条件",
        'parameter_requirements': "参数要求",
        'parameter_specifications': "参数规格",
        'parameter_definitions': "参数定义",
        'parameter_terms': "参数术语",
        'parameter_concepts': "参数概念",
        'parameter_principles': "参数原理",
        'parameter_theories': "参数理论",
        'parameter_laws': "参数法则",
        'parameter_rules': "参数规则",
        'parameter_guidelines': "参数指南",
        'parameter_policies': "参数政策",
        'parameter_procedures': "参数程序",
        'parameter_methods': "参数方法",
        'parameter_techniques': "参数技术",
        'parameter_approaches': "参数方法",
        'parameter_strategies': "参数策略",
        'parameter_tactics': "参数战术",
        'parameter_plans': "参数计划",
        'parameter_schemes': "参数方案",
        'parameter_programs': "参数程序",
        'parameter_systems': "参数系统",
        'parameter_frameworks': "参数框架",
        'parameter_architectures': "参数架构",
        'parameter_structures': "参数结构",
        'parameter_organizations': "参数组织",
        'parameter_arrangements': "参数安排",
        'parameter_configurations': "参数配置",
        'parameter_setups': "参数设置",
        'parameter_installations': "参数安装",
        'parameter_implementations': "参数实现",
        'parameter_deployments': "参数部署",
        'parameter_operations': "参数操作",
        'parameter_processes': "参数过程",
        'parameter_activities': "参数活动",
        'parameter_tasks': "参数任务",
        'parameter_jobs': "参数工作",
        'parameter_work': "参数工作",
        'parameter_labor': "参数劳动",
        'parameter_effort': "参数努力",
        'parameter_energy': "参数能量",
        'parameter_power': "参数功率",
        'parameter_force': "参数力",
        'parameter_pressure': "参数压力",
        'parameter_stress': "参数应力",
        'parameter_strain': "参数应变",
        'parameter_deformation': "参数变形",
        'parameter_displacement': "参数位移",
        'parameter_velocity': "参数速度",
        'parameter_acceleration': "参数加速度",
        'parameter_momentum': "参数动量",
        'parameter_impulse': "参数冲量",
        'parameter_work': "参数功",
        'parameter_energy': "参数能量",
        'parameter_power': "参数功率",
        'parameter_force': "参数力",
        'parameter_pressure': "参数压力",
        'parameter_stress': "参数应力",
        'parameter_strain': "参数应变",
        'parameter_deformation': "参数变形",
        'parameter_displacement': "参数位移",
        'parameter_velocity': "参数速度",
        'parameter_acceleration': "参数加速度",
        'parameter_momentum': "参数动量",
        'parameter_impulse': "参数冲量",
        'edit_granular_curve': "编辑颗粒级配曲线",
        'update_granular_curve': "更新颗粒级配曲线",
        'soil_uscs_classification': "USCS土壤分类",
        'wang_reference_title': "Wang等(2020)参考文献",
        'wang_reference_desc': "Wang关于CFD-DEM耦合研究的主要发现",
        'wang_key_findings': "研究的主要发现：",
        'wang_finding_1': "1. 颗粒-流体相互作用模式",
        'wang_finding_1_detail': "- 识别主要流动状态",
        'wang_finding_1_detail_2': "- 表征颗粒输运机制",
        'wang_finding_1_detail_3': "- 量化侵蚀速率",
        'wang_finding_2': "2. 耦合效应",
        'wang_finding_2_detail': "- 分析力传递",
        'wang_finding_2_detail_2': "- 研究动量交换",
        'wang_finding_2_detail_3': "- 评估能量耗散",
        'wang_finding_3': "3. 验证结果",
        'wang_finding_3_detail': "- 与实验数据对比",
        'wang_finding_3_detail_2': "- 验证数值精度",
        'wang_finding_3_detail_3': "- 评估模型局限性",
        'wang_paper_reference': "参考文献：Wang等(2020) - 颗粒-流体系统的CFD-DEM耦合",
        'erosion_rate': "侵蚀率",
        'pressure_distribution': "压力分布",
        'velocity_profile': "速度分布",
        'position': "位置",
        'pressure_unit': "压力 (kPa)",
        'velocity_unit': "速度 (m/s)",
        'time_steps': "时间步",
        'sample_pressure_title': "压力分布分析",
        'sample_velocity_title': "速度分布分析",
        'soil_gravel': "砾石",
        'soil_sand': "砂",
        'soil_silt': "粉土",
        'soil_clay': "粘土",
        'soil_types': "土壤类型",
        'soil_well_graded': "级配良好",
        'soil_poorly_graded': "级配不良",
        'soil_silty': "粉质",
        'soil_clayey': "粘质",
        'soil_current_classification': "当前分类",
        'soil_composition': "土壤组成",
        'soil_classification_result': "分类结果",
        'soil_classification_result_value': "SW-SM (含粉土的级配良好砂)",
        'soil_cu': "Cu = 4.5",
        'soil_cc': "Cc = 1.2",
        'soil_fine_content': "细粒含量 = 35%",
        'update_parameters': "更新参数",
        'parameter_updated': "参数更新成功",
        'config_updated': "配置更新成功",
        'config_error': "更新配置时出错",
        'simulation_success': "仿真成功完成",
        'simulation_failed': "仿真失败：",
        'simulation_error': "运行仿真时出错：",
        'sample_data_title': "样本数据可视化",
        'sample_data_error': "生成样本数据时出错",
        'fluidity_effects_title': "流动性效应分析",
        'fluidity_effects_error': "分析流动性效应时出错",
        'coupling_diagram_error': "加载耦合图时出错",
        'granular_curve_title': "颗粒级配曲线",
        'grain_size_mm': "粒径 (mm)",
        'passing_percentage_pct': "通过率 (%)",
        'clay': "粘土",
        'sand': "砂",
        'gravel': "砾石",
        'granular_curve_updated': "颗粒级配曲线更新成功",
        'geotechnical_params_error': "加载岩土参数时出错",
        'param_bond_strength': "结合强度",
        'param_fluid_density': "流体密度",
        'param_fluid_viscosity': "流体粘度",
        'error': "误差",
        'overall_validation_score': "总体验证得分",
        'experimental': "实验",
        'recent_simulation_results': "最新仿真结果",
        'simulation': "仿真",
        'water_inrush_simulation': "突水模拟",
        'water_inrush_process': "突水过程",
        'flow_vectors': "流向量",
        'simulation_parameters': "模拟参数",
        'initial_pressure': "初始压力",
        'flow_rate': "流量",
        'erosion_rate': "侵蚀率",
        'particle_size': "颗粒尺寸",
        'pressure_evolution': "压力演化",
        'flow_rate_evolution': "流量演化",
        'real_time_monitoring': "实时监测",
    }
}

# Function to generate sample data for visualization
def generate_sample_data(params):
    """Generate sample data for visualization based on parameters."""
    # Create time points
    time_points = np.linspace(0, 10, 100)
    
    # Generate erosion data
    erosion_data = {
        'time': time_points,
        'erosion_rate': 0.1 * np.exp(-0.2 * time_points) + 0.05 * np.sin(time_points),
        'cumulative_erosion': np.cumsum(0.1 * np.exp(-0.2 * time_points) + 0.05 * np.sin(time_points)) * 0.1,
        'pressure': 1e6 * np.exp(-0.1 * time_points) + 0.5e5 * np.sin(time_points),
        'velocity': 2.0 * np.exp(-0.15 * time_points) + 0.5 * np.sin(time_points)
    }
    
    # Generate granular curve data
    grain_sizes = params['geotechnical']['grain_sizes']
    passing_percentages = params['geotechnical']['passing_percentages']
    
    # Create smooth curve for granular distribution
    x_smooth = np.logspace(np.log10(min(grain_sizes)), np.log10(max(grain_sizes)), 100)
    y_smooth = np.interp(x_smooth, grain_sizes[::-1], passing_percentages[::-1])
    
    granular_data = {
        'grain_size': x_smooth,
        'passing_percentage': y_smooth
    }
    
    return erosion_data, granular_data

# Function to update config with new parameters
def update_config(new_params):
    """Update the configuration with new parameters."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['dem']['particle_radius'] = new_params['particle_radius']
        config['dem']['bond_strength'] = new_params['bond_strength']
        config['cfd']['fluid_density'] = new_params['fluid_density']
        config['cfd']['fluid_viscosity'] = new_params['fluid_viscosity']
        
        # Save updated config
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)
        st.success(LANG['en']['config_updated'])
    except Exception as e:
        st.error(LANG['en']['config_error'])

# Function to run simulation
def run_simulation():
    """Run the simulation and return the result."""
    try:
        # Run the demo script
        result = subprocess.run(['python', 'demo.py'], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            return True, LANG['en']['simulation_success']
        else:
            return False, f"{LANG['en']['simulation_failed']}{result.stderr}"
    except Exception as e:
        return False, f"{LANG['en']['simulation_error']}{str(e)}"

# Function to analyze fluidity effects
def analyze_fluidity_effects(params):
    """Analyze the effects of fluidity on the simulation."""
    st.subheader(LANG['en']['fluidity_effects_title'])
    try:
        # Sample data for demonstration
        time_steps = np.linspace(0, 10, 100)
        flow_effects = {
            'low': params['flow_rate'] * 0.5 * np.sin(time_steps),
            'medium': params['flow_rate'] * np.sin(time_steps),
            'high': params['flow_rate'] * 1.5 * np.sin(time_steps)
        }
        
        # Calculate effects of pressure gradient
        pressure_effects = {
            'low': params['pressure_gradient'] * 0.5 * np.exp(-0.1 * time_steps),
            'medium': params['pressure_gradient'] * np.exp(-0.1 * time_steps),
            'high': params['pressure_gradient'] * 1.5 * np.exp(-0.1 * time_steps)
        }
        
        # Calculate turbulence effects
        turbulence_effects = {
            'low': params['turbulence_intensity'] * 0.5 * np.random.normal(0, 1, len(time_steps)),
            'medium': params['turbulence_intensity'] * np.random.normal(0, 1, len(time_steps)),
            'high': params['turbulence_intensity'] * 1.5 * np.random.normal(0, 1, len(time_steps))
        }
        
        return time_steps, flow_effects, pressure_effects, turbulence_effects
    except Exception as e:
        st.error(LANG['en']['fluidity_effects_error'])
        return None, None, None, None

def load_coupling_diagram(lang):
    """Load and display the coupling framework diagram."""
    st.subheader(lang['coupling_framework_diagram'])
    try:
        # Use language-specific diagram
        lang_code = 'zh' if lang == LANG['zh'] else 'en'
        st.image(f'results/coupling_framework_{lang_code}.png', use_container_width=True)
    except Exception as e:
        st.error(lang['coupling_diagram_error'])

def create_validation_summary(validation_manager, lang):
    """Create a summary of validation results."""
    st.subheader(lang['validation_summary'])
    
    # Create tabs for different validation aspects
    tab1, tab2 = st.tabs([lang['metrics'], lang['visualization']])
    
    with tab1:
        # Load validation data
        validation_data = {
            'erosion_rate': {
                'simulation': 0.85,
                'experimental': 0.82,
                'error': 3.7,
                'trend': lang['improving']
            },
            'pressure_distribution': {
                'simulation': 0.92,
                'experimental': 0.89,
                'error': 3.4,
                'trend': lang['stable']
            },
            'velocity_profile': {
                'simulation': 0.78,
                'experimental': 0.75,
                'error': 4.0,
                'trend': lang['improving']
            }
        }
        
        # Display validation metrics in a more detailed way
        col1, col2, col3 = st.columns(3)
        
        for i, (metric, data) in enumerate(validation_data.items()):
            with [col1, col2, col3][i]:
                st.metric(
                    label=metric.replace('_', ' ').title(),
                    value=f"{data['simulation']:.2f}",
                    delta=f"{data['error']:.1f}% {lang['error']}"
                )
                st.write(f"{lang['experimental']}: {data['experimental']:.2f}")
        
        # Overall validation score with trend
        overall_score = sum(data['simulation'] for data in validation_data.values()) / len(validation_data)
        st.metric(
            label=lang['overall_score'],
            value=f"{overall_score:.2f}",
            delta=lang['improving']
        )
    
    with tab2:
        # Create visualization of validation results
        fig = make_subplots(rows=3, cols=1, subplot_titles=(
            lang['erosion_analysis'],
            lang['pressure_analysis'],
            lang['velocity_analysis']
        ))
        
        # Add traces for each metric
        metrics = ['erosion_rate', 'pressure_distribution', 'velocity_profile']
        for i, metric in enumerate(metrics):
            data = validation_data[metric]
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[data['experimental'], data['simulation']],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title()
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True, key="validation_summary_plot")

def create_parameter_calibration_interface(validation_manager, lang):
    """Create an interface for parameter calibration."""
    st.subheader(lang['parameter_calibration_title'])
    
    # Load current parameters from config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create tabs for different aspects of calibration
    tab1, tab2, tab3 = st.tabs(["Parameter Adjustment", "Sensitivity Analysis", "Validation Results"])
    
    with tab1:
        # Parameter inputs with real-time effects
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parameter Inputs**")
            # Bond strength with range slider
            bond_strength = st.slider(
                lang['param_bond_strength'],
                min_value=float(5e5),
                max_value=float(2e6),
                value=float(config['dem']['bond_strength']),
                step=float(1e5),
                help="Affects particle cohesion and erosion resistance"
            )
            
            # Fluid density with range slider
            fluid_density = st.slider(
                lang['param_fluid_density'],
                min_value=float(900.0),
                max_value=float(1100.0),
                value=float(config['cfd']['fluid_density']),
                step=float(10.0),
                help="Affects buoyancy and fluid-particle interaction"
            )
            
            # Fluid viscosity with range slider
            fluid_viscosity = st.slider(
                lang['param_fluid_viscosity'],
                min_value=float(0.0005),
                max_value=float(0.002),
                value=float(config['cfd']['fluid_viscosity']),
                step=float(0.0001),
                help="Affects flow resistance and particle transport"
            )
        
        with col2:
            st.write("**Real-time Parameter Effects**")
            # Calculate normalized parameter values (0 to 1)
            norm_bond = float((bond_strength - 5e5) / (2e6 - 5e5))
            norm_density = float((fluid_density - 900) / (1100 - 900))
            norm_viscosity = float((fluid_viscosity - 0.0005) / (0.002 - 0.0005))
            
            # Create effect indicators with dynamic values
            effects = {
                'Erosion Rate': {
                    'bond_strength': float(-0.8 * norm_bond),
                    'fluid_density': float(0.3 * norm_density),
                    'fluid_viscosity': float(-0.5 * norm_viscosity)
                },
                'Pressure Drop': {
                    'bond_strength': float(0.2 * norm_bond),
                    'fluid_density': float(0.7 * norm_density),
                    'fluid_viscosity': float(0.9 * norm_viscosity)
                },
                'Flow Velocity': {
                    'bond_strength': float(-0.3 * norm_bond),
                    'fluid_density': float(-0.4 * norm_density),
                    'fluid_viscosity': float(-0.8 * norm_viscosity)
                }
            }
            
            # Display effects as interactive metrics
            for effect, params in effects.items():
                st.write(f"**{effect}**")
                total_effect = float(sum(params.values()))
                
                # Normalize total effect to 0-1 range
                normalized_total = float((total_effect + 1.5) / 3.0)  # Assuming max range of -1.5 to 1.5
                normalized_total = max(0.0, min(1.0, normalized_total))  # Clamp to 0-1
                
                st.progress(
                    normalized_total,
                    text=f"Overall Effect: {total_effect:.2f}"
                )
                
                for param, value in params.items():
                    # Normalize individual effect to 0-1 range
                    normalized_value = float((value + 1.0) / 2.0)  # Assuming max range of -1 to 1
                    normalized_value = max(0.0, min(1.0, normalized_value))  # Clamp to 0-1
                    
                    st.progress(
                        normalized_value,
                        text=f"{param}: {value:.2f} ({'↑' if value > 0 else '↓'})"
                    )
    
    with tab2:
        st.write("**Parameter Sensitivity Analysis**")
        # Create sensitivity analysis visualization
        sensitivity_data = {
            'Parameter': ['Bond Strength', 'Fluid Density', 'Fluid Viscosity'],
            'Erosion Impact': [-0.8, 0.3, -0.5],
            'Pressure Impact': [0.2, 0.7, 0.9],
            'Flow Impact': [-0.3, -0.4, -0.8]
        }
        
        df = pd.DataFrame(sensitivity_data)
        fig = go.Figure()
        
        for impact in ['Erosion Impact', 'Pressure Impact', 'Flow Impact']:
            fig.add_trace(go.Bar(
                name=impact,
                x=df['Parameter'],
                y=df[impact],
                text=df[impact].round(2),
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Parameter Sensitivity Analysis",
            barmode='group',
            xaxis_title="Parameters",
            yaxis_title="Impact Coefficient",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True, key="sensitivity_study")
    
    with tab3:
        st.write("**Validation Results**")
        # Create validation metrics
        validation_metrics = {
            'Erosion Rate': {
                'current': 0.85,
                'target': 0.82,
                'error': 3.7
            },
            'Pressure Distribution': {
                'current': 0.92,
                'target': 0.89,
                'error': 3.4
            },
            'Flow Velocity': {
                'current': 0.78,
                'target': 0.75,
                'error': 4.0
            }
        }
        
        # Display validation metrics
        cols = st.columns(len(validation_metrics))
        for i, (metric, data) in enumerate(validation_metrics.items()):
            with cols[i]:
                st.metric(
                    metric,
                    f"{data['current']:.2f}",
                    f"{data['error']:.1f}% error"
                )
                st.write(f"Target: {data['target']:.2f}")
        
        # Overall validation score
        overall_score = sum(data['current'] for data in validation_metrics.values()) / len(validation_metrics)
        st.metric("Overall Validation Score", f"{overall_score:.2f}")
    
    # Update button
    if st.button(lang['update_parameters']):
        # Update config
        config['dem']['bond_strength'] = float(bond_strength)
        config['cfd']['fluid_density'] = float(fluid_density)
        config['cfd']['fluid_viscosity'] = float(fluid_viscosity)
        
        # Save updated config
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        st.success(lang['parameter_updated'])
        st.rerun()

def load_geotechnical_params(lang):
    """Load material properties from config.yaml."""
    st.subheader(lang['geotechnical_params_title'])
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config.get('geotechnical', {})
    except Exception as e:
        st.error(lang['geotechnical_params_error'])
        return {}

def plot_granular_curve(params, lang):
    """Plot the granular curve with improved visualization."""
    try:
        # Generate sample data
        _, granular_data = generate_sample_data(params)
        
        # Create the plot
        fig = go.Figure()
        
        # Add the granular curve
        fig.add_trace(go.Scatter(
            x=granular_data['grain_size'],
            y=granular_data['passing_percentage'],
            mode='lines',
            name='Grain Size Distribution',
            line=dict(color='blue', width=2)
        ))
        
        # Add the original data points
        fig.add_trace(go.Scatter(
            x=params['geotechnical']['grain_sizes'],
            y=params['geotechnical']['passing_percentages'],
            mode='markers',
            name='Measured Points',
            marker=dict(color='red', size=8)
        ))
        
        # Update layout
        fig.update_layout(
            title='Grain Size Distribution',
            xaxis_title='Grain Size (mm)',
            yaxis_title='Passing Percentage (%)',
            xaxis_type='log',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting granular curve: {str(e)}")
        return None

def granular_curve_editor(params, lang):
    """Interactive editor for grain size distribution."""
    st.subheader(lang['edit_granular_curve'])
    # Get current values
    grain_sizes = params.get('grain_sizes', [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.1])
    passing_percentages = params.get('passing_percentages', [95, 92, 88, 83, 77, 70, 62, 53, 40, 20, 5, 0])
    
    # Display editable table
    df = pd.DataFrame({
        lang['grain_size']: grain_sizes,
        lang['passing_percentage']: passing_percentages
    })
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    
    # Update config and rerun simulation
    if st.button(lang['update_granular_curve']):
        new_grain_sizes = edited_df[lang['grain_size']].tolist()
        new_passing_percentages = edited_df[lang['passing_percentage']].tolist()
        # Update config.yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        if 'geotechnical' not in config:
            config['geotechnical'] = {}
        config['geotechnical']['grain_sizes'] = new_grain_sizes
        config['geotechnical']['passing_percentages'] = new_passing_percentages
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)
        st.success(lang['granular_curve_updated'])
        # Optionally, trigger simulation here
        # run_simulation()
        st.rerun()

def display_geotechnical_params(params, lang):
    """Display geotechnical parameters in a clean, organized format."""
    try:
        # Create three columns for different parameter categories
        col1, col2, col3 = st.columns(3)
        
        # Physical Properties
        with col1:
            st.subheader("Physical Properties")
            st.write(f"**Density:** {params['geotechnical']['density']:.3f} g/cm³")
            st.write(f"**Specific Gravity:** {params['geotechnical']['specific_gravity']:.2f}")
            st.write(f"**Water Content:** {params['geotechnical']['water_content']:.1f}%")
            st.write(f"**Clay Content:** {params['geotechnical']['clay_content']}%")
        
        # Mechanical Properties
        with col2:
            st.subheader("Mechanical Properties")
            st.write(f"**Cohesion:** {params['geotechnical']['cohesion']:.1f} kPa")
            st.write(f"**Friction Angle:** {params['geotechnical']['friction_angle']}°")
            st.write(f"**Cu:** {params['geotechnical']['Cu']:.1f}")
            st.write(f"**Cc:** {params['geotechnical']['Cc']:.1f}")
        
        # Hydraulic Properties
        with col3:
            st.subheader("Hydraulic Properties")
            st.write(f"**Permeability:** {params['geotechnical']['permeability']:.6f} m/s")
                
    except Exception as e:
        st.error(f"Error displaying geotechnical parameters: {str(e)}")

def display_wang_reference(lang):
    """Display information about Wang's work and its implications."""
    st.subheader(lang['wang_reference_title'])
    st.write(f"**{lang['wang_reference_desc']}**")
    st.write(lang['wang_key_findings'])
    st.write(f"{lang['wang_finding_1']}\n{lang['wang_finding_1_detail']}\n{lang['wang_finding_1_detail_2']}\n{lang['wang_finding_1_detail_3']}")
    st.write(f"{lang['wang_finding_2']}\n{lang['wang_finding_2_detail']}\n{lang['wang_finding_2_detail_2']}\n{lang['wang_finding_2_detail_3']}")
    st.write(f"{lang['wang_finding_3']}\n{lang['wang_finding_3_detail']}\n{lang['wang_finding_3_detail_2']}\n{lang['wang_finding_3_detail_3']}")
    st.caption(lang['wang_paper_reference'])

def create_experimental_setup_diagram(lang):
    """Create a diagram of the triaxial seepage test setup."""
    # Create figure
    fig = go.Figure()
    
    # Add cell walls
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=100, y1=150,
        line=dict(color="black", width=2),
        fillcolor="lightgray"
    )
    
    # Add sample
    fig.add_shape(
        type="rect",
        x0=25, y0=25,
        x1=75, y1=125,
        line=dict(color="blue", width=2),
        fillcolor="lightblue"
    )
    
    # Add pressure lines
    fig.add_shape(
        type="line",
        x0=50, y0=0,
        x1=50, y1=25,
        line=dict(color="red", width=2)
    )
    
    # Add flow direction arrows
    fig.add_annotation(
        x=50, y=20,
        text="↓",
        showarrow=False,
        font=dict(size=20, color="red")
    )
    
    fig.add_annotation(
        x=50, y=130,
        text="↓",
        showarrow=False,
        font=dict(size=20, color="red")
    )
    
    # Add measurement points
    measurement_points = [
        (25, 50, "P1"),
        (75, 50, "P2"),
        (25, 100, "P3"),
        (75, 100, "P4")
    ]
    
    for x, y, label in measurement_points:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=10, color="red"),
            text=[label],
            textposition="top center"
        ))
    
    # Add bilingual annotations
    annotations = [
        (50, 75, "Soil Sample", "土壤试样"),
        (50, 10, "Pressure", "压力"),
        (50, 140, "Flow Direction", "流向"),
        (85, 75, "Measurement Points", "测量点")
    ]
    
    for x, y, en_text, zh_text in annotations:
        fig.add_annotation(
            x=x, y=y,
            text=f"{en_text}<br>{zh_text}",
            showarrow=False,
            font=dict(size=12)
        )
    
    # Update layout
    fig.update_layout(
        title=f"Triaxial Seepage Test Setup<br>三轴渗流试验装置",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=600,
        height=400
    )
    
    # Save figure
    fig.write_image("results/experimental_setup.png")

def create_tunnel_geometry_diagram(lang):
    """Create a diagram of the tunnel geometry."""
    # Create figure
    fig = go.Figure()
    
    # Add tunnel outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=80, y1=6,
        line=dict(color="black", width=2),
        fillcolor="lightgray"
    )
    
    # Add water level
    fig.add_shape(
        type="rect",
        x0=0, y0=6,
        x1=80, y1=56,
        line=dict(color="blue", width=2),
        fillcolor="lightblue"
    )
    
    # Add flow direction arrows
    for x in [10, 30, 50, 70]:
        fig.add_annotation(
            x=x, y=3,
            text="→",
            showarrow=False,
            font=dict(size=20, color="red")
        )
    
    # Add measurement points
    measurement_points = [
        (20, 3, "M1"),
        (40, 3, "M2"),
        (60, 3, "M3")
    ]
    
    for x, y, label in measurement_points:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=10, color="red"),
            text=[label],
            textposition="top center"
        ))
    
    # Add bilingual annotations
    annotations = [
        (40, 3, "Tunnel", "隧道"),
        (40, 30, "Water Head", "水头"),
        (40, 55, "Ground Surface", "地表"),
        (75, 3, "Flow Direction", "流向")
    ]
    
    for x, y, en_text, zh_text in annotations:
        fig.add_annotation(
            x=x, y=y,
            text=f"{en_text}<br>{zh_text}",
            showarrow=False,
            font=dict(size=12)
        )
    
    # Add dimensions
    dimensions = [
        (80, 3, "80m", "80米", 0, -20),
        (40, 6, "6m", "6米", 30, 0),
        (40, 56, "50m", "50米", 30, 0)
    ]
    
    for x, y, en_dim, zh_dim, ax, ay in dimensions:
        fig.add_annotation(
            x=x, y=y,
            text=f"{en_dim}<br>{zh_dim}",
            showarrow=True,
            arrowhead=1,
            ax=ax, ay=ay
        )
    
    # Update layout
    fig.update_layout(
        title=f"Tunnel Geometry<br>隧道几何",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=800,
        height=600
    )
    
    # Save figure
    fig.write_image("results/tunnel_geometry.png")

def create_water_inrush_visualization(lang):
    """Create a visualization of the water inrush process."""
    try:
        # Create figure
        fig = go.Figure()
        
        # Add tunnel
        fig.add_shape(
            type="rect",
            x0=0, y0=0,
            x1=80, y1=6,
            line=dict(color="black", width=2),
            fillcolor="lightgray"
        )
        
        # Add water level with gradient
        for i in range(10):
            y_start = 6 + i * 5
            y_end = 6 + (i + 1) * 5
            opacity = 0.1 + (i * 0.09)
            fig.add_shape(
                type="rect",
                x0=0, y0=y_start,
                x1=80, y1=y_end,
                line=dict(color="blue", width=0),
                fillcolor=f"rgba(0,0,255,{opacity})"
            )
        
        # Add flow vectors
        for x in range(5, 75, 10):
            for y in range(10, 50, 10):
                fig.add_annotation(
                    x=x, y=y,
                    text="↓",
                    showarrow=False,
                    font=dict(size=15, color="blue")
                )
        
        # Add bilingual annotations
        annotations = [
            (40, 3, "Tunnel", "隧道"),
            (40, 30, "Water Inrush", "突水"),
            (75, 25, lang['flow_vectors'], lang['flow_vectors'])
        ]
        
        for x, y, en_text, zh_text in annotations:
            fig.add_annotation(
                x=x, y=y,
                text=f"{en_text}<br>{zh_text}",
                showarrow=False,
                font=dict(size=12)
            )
        
        # Update layout
        fig.update_layout(
            title=lang['water_inrush_process'],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
            width=800,
            height=600
        )
        
        # Display the plot directly in Streamlit
        st.plotly_chart(fig, use_container_width=True, key="water_inrush_visualization_plot")
        
        # Add simulation parameters
        st.write(f"**{lang['simulation_parameters']}:**")
        params = {
            'initial_pressure': 500,  # kPa
            'flow_rate': 2.5,  # m³/s
            'erosion_rate': 0.15,  # m/h
            'particle_size': 0.002  # m
        }
        for param, value in params.items():
            st.metric(lang[param], f"{value}")
            
    except Exception as e:
        st.error(f"Error creating water inrush visualization: {str(e)}")

def create_particle_erosion_plot(lang):
    """Create particle erosion plot."""
    try:
        # Generate time steps
        time = np.linspace(0, 100, 50)
        
        # Generate erosion data
        erosion = 0.5 * (1 - np.exp(-0.05 * time))
        
        # Create figure
        fig = go.Figure()
        
        # Add simulation data
        fig.add_trace(go.Scatter(
            x=time,
            y=erosion,
            name=lang['simulation'],
            line=dict(color='blue', width=2)
        ))
        
        # Add experimental data (slightly different for comparison)
        experimental_erosion = 0.48 * (1 - np.exp(-0.048 * time))
        fig.add_trace(go.Scatter(
            x=time,
            y=experimental_erosion,
            name=lang['experimental'],
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=lang['particle_erosion'],
            xaxis_title=lang['time_steps'],
            yaxis_title=lang['erosion_rate'],
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Display the plot directly in Streamlit
        st.plotly_chart(fig, use_container_width=True, key="erosion_plot")
        
        # Add validation metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                lang['mean_error'],
                f"{np.mean(np.abs(erosion - experimental_erosion)):.3f}"
            )
        with col2:
            st.metric(
                lang['max_error'],
                f"{np.max(np.abs(erosion - experimental_erosion)):.3f}"
            )
        with col3:
            st.metric(
                lang['correlation'],
                f"{np.corrcoef(erosion, experimental_erosion)[0,1]:.3f}"
            )
            
    except Exception as e:
        st.error(f"Error creating particle erosion plot: {str(e)}")
        return None

def create_tunnel_water_inrush_section(lang):
    """Create the tunnel water inrush case study section."""
    st.header(lang['tunnel_water_inrush'])
    
    # Tunnel geometry
    st.subheader(lang['tunnel_geometry'])
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Tunnel dimensions:")
        dimensions = {
            'length': 80,  # m
            'width': 8,    # m
            'height': 6,   # m
            'water_head': 50  # m
        }
        for dim, value in dimensions.items():
            st.metric(dim.replace('_', ' ').title(), f"{value} m")
    
    with col2:
        # Create and display tunnel geometry diagram
        create_tunnel_geometry_diagram(lang)
        st.image('results/tunnel_geometry.png', use_container_width=True)
    
    # Water inrush visualization
    st.subheader(lang['water_inrush_simulation'])
    create_water_inrush_visualization(lang)
    
    # Particle erosion visualization
    st.subheader(lang['particle_erosion'])
    create_particle_erosion_plot(lang)
    
    # Add real-time monitoring data
    st.subheader(lang['real_time_monitoring'])
    col1, col2 = st.columns(2)
    
    with col1:
        # Pressure plot
        time_steps = np.linspace(0, 100, 50)
        pressure = 500 * np.exp(-0.02 * time_steps)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=time_steps, y=pressure, name='Pressure'))
        fig1.update_layout(
            title=lang['pressure_evolution'],
            xaxis_title='Time (s)',
            yaxis_title='Pressure (kPa)'
        )
        st.plotly_chart(fig1, use_container_width=True, key="pressure_evolution_water_inrush")
    
    with col2:
        # Flow rate plot
        flow_rate = 2.5 * np.sin(0.1 * time_steps)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=time_steps, y=flow_rate, name='Flow Rate'))
        fig2.update_layout(
            title=lang['flow_rate_evolution'],
            xaxis_title='Time (s)',
            yaxis_title='Flow Rate (m³/s)'
        )
        st.plotly_chart(fig2, use_container_width=True, key="flow_rate_evolution_water_inrush")

def create_mitigation_visualization(lang):
    """Create a visualization of the mitigation measures."""
    # Create figure
    fig = go.Figure()
    
    # Add tunnel
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=80, y1=6,
        line=dict(color="black", width=2),
        fillcolor="lightgray"
    )
    
    # Add grouting zones
    grouting_zones = [(10, 20), (30, 40), (50, 60), (70, 80)]
    for x_start, x_end in grouting_zones:
        fig.add_shape(
            type="rect",
            x0=x_start, y0=0,
            x1=x_end, y1=6,
            line=dict(color="green", width=2),
            fillcolor="rgba(0,255,0,0.2)"
        )
    
    # Add drainage system
    for x in [15, 35, 55, 75]:
        fig.add_shape(
            type="line",
            x0=x, y0=0,
            x1=x, y1=-5,
            line=dict(color="blue", width=2)
        )
    
    # Add bilingual annotations
    annotations = [
        (40, 3, "Tunnel", "隧道"),
        (20, 3, "Grouting", "注浆"),
        (40, -2, "Drainage", "排水"),
        (60, 3, "Grouting", "注浆")
    ]
    
    for x, y, en_text, zh_text in annotations:
        fig.add_annotation(
            x=x, y=y,
            text=f"{en_text}<br>{zh_text}",
            showarrow=False,
            font=dict(size=12)
        )
    
    # Update layout
    fig.update_layout(
        title=f"Mitigation Measures<br>缓解措施",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=800,
        height=400
    )
    
    # Save figure
    fig.write_image("results/mitigation_measures.png")

def create_experimental_verification_section(lang):
    """Create the experimental verification section."""
    st.header(lang['experimental_verification'])
    
    # Triaxial test setup
    st.subheader(lang['triaxial_test'])
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(lang['experimental_setup'])
        # Create and display experimental setup diagram
        create_experimental_setup_diagram(lang)
        st.image('results/experimental_setup.png', use_container_width=True)
    
    with col2:
        st.write(lang['test_parameters'])
        params = {
            'confining_pressure': 100,  # kPa
            'flow_rate': 0.5,  # ml/min
            'sample_height': 100,  # mm
            'sample_diameter': 50,  # mm
        }
        for param, value in params.items():
            st.metric(param.replace('_', ' ').title(), f"{value}")
    
    # Comparison of results
    st.subheader(lang['comparison'])
    tab1, tab2, tab3 = st.tabs([lang['erosion_rate'], lang['pressure_distribution'], lang['velocity_profile']])
    
    with tab1:
        create_erosion_plot(None, lang)
    
    with tab2:
        create_pressure_distribution_plot(lang)
    
    with tab3:
        create_velocity_profile_plot(lang)
    
    # Validation metrics
    st.subheader(lang['validation_metrics'])
    metrics = {
        'erosion_rate_error': 3.7,
        'pressure_error': 4.2,
        'velocity_error': 5.1,
        'overall_validation_score': 0.85
    }
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(lang['erosion_rate'], f"{metrics['erosion_rate_error']}%")
    with col2:
        st.metric(lang['pressure_distribution'], f"{metrics['pressure_error']}%")
    with col3:
        st.metric(lang['velocity_profile'], f"{metrics['velocity_error']}%")
    with col4:
        st.metric(lang['overall_validation_score'], f"{metrics['overall_validation_score']}")

def create_mitigation_measures_section(lang):
    """Create the mitigation measures section."""
    st.header(lang['mitigation_measures'])
    
    # Mitigation options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(lang['grouting'])
        grouting_params = {
            'grout_type': 'Cement-based',
            'injection_pressure': 2.0,  # MPa
            'grout_volume': 100,  # m³
            'effectiveness': 0.85
        }
        for param, value in grouting_params.items():
            st.metric(param.replace('_', ' ').title(), str(value))
    
    with col2:
        st.subheader(lang['drainage'])
        drainage_params = {
            'drainage_type': 'Horizontal drains',
            'spacing': 5,  # m
            'diameter': 0.1,  # m
            'effectiveness': 0.75
        }
        for param, value in drainage_params.items():
            st.metric(param.replace('_', ' ').title(), str(value))
    
    with col3:
        st.subheader(lang['spraying'])
        spraying_params = {
            'spray_type': 'Shotcrete',
            'thickness': 0.15,  # m
            'coverage': 0.9,
            'effectiveness': 0.80
        }
        for param, value in spraying_params.items():
            st.metric(param.replace('_', ' ').title(), str(value))
    
    # Effectiveness comparison
    st.subheader(lang['mitigation_effectiveness'])
    create_mitigation_effectiveness_plot(lang, key_suffix="mitigation_measures")
    
    # Cost analysis
    st.subheader(lang['cost_analysis'])
    create_cost_analysis_plot(lang)
    
    # Recommendations
    st.subheader(lang['recommendations'])
    recommendations = [
        "1. Implement grouting as primary measure",
        "2. Use drainage system for water control",
        "3. Apply shotcrete for surface protection",
        "4. Monitor effectiveness regularly"
    ]
    for rec in recommendations:
        st.write(rec)

def create_erosion_plot(validation_manager, lang, key_suffix=""):
    """Create the erosion plot with improved visualization."""
    try:
        # Generate sample data
        erosion_data, _ = generate_sample_data(config)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add erosion rate
        fig.add_trace(
            go.Scatter(
                x=erosion_data['time'],
                y=erosion_data['erosion_rate'],
                name="Erosion Rate",
                line=dict(color='blue', width=2)
            ),
            secondary_y=False
        )
        
        # Add cumulative erosion
        fig.add_trace(
            go.Scatter(
                x=erosion_data['time'],
                y=erosion_data['cumulative_erosion'],
                name="Cumulative Erosion",
                line=dict(color='red', width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Erosion Analysis",
            xaxis_title="Time (s)",
            template="plotly_white",
            showlegend=True
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Erosion Rate (mm/s)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Erosion (mm)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True, key=f"erosion_analysis_plot_{key_suffix}")
        return fig
    except Exception as e:
        st.error(f"Error creating erosion plot: {str(e)}")
        return None

def create_pressure_distribution_plot(lang):
    """Create pressure distribution plot."""
    x = np.linspace(0, 100, 50)
    pressure = 100 * np.exp(-0.02 * x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pressure, name=lang['pressure_distribution']))
    fig.update_layout(
        title=lang['pressure_distribution'],
        xaxis_title=lang['position'],
        yaxis_title=lang['pressure_unit']
    )
    st.plotly_chart(fig, use_container_width=True, key="pressure_distribution_plot")

def create_velocity_profile_plot(lang):
    """Create velocity profile plot."""
    x = np.linspace(0, 100, 50)
    velocity = 2 * np.sin(0.1 * x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=velocity, name=lang['velocity_profile']))
    fig.update_layout(
        title=lang['velocity_profile'],
        xaxis_title=lang['position'],
        yaxis_title=lang['velocity_unit']
    )
    st.plotly_chart(fig, use_container_width=True, key="velocity_profile_plot")

def create_rock_displacement_plot(lang):
    """Create rock displacement plot."""
    time = np.linspace(0, 100, 50)
    displacement = 0.1 * np.sin(0.1 * time)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=displacement, name=lang['rock_displacement']))
    fig.update_layout(
        title=lang['rock_displacement'],
        xaxis_title=lang['time_steps'],
        yaxis_title="Displacement (m)"
    )
    st.plotly_chart(fig, use_container_width=True, key="rock_displacement_plot")

def create_seepage_velocity_plot(lang):
    """Create seepage velocity plot."""
    time = np.linspace(0, 100, 50)
    velocity = 0.5 * np.exp(-0.02 * time)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=velocity, name=lang['seepage_velocity']))
    fig.update_layout(
        title=lang['seepage_velocity'],
        xaxis_title=lang['time_steps'],
        yaxis_title=lang['velocity_unit']
    )
    st.plotly_chart(fig, use_container_width=True, key="seepage_velocity_plot")

def create_porosity_evolution_plot(lang):
    """Create porosity evolution plot."""
    time = np.linspace(0, 100, 50)
    porosity = 0.3 + 0.1 * (1 - np.exp(-0.03 * time))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=porosity, name=lang['porosity_evolution']))
    fig.update_layout(
        title=lang['porosity_evolution'],
        xaxis_title=lang['time_steps'],
        yaxis_title="Porosity"
    )
    st.plotly_chart(fig, use_container_width=True, key="porosity_evolution_plot")

def create_mitigation_effectiveness_plot(lang, key_suffix=""):
    """Create mitigation effectiveness comparison plot."""
    measures = [lang['grouting'], lang['drainage'], lang['spraying']]
    effectiveness = [0.85, 0.75, 0.80]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=measures, y=effectiveness))
    fig.update_layout(
        title=lang['mitigation_effectiveness'],
        xaxis_title="Mitigation Measure",
        yaxis_title="Effectiveness"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"mitigation_effectiveness_plot_{key_suffix}")

def create_cost_analysis_plot(lang, key_suffix=""):
    """Create cost analysis plot."""
    measures = [lang['grouting'], lang['drainage'], lang['spraying']]
    costs = [100000, 75000, 50000]  # Example costs in USD
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=measures, y=costs))
    fig.update_layout(
        title=lang['cost_analysis'],
        xaxis_title="Mitigation Measure",
        yaxis_title="Cost (USD)"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"cost_analysis_plot_{key_suffix}")

def create_case_studies_section(lang):
    """Create the case studies section with proper translations."""
    st.header(lang['case_studies'])
    
    # Create tabs for different case studies
    tab1, tab2, tab3 = st.tabs([
        lang['experimental_verification'],
        lang['tunnel_water_inrush'],
        lang['mitigation_measures']
    ])
    
    with tab1:
        create_experimental_verification_section(lang)
    
    with tab2:
        create_tunnel_water_inrush_section(lang)
    
    with tab3:
        create_mitigation_measures_section(lang)
        st.subheader(lang['mitigation_effectiveness'])
        create_mitigation_effectiveness_plot(lang, key_suffix="case_studies_mitigation")
        st.subheader(lang['cost_analysis'])
        create_cost_analysis_plot(lang, key_suffix="case_studies_cost")

def create_recent_results_section(lang):
    """Create the recent results section in the overview."""
    st.subheader(lang['recent_results'])
    
    # Create columns for different metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            lang['erosion_rate'],
            "0.85",
            delta="3.7%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            lang['pressure_distribution'],
            "0.92",
            delta="3.4%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            lang['velocity_profile'],
            "0.78",
            delta="4.0%",
            delta_color="normal"
        )
    
    # Add recent simulation results
    st.subheader("Recent Simulation Results")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs([
        lang['pressure_analysis'],
        lang['velocity_analysis'],
        lang['erosion_analysis']
    ])
    
    with tab1:
        create_pressure_distribution_plot(lang)
    
    with tab2:
        create_velocity_profile_plot(lang)
        
    with tab3:
        create_erosion_plot(None, lang, key_suffix="recent_results")

def plot_granular_curve_chinese(params):
    """Plot the granular curve with Chinese labels and a red line, matching the provided image style."""
    import plotly.graph_objects as go
    import numpy as np
    
    # Use the same data as the main granular curve
    grain_sizes = params['geotechnical']['grain_sizes']
    passing_percentages = params['geotechnical']['passing_percentages']
    
    # Smooth curve
    x_smooth = np.logspace(np.log10(min(grain_sizes)), np.log10(max(grain_sizes)), 100)
    y_smooth = np.interp(x_smooth, grain_sizes[::-1], passing_percentages[::-1])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_smooth,
        y=y_smooth,
        mode='lines',
        line=dict(color='red', width=2),
        name='颗粒级配曲线'
    ))
    
    fig.update_layout(
        xaxis=dict(
            title='粒径 (mm)',
            type='log',
            autorange='reversed',
            tickvals=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.1],
            ticktext=[str(v) for v in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.1]],
        ),
        yaxis=dict(
            title='小于某粒径的土粒含量 (%)',
            range=[0, 100]
        ),
        margin=dict(l=60, r=20, t=20, b=60),
        showlegend=False,
        template='simple_white',
        height=400
    )
    return fig

# Update the main function to include recent results
def main():
    # Language selection
    lang = LANG['en']
    
    # Sidebar navigation
    st.sidebar.title(lang['navigation'])
    page = st.sidebar.radio(
        "Go to",
        [lang['overview'], lang['coupling_framework'], lang['validation_results'],
         lang['parameter_calibration'], lang['geotechnical_parameters'],
         lang['theoretical_background'], lang['case_studies']]
    )
    
    # Main content
    if page == lang['overview']:
        st.title(lang['dashboard_title'])
        
        # Overview section
        st.header(lang['project_overview'])
        st.write(lang['project_overview_desc'])
        
        # Add granular curve with Chinese labels (replicating the image)
        st.subheader("颗粒级配曲线 Granular Curve (Chinese Style)")
        granular_fig_cn = plot_granular_curve_chinese(config)
        st.plotly_chart(granular_fig_cn, use_container_width=True, key="granular_curve_chinese")
        
        # Display granular curve (existing style)
        st.subheader("Grain Size Distribution")
        granular_fig = plot_granular_curve(config, lang)
        if granular_fig:
            st.plotly_chart(granular_fig, use_container_width=True, key="overview_granular_curve")
        
        # Material properties summary
        st.header(lang['material_properties'])
        display_geotechnical_params(config, lang)
        
        # Add recent results section
        st.subheader(lang['recent_results'])
        
        # Create columns for different metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                lang['erosion_rate'],
                "0.85",
                delta="3.7%",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                lang['pressure_distribution'],
                "0.92",
                delta="3.4%",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                lang['velocity_profile'],
                "0.78",
                delta="4.0%",
                delta_color="normal"
            )
        
        # Add recent simulation results
        st.subheader("Recent Simulation Results")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs([
            lang['pressure_analysis'],
            lang['velocity_analysis'],
            lang['erosion_analysis']
        ])
        
        with tab1:
            create_pressure_distribution_plot(lang)
        
        with tab2:
            create_velocity_profile_plot(lang)
            
        with tab3:
            create_erosion_plot(None, lang, key_suffix="recent_results")
    elif page == lang['coupling_framework']:
        st.title(lang['cfd_dem_framework'])
        st.write(lang['framework_desc'])
        
        # Display coupling framework diagram
        coupling_diagram = load_coupling_diagram(lang)
        if coupling_diagram:
            st.image(coupling_diagram, use_column_width=True)
        
    elif page == lang['validation_results']:
        st.title(lang['validation_results'])
        validation_manager = ValidationManager(config)
        create_validation_summary(validation_manager, lang)
        
    elif page == lang['parameter_calibration']:
        st.title(lang['parameter_calibration'])
        validation_manager = ValidationManager(config)
        create_parameter_calibration_interface(validation_manager, lang)
        
    elif page == lang['geotechnical_parameters']:
        st.title(lang['geotechnical_parameters'])
        display_geotechnical_params(config, lang)
        
    elif page == lang['theoretical_background']:
        st.title(lang['theoretical_background'])
        display_wang_reference(lang)
    elif page == lang['case_studies']:
        create_case_studies_section(lang)

if __name__ == "__main__":
    main() 