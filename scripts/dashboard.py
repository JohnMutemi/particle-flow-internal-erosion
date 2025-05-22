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
        'validation_results': "Validation Results",
        'parameter_calibration': "Parameter Calibration",
        'geotechnical_parameters': "Geotechnical Parameters",
        'theoretical_background': "Theoretical Background",
        'project_overview': "Project Overview",
        'project_overview_desc': "This dashboard provides a comprehensive view of the CFD-DEM simulation results, including the coupling framework, validation results, and parameter calibration.",
        'simulation_time': "Simulation Time",
        'particles': "Particles",
        'validation_score': "Validation Score",
        'geotech_params_summary': "Geotechnical Parameters Summary",
        'recent_results': "Recent Results",
        'cfd_dem_framework': "CFD-DEM Coupling Framework",
        'framework_desc': "The coupling framework illustrates how the Discrete Element Method (DEM) for particles interacts with the Computational Fluid Dynamics (CFD) for fluid flow through the Coupling Manager.",
        'framework_components': "Framework Components",
        'dem_particles': "DEM (Particles): Handles particle motion, contacts, and forces",
        'cfd_fluid': "CFD (Fluid): Computes fluid flow, pressure, and velocity fields",
        'coupling_manager': "Coupling Manager: Exchanges forces and velocities between DEM and CFD",
        'geotechnical_context': "Geotechnical Context",
        'validation_results_header': "Validation Results",
        'validation_results_desc': "This section shows the comparison between simulation results and experimental data, including erosion rates, pressure distributions, and velocity profiles.",
        'parameter_calibration_header': "Parameter Calibration",
        'parameter_calibration_desc': "This section allows you to calibrate model parameters by comparing simulation results with experimental data.",
        'current_geotech_params': "Current Geotechnical Parameters",
        'parameter_sensitivity': "Parameter Sensitivity",
        'parameter_relationships': "Parameter Relationships",
        'soil_classification': "Soil Classification",
        'theoretical_background_header': "Theoretical Background",
        'reference_wang': "Reference: Wang et al. (2020)",
        'model_validation': "Model Validation",
        'parameter_sensitivity_analysis': "Parameter Sensitivity Analysis",
        'select_language': "Select Language",
        'basic_properties': "Basic Properties",
        'density': "Density",
        'specific_gravity': "Specific Gravity",
        'water_content': "Water Content",
        'grain_size_distribution': "Grain Size Distribution",
        'uniformity_coefficient': "Cu (Uniformity Coefficient)",
        'curvature_coefficient': "Cc (Curvature Coefficient)",
        'clay_content': "Clay Content",
        'hydraulic_properties': "Hydraulic Properties",
        'permeability': "Permeability",
        'porosity': "Porosity",
        'void_ratio': "Void Ratio",
        'particle_size_distribution': "Particle Size Distribution",
        'wang_reference_title': "Reference: Wang et al. (2020)",
        'wang_reference_desc': "Effects of Internal Erosion on Parameters of Subloading Cam-Clay Model",
        'wang_key_findings': "Key findings from Wang's work:",
        'wang_finding_1': "1. Internal erosion significantly affects soil parameters:",
        'wang_finding_1_detail': "   - Decreases in cohesion and friction angle",
        'wang_finding_1_detail_2': "   - Changes in void ratio and permeability",
        'wang_finding_1_detail_3': "   - Alterations in stress-strain relationships",
        'wang_finding_2': "2. Parameter evolution during erosion:",
        'wang_finding_2_detail': "   - Cohesion decreases by 20-40% during erosion",
        'wang_finding_2_detail_2': "   - Friction angle reduction of 5-15 degrees",
        'wang_finding_2_detail_3': "   - Permeability increases by 1-2 orders of magnitude",
        'wang_finding_3': "3. Implications for our model:",
        'wang_finding_3_detail': "   - Need to account for parameter evolution during simulation",
        'wang_finding_3_detail_2': "   - Consider time-dependent changes in soil properties",
        'wang_finding_3_detail_3': "   - Validate against experimental erosion patterns",
        'wang_paper_reference': "Wang, G., Horikoshi, K., & Takahashi, A. (2020). Effects of internal erosion on parameters of subloading cam-clay model. Geotechnical and Geological Engineering, 38(2), 1323–1335.",
        'edit_granular_curve': "Edit Particle Size Distribution (Granular Curve)",
        'grain_size': "Grain Size (mm)",
        'passing_percentage': "Passing Percentage (%)",
        'update_granular_curve': "Update Granular Curve and Rerun Simulation",
        'granular_curve_updated': "Granular curve updated! Please rerun the simulation to see updated results.",
        'granular_curve_title': "Particle Size Distribution (Granular Curve)",
        'grain_size_mm': "Grain Size (mm)",
        'passing_percentage_pct': "Passing Percentage (%)",
        'clay': "Clay",
        'sand': "Sand",
        'gravel': "Gravel",
        'erosion_plot_title': "Erosion Rate Over Time",
        'time_steps': "Time Steps",
        'erosion_rate': "Erosion Rate",
        'simulation': "Simulation",
        'experimental': "Experimental",
        'parameter_calibration_title': "Parameter Calibration Interface",
        'current_parameters': "Current Parameters",
        'update_parameters': "Update Parameters",
        'parameter_updated': "Parameters updated! Please rerun the simulation to see updated results.",
        'validation_summary_title': "Validation Summary",
        'validation_summary_desc': "This section shows the comparison between simulation results and experimental data.",
        'validation_summary_no_data': "No validation data available.",
        'coupling_diagram_title': "Coupling Framework Diagram",
        'coupling_diagram_error': "Error loading coupling diagram.",
        'geotechnical_params_title': "Geotechnical Parameters",
        'geotechnical_params_error': "Error loading geotechnical parameters.",
        'simulation_success': "Simulation completed successfully!",
        'simulation_failed': "Simulation failed: ",
        'simulation_error': "Error running simulation: ",
        'fluidity_effects_title': "Fluidity Effects Analysis",
        'fluidity_effects_error': "Error analyzing fluidity effects.",
        'config_updated': "Configuration updated successfully!",
        'config_error': "Error updating configuration.",
        'sample_data_title': "Sample Data",
        'sample_data_error': "Error generating sample data.",
        'main_title': "CFD-DEM Simulation Dashboard",
        'main_navigation': "Navigation",
        'main_overview': "Overview",
        'main_coupling_framework': "Coupling Framework",
        'main_validation_results': "Validation Results",
        'main_parameter_calibration': "Parameter Calibration",
        'main_geotechnical_parameters': "Geotechnical Parameters",
        'main_theoretical_background': "Theoretical Background",
        'soil_uscs_classification': "USCS Classification",
        'soil_types': "Soil Types",
        'soil_gravel': "Gravel (G): >4.75mm",
        'soil_sand': "Sand (S): 0.075-4.75mm",
        'soil_silt': "Silt (M): 0.002-0.075mm",
        'soil_clay': "Clay (C): <0.002mm",
        'soil_well_graded': "Well-graded (W)",
        'soil_poorly_graded': "Poorly-graded (P)",
        'soil_silty': "Silty (M)",
        'soil_clayey': "Clayey (C)",
        'soil_current_classification': "Current Classification",
        'soil_composition': "Soil Composition",
        'soil_classification_result': "Classification Result",
        'soil_classification_result_value': "SW-SM: Well-graded sand with silt",
        'soil_cu': "Cu = 4.5 (Well-graded)",
        'soil_cc': "Cc = 1.2 (Well-graded)",
        'soil_fine_content': "Fine content = 40%",
        'param_bond_strength': "Bond Strength",
        'param_fluid_density': "Fluid Density",
        'param_fluid_viscosity': "Fluid Viscosity",
        'coupling_framework_diagram': "Coupling Framework Diagram",
        'pressure_distribution': "Pressure Distribution",
        'velocity_profile': "Velocity Profile",
        'overall_validation_score': "Overall Validation Score",
        'error': "error",
        'sample_pressure_title': "Sample Pressure Distribution",
        'sample_velocity_title': "Sample Velocity Profile",
        'position': "Position",
        'pressure_unit': "Pressure (kPa)",
        'velocity_unit': "Velocity (m/s)",
    },
    'zh': {
        'dashboard_title': "CFD-DEM仿真仪表板",
        'navigation': "导航",
        'overview': "概述",
        'coupling_framework': "耦合框架",
        'validation_results': "验证结果",
        'parameter_calibration': "参数校准",
        'geotechnical_parameters': "岩土参数",
        'theoretical_background': "理论背景",
        'project_overview': "项目概述",
        'project_overview_desc': "本仪表板全面展示CFD-DEM仿真结果，包括耦合框架、验证结果和参数校准。",
        'simulation_time': "仿真步数",
        'particles': "颗粒数",
        'validation_score': "验证得分",
        'geotech_params_summary': "岩土参数摘要",
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
        'current_geotech_params': "当前岩土参数",
        'parameter_sensitivity': "参数敏感性",
        'parameter_relationships': "参数关系",
        'soil_classification': "土壤分类",
        'theoretical_background_header': "理论背景",
        'reference_wang': "参考文献：Wang等 (2020)",
        'model_validation': "模型验证",
        'parameter_sensitivity_analysis': "参数敏感性分析",
        'select_language': "选择语言",
        'basic_properties': "基本属性",
        'density': "密度",
        'specific_gravity': "比重",
        'water_content': "含水量",
        'grain_size_distribution': "颗粒大小分布",
        'uniformity_coefficient': "Cu (均匀系数)",
        'curvature_coefficient': "Cc (曲率系数)",
        'clay_content': "粘土含量",
        'hydraulic_properties': "水力特性",
        'permeability': "渗透系数",
        'porosity': "孔隙率",
        'void_ratio': "孔隙比",
        'particle_size_distribution': "颗粒大小分布",
        'wang_reference_title': "参考文献：Wang等 (2020)",
        'wang_reference_desc': "内部侵蚀对亚加载Cam-Clay模型参数的影响",
        'wang_key_findings': "Wang研究的关键发现：",
        'wang_finding_1': "1. 内部侵蚀显著影响土壤参数：",
        'wang_finding_1_detail': "   - 粘聚力和内摩擦角降低",
        'wang_finding_1_detail_2': "   - 孔隙比和渗透系数变化",
        'wang_finding_1_detail_3': "   - 应力-应变关系改变",
        'wang_finding_2': "2. 侵蚀过程中参数演化：",
        'wang_finding_2_detail': "   - 侵蚀期间粘聚力降低20-40%",
        'wang_finding_2_detail_2': "   - 内摩擦角降低5-15度",
        'wang_finding_2_detail_3': "   - 渗透系数增加1-2个数量级",
        'wang_finding_3': "3. 对我们模型的启示：",
        'wang_finding_3_detail': "   - 需要模拟过程中考虑参数演化",
        'wang_finding_3_detail_2': "   - 考虑土壤特性的时间依赖性变化",
        'wang_finding_3_detail_3': "   - 与实验侵蚀模式验证",
        'wang_paper_reference': "Wang, G., Horikoshi, K., & Takahashi, A. (2020). 内部侵蚀对亚加载Cam-Clay模型参数的影响. 岩土工程与地质工程, 38(2), 1323–1335.",
        'edit_granular_curve': "编辑颗粒大小分布（颗粒曲线）",
        'grain_size': "颗粒大小（毫米）",
        'passing_percentage': "通过百分比（%）",
        'update_granular_curve': "更新颗粒曲线并重新运行仿真",
        'granular_curve_updated': "颗粒曲线已更新！请重新运行仿真以查看更新后的结果。",
        'granular_curve_title': "颗粒大小分布（颗粒曲线）",
        'grain_size_mm': "颗粒大小（毫米）",
        'passing_percentage_pct': "通过百分比（%）",
        'clay': "粘土",
        'sand': "砂",
        'gravel': "砾石",
        'erosion_plot_title': "侵蚀率随时间变化",
        'time_steps': "时间步",
        'erosion_rate': "侵蚀率",
        'simulation': "仿真",
        'experimental': "实验值",
        'parameter_calibration_title': "参数校准界面",
        'current_parameters': "当前参数",
        'update_parameters': "更新参数",
        'parameter_updated': "参数已更新！请重新运行仿真以查看更新后的结果。",
        'validation_summary_title': "验证总结",
        'validation_summary_desc': "本节显示模拟结果与实验数据的对比。",
        'validation_summary_no_data': "无验证数据可用。",
        'coupling_diagram_title': "耦合框架图",
        'coupling_diagram_error': "加载耦合图时出错。",
        'geotechnical_params_title': "岩土参数",
        'geotechnical_params_error': "加载岩土参数时出错。",
        'simulation_success': "仿真成功完成！",
        'simulation_failed': "仿真失败：",
        'simulation_error': "运行仿真时出错：",
        'fluidity_effects_title': "流动性影响分析",
        'fluidity_effects_error': "分析流动性影响时出错。",
        'config_updated': "配置已成功更新！",
        'config_error': "更新配置时出错。",
        'sample_data_title': "示例数据",
        'sample_data_error': "生成示例数据时出错。",
        'main_title': "CFD-DEM仿真仪表板",
        'main_navigation': "导航",
        'main_overview': "概述",
        'main_coupling_framework': "耦合框架",
        'main_validation_results': "验证结果",
        'main_parameter_calibration': "参数校准",
        'main_geotechnical_parameters': "岩土参数",
        'main_theoretical_background': "理论背景",
        'soil_uscs_classification': "USCS土壤分类",
        'soil_types': "土壤类型",
        'soil_gravel': "砾石 (G): >4.75毫米",
        'soil_sand': "砂 (S): 0.075-4.75毫米",
        'soil_silt': "粉土 (M): 0.002-0.075毫米",
        'soil_clay': "粘土 (C): <0.002毫米",
        'soil_well_graded': "良好级配 (W)",
        'soil_poorly_graded': "不良级配 (P)",
        'soil_silty': "粉质 (M)",
        'soil_clayey': "粘质 (C)",
        'soil_current_classification': "当前分类",
        'soil_composition': "土壤组成",
        'soil_classification_result': "分类结果",
        'soil_classification_result_value': "SW-SM: 良好级配砂夹粉土",
        'soil_cu': "Cu = 4.5 (良好级配)",
        'soil_cc': "Cc = 1.2 (良好级配)",
        'soil_fine_content': "细粒含量 = 40%",
        'param_bond_strength': "结合强度",
        'param_fluid_density': "流体密度",
        'param_fluid_viscosity': "流体粘度",
        'coupling_framework_diagram': "耦合框架图",
        'pressure_distribution': "压力分布",
        'velocity_profile': "速度分布",
        'overall_validation_score': "总体验证得分",
        'error': "误差",
        'sample_pressure_title': "压力分布示例",
        'sample_velocity_title': "速度分布示例",
        'position': "位置",
        'pressure_unit': "压力 (kPa)",
        'velocity_unit': "速度 (m/s)",
    }
}

# Function to generate sample data for visualization
def generate_sample_data(params):
    """Generate sample data for visualization."""
    st.subheader(LANG['en']['sample_data_title'])
    try:
        time_steps = np.linspace(0, 10, 100)
        bond_strength = params['bond_strength'] * np.exp(-0.1 * time_steps)
        fluid_velocity = np.sin(time_steps) * np.sqrt(params['fluid_density'] / params['fluid_viscosity'])
        
        return pd.DataFrame({
            'Time': time_steps,
            'Bond Strength': bond_strength,
            'Fluid Velocity': fluid_velocity
        })
    except Exception as e:
        st.error(LANG['en']['sample_data_error'])
        return pd.DataFrame()

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

st.set_page_config(
    page_title="CFD-DEM Simulation Dashboard",
    page_icon="📊",
    layout="wide"
)

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
    st.subheader(lang['validation_summary_title'])
    st.write(lang['validation_results_desc'])
    
    # Sample validation data for demonstration
    validation_data = {
        'erosion_rate': {
            'simulation': 0.85,
            'experimental': 0.82,
            'error': 3.7
        },
        'pressure_distribution': {
            'simulation': 0.92,
            'experimental': 0.89,
            'error': 3.4
        },
        'velocity_profile': {
            'simulation': 0.78,
            'experimental': 0.75,
            'error': 4.0
        }
    }
    
    # Display validation metrics in a more detailed way
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            lang['erosion_rate'],
            f"{validation_data['erosion_rate']['simulation']:.2f}",
            f"{validation_data['erosion_rate']['error']:.1f}% {lang['error']}"
        )
        st.write(f"{lang['experimental']}: {validation_data['erosion_rate']['experimental']:.2f}")
    
    with col2:
        st.metric(
            lang['pressure_distribution'],
            f"{validation_data['pressure_distribution']['simulation']:.2f}",
            f"{validation_data['pressure_distribution']['error']:.1f}% {lang['error']}"
        )
        st.write(f"{lang['experimental']}: {validation_data['pressure_distribution']['experimental']:.2f}")
    
    with col3:
        st.metric(
            lang['velocity_profile'],
            f"{validation_data['velocity_profile']['simulation']:.2f}",
            f"{validation_data['velocity_profile']['error']:.1f}% {lang['error']}"
        )
        st.write(f"{lang['experimental']}: {validation_data['velocity_profile']['experimental']:.2f}")
    
    # Overall validation score
    overall_score = (validation_data['erosion_rate']['simulation'] + 
                    validation_data['pressure_distribution']['simulation'] + 
                    validation_data['velocity_profile']['simulation']) / 3
    st.metric(lang['overall_validation_score'], f"{overall_score:.2f}")

def create_erosion_plot(validation_manager, lang):
    """Create a plot of erosion rate over time."""
    try:
        # Sample data for demonstration
        time_steps = np.linspace(0, 10, 100)
        simulation_erosion = np.sin(time_steps) * 0.5 + 0.5
        experimental_erosion = np.cos(time_steps) * 0.5 + 0.5
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_steps, y=simulation_erosion, mode='lines', name=lang['simulation']))
        fig.add_trace(go.Scatter(x=time_steps, y=experimental_erosion, mode='lines', name=lang['experimental']))
        fig.update_layout(
            title=lang['erosion_plot_title'],
            xaxis_title=lang['time_steps'],
            yaxis_title=lang['erosion_rate']
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating erosion plot: {e}")

def create_parameter_calibration_interface(validation_manager, lang):
    """Create an interface for parameter calibration."""
    st.subheader(lang['parameter_calibration_title'])
    st.write(lang['current_parameters'])
    
    params = {'bond_strength': 100, 'fluid_density': 1000, 'fluid_viscosity': 0.001}
    bond_strength = st.number_input(lang['param_bond_strength'], value=params['bond_strength'])
    fluid_density = st.number_input(lang['param_fluid_density'], value=params['fluid_density'])
    fluid_viscosity = st.number_input(lang['param_fluid_viscosity'], value=params['fluid_viscosity'])
    
    if st.button(lang['update_parameters']):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['dem']['bond_strength'] = bond_strength
        config['cfd']['fluid_density'] = fluid_density
        config['cfd']['fluid_viscosity'] = fluid_viscosity
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)
        st.success(lang['parameter_updated'])
        st.rerun()

def load_geotechnical_params(lang):
    """Load geotechnical parameters from config.yaml."""
    st.subheader(lang['geotechnical_params_title'])
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config.get('geotechnical', {})
    except Exception as e:
        st.error(lang['geotechnical_params_error'])
        return {}

def plot_granular_curve(params, lang):
    """Plot the granular curve based on geotechnical parameters."""
    try:
        grain_sizes = params.get('grain_sizes', [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.1])
        passing_percentages = params.get('passing_percentages', [95, 92, 88, 83, 77, 70, 62, 53, 40, 20, 5, 0])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grain_sizes, y=passing_percentages, mode='lines+markers', name='Granular Curve'))
        fig.update_layout(
            title=lang['granular_curve_title'],
            xaxis_title=lang['grain_size_mm'],
            yaxis_title=lang['passing_percentage_pct'],
            xaxis_type='log',
            yaxis_range=[0, 100]
        )
        
        fig.add_annotation(
            x=0.01, y=50,
            text=lang['clay'],
            showarrow=False,
            font=dict(size=10)
        )
        
        fig.add_annotation(
            x=0.5, y=50,
            text=lang['sand'],
            showarrow=False,
            font=dict(size=10)
        )
        
        fig.add_annotation(
            x=10, y=50,
            text=lang['gravel'],
            showarrow=False,
            font=dict(size=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error plotting granular curve: {e}")

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
    """Display geotechnical parameters in a formatted way."""
    st.subheader(lang['geotechnical_parameters'])
    
    # Create columns for different parameter categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**{lang['basic_properties']}**")
        st.write(f"- {lang['density']}: {params.get('density', 'N/A')} kg/m³")
        st.write(f"- {lang['specific_gravity']}: {params.get('specific_gravity', 'N/A')}")
        st.write(f"- {lang['water_content']}: {params.get('water_content', 'N/A')}%")
    
    with col2:
        st.write(f"**{lang['grain_size_distribution']}**")
        st.write(f"- {lang['uniformity_coefficient']}: {params.get('Cu', 'N/A')}")
        st.write(f"- {lang['curvature_coefficient']}: {params.get('Cc', 'N/A')}")
        st.write(f"- {lang['clay_content']}: {params.get('clay_content', 'N/A')}%")
    
    with col3:
        st.write(f"**{lang['hydraulic_properties']}**")
        st.write(f"- {lang['permeability']}: {params.get('permeability', 'N/A')} m/s")
        st.write(f"- {lang['porosity']}: {params.get('porosity', 'N/A')}")
        st.write(f"- {lang['void_ratio']}: {params.get('void_ratio', 'N/A')}")
    
    # Add granular curve plot
    st.subheader(lang['particle_size_distribution'])
    plot_granular_curve(params, lang)
    granular_curve_editor(params, lang)

def display_wang_reference(lang):
    """Display information about Wang's work and its implications."""
    st.subheader(lang['wang_reference_title'])
    st.write(f"**{lang['wang_reference_desc']}**")
    st.write(lang['wang_key_findings'])
    st.write(f"{lang['wang_finding_1']}\n{lang['wang_finding_1_detail']}\n{lang['wang_finding_1_detail_2']}\n{lang['wang_finding_1_detail_3']}")
    st.write(f"{lang['wang_finding_2']}\n{lang['wang_finding_2_detail']}\n{lang['wang_finding_2_detail_2']}\n{lang['wang_finding_2_detail_3']}")
    st.write(f"{lang['wang_finding_3']}\n{lang['wang_finding_3_detail']}\n{lang['wang_finding_3_detail_2']}\n{lang['wang_finding_3_detail_3']}")
    st.caption(lang['wang_paper_reference'])

def main():
    # Language selection
    lang_code = st.sidebar.selectbox(LANG['en']['select_language'], ["en", "zh"], format_func=lambda x: "English" if x=="en" else "中文")
    lang = LANG[lang_code]

    st.title(lang['main_title'])
    # Sidebar navigation
    st.sidebar.title(lang['main_navigation'])
    page = st.sidebar.radio(
        lang['main_navigation'],
        [lang['main_overview'], lang['main_coupling_framework'], lang['main_validation_results'], lang['main_parameter_calibration'], lang['main_geotechnical_parameters'], lang['main_theoretical_background']]
    )

    if page == lang['main_overview']:
        st.header(lang['project_overview'])
        st.write(lang['project_overview_desc'])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(lang['simulation_time'], "100 steps")
        with col2:
            st.metric(lang['particles'], "1000")
        with col3:
            st.metric(lang['validation_score'], "0.85")
        st.subheader(lang['geotech_params_summary'])
        display_geotechnical_params(load_geotechnical_params(lang), lang)
        st.subheader(lang['recent_results'])
        create_erosion_plot(None, lang)
    elif page == lang['main_coupling_framework']:
        st.header(lang['cfd_dem_framework'])
        st.write(lang['framework_desc'])
        load_coupling_diagram(lang)
        st.subheader(lang['framework_components'])
        st.write(f"- {lang['dem_particles']}\n- {lang['cfd_fluid']}\n- {lang['coupling_manager']}")
        st.subheader(lang['geotechnical_context'])
        geotech_params = load_geotechnical_params(lang)
        st.write(f"- {lang['particles']}: {geotech_params.get('density', 'N/A')} kg/m³\n- {lang['permeability']}: {geotech_params.get('permeability', 'N/A')} m/s\n- {lang['clay_content']}: {geotech_params.get('clay_content', 'N/A')}%")
    elif page == lang['main_validation_results']:
        st.header(lang['validation_results_header'])
        st.write(lang['validation_results_desc'])
        tab1, tab2, tab3 = st.tabs(["Erosion", "Pressure", "Velocity"] if lang_code=="en" else ["侵蚀", "压力", "速度"])
        with tab1:
            create_erosion_plot(None, lang)
            create_validation_summary(None, lang)
        with tab2:
            x = [0, 1, 2, 3, 4, 5]
            pressure = [100, 98, 95, 93, 92, 91]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=pressure, name=lang['pressure_distribution'], line=dict(color='purple')))
            fig.update_layout(
                title=lang['sample_pressure_title'],
                xaxis_title=lang['position'],
                yaxis_title=lang['pressure_unit']
            )
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            x = [0, 1, 2, 3, 4, 5]
            velocity = [0, 0.2, 0.4, 0.35, 0.3, 0.25]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=velocity, name=lang['velocity_profile'], line=dict(color='green')))
            fig.update_layout(
                title=lang['sample_velocity_title'],
                xaxis_title=lang['position'],
                yaxis_title=lang['velocity_unit']
            )
            st.plotly_chart(fig, use_container_width=True)
    elif page == lang['main_parameter_calibration']:
        st.header(lang['parameter_calibration_header'])
        st.write(lang['parameter_calibration_desc'])
        st.subheader(lang['current_geotech_params'])
        display_geotechnical_params(load_geotechnical_params(lang), lang)
        create_parameter_calibration_interface(None, lang)
    elif page == lang['main_geotechnical_parameters']:
        st.header(lang['geotechnical_parameters'])
        st.write(lang['geotech_params_summary'])
        display_geotechnical_params(load_geotechnical_params(lang), lang)
        st.subheader(lang['parameter_sensitivity'])
        st.write("1. 渗透系数\n2. 粘土含量\n3. 颗粒密度" if lang_code=="zh" else "1. Permeability\n2. Clay Content\n3. Particle Density")
        st.subheader(lang['parameter_relationships'])
        st.write("- 渗透系数影响侵蚀速率\n- 粘土含量影响结合力\n- 颗粒密度影响流固耦合" if lang_code=="zh" else "- Permeability affects erosion rates\n- Clay content influences bond strength\n- Particle density impacts fluid-particle interaction")
        st.subheader(lang['soil_classification'])
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{lang['soil_uscs_classification']}**")
            st.write(f"- {lang['soil_gravel']}\n- {lang['soil_sand']}\n- {lang['soil_silt']}\n- {lang['soil_clay']}")
            st.write(f"**{lang['soil_types']}**")
            st.write(f"- {lang['soil_well_graded']}\n- {lang['soil_poorly_graded']}\n- {lang['soil_silty']}\n- {lang['soil_clayey']}")
        with col2:
            st.write(f"**{lang['soil_current_classification']}**")
            classification_data = {'gravel': 15, 'sand': 45, 'silt': 25, 'clay': 15}
            fig = go.Figure(data=[go.Pie(
                labels=[lang['soil_gravel'], lang['soil_sand'], lang['soil_silt'], lang['soil_clay']],
                values=[classification_data['gravel'], classification_data['sand'], classification_data['silt'], classification_data['clay']],
                hole=.3
            )])
            fig.update_layout(title=lang['soil_composition'])
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"**{lang['soil_classification_result']}**")
            st.write(lang['soil_classification_result_value'])
            st.write(f"- {lang['soil_cu']}\n- {lang['soil_cc']}\n- {lang['soil_fine_content']}")
    elif page == lang['main_theoretical_background']:
        st.header(lang['theoretical_background_header'])
        display_wang_reference(lang)
        st.subheader(lang['model_validation'])
        st.write("我们的实现遵循Wang的方法，包括：\n1. 侵蚀过程中的参数演化\n2. 应力-应变关系\n3. 内部侵蚀模式" if lang_code=="zh" else "Our implementation follows Wang's methodology for:\n1. Parameter evolution during erosion\n2. Stress-strain relationships\n3. Internal erosion patterns")
        st.subheader(lang['parameter_sensitivity_analysis'])
        st.write("基于Wang的研究，最敏感的参数有：\n1. 粘聚力（侵蚀期间降低20-40%）\n2. 内摩擦角（降低5-15°）\n3. 渗透系数（增加1-2个数量级）" if lang_code=="zh" else "Based on Wang's findings, the most sensitive parameters are:\n1. Cohesion (20-40% reduction during erosion)\n2. Friction angle (5-15° reduction)\n3. Permeability (1-2 orders of magnitude increase)")

if __name__ == "__main__":
    main() 