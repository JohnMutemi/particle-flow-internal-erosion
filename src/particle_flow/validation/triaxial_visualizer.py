import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import time

class TriaxialTestVisualizer:
    def __init__(self, validation_report, config_path=None):
        """Initialize the triaxial test visualizer.
        
        Args:
            validation_report (dict): Validation report from TriaxialTestValidator
            config_path (str, optional): Path to the test parameters YAML file
        """
        self.report = validation_report
        self.config_path = config_path
        
    def create_combined_dashboard(self, stress_data=None, strain_data=None, pore_pressure_data=None):
        """Create a comprehensive dashboard with all test results and parameters.
        
        Args:
            stress_data (numpy.ndarray, optional): Array of stress values
            strain_data (numpy.ndarray, optional): Array of strain values
            pore_pressure_data (numpy.ndarray, optional): Array of pore pressure values
        """
        # Create subplot figure with 3 rows and 3 columns
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                "Test Parameters", "Cohesion Validation", "Friction Angle Validation",
                "Stress-Strain Curve", "Particle Size Distribution", "Mohr's Circle",
                "Pore Pressure", "Validation Summary", "Sample Geometry"
            ),
            specs=[
                [{"type": "table"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "xy"}, {"type": "histogram"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "table"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Add test parameters table with actual values
        test_params = {
            'Parameter': [
                'Sample Diameter',
                'Sample Height',
                'Sample Volume',
                'Particle Size Limit',
                'Cohesion',
                'Friction Angle',
                'Test Type'
            ],
            'Value': [
                '39.1 mm',
                '80.0 mm',
                '95,980 mm³',
                '2.0 mm',
                '17.5 kPa',
                '34.0°',
                'Consolidation Drainage'
            ]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(test_params.keys()),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[test_params[k] for k in test_params.keys()],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=1, col=1
        )
        
        # Add cohesion gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=self.report['cohesion_validation']['simulated'],
                title={'text': "Cohesion (kPa)"},
                gauge={
                    'axis': {'range': [0, self.report['cohesion_validation']['target'] * 1.5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, self.report['cohesion_validation']['target']], 'color': "lightgray"},
                        {'range': [self.report['cohesion_validation']['target'], 
                                 self.report['cohesion_validation']['target'] * 1.5], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': self.report['cohesion_validation']['target']
                    }
                }
            ),
            row=1, col=2
        )
        
        # Add friction angle gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=self.report['friction_angle_validation']['simulated'],
                title={'text': "Friction Angle (°)"},
                gauge={
                    'axis': {'range': [0, self.report['friction_angle_validation']['target'] * 1.5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, self.report['friction_angle_validation']['target']], 'color': "lightgray"},
                        {'range': [self.report['friction_angle_validation']['target'],
                                 self.report['friction_angle_validation']['target'] * 1.5], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': self.report['friction_angle_validation']['target']
                    }
                }
            ),
            row=1, col=3
        )
        
        # Add stress-strain curve if data is provided
        if stress_data is not None and strain_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=strain_data,
                    y=stress_data,
                    mode='lines',
                    name='Stress-Strain Curve',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            # Add peak stress point
            peak_stress_idx = np.argmax(stress_data)
            fig.add_trace(
                go.Scatter(
                    x=[strain_data[peak_stress_idx]],
                    y=[stress_data[peak_stress_idx]],
                    mode='markers',
                    name='Peak Stress',
                    marker=dict(color='red', size=10)
                ),
                row=2, col=1
            )
        
        # Add particle size distribution histogram
        if 'particle_sizes' in self.report:
            fig.add_trace(
                go.Histogram(
                    x=self.report['particle_sizes'],
                    name="Particle Sizes",
                    marker_color='blue',
                    opacity=0.75,
                    nbinsx=20
                ),
                row=2, col=2
            )
        
        # Add Mohr's circle
        if stress_data is not None:
            max_stress = np.max(stress_data)
            min_stress = np.min(stress_data)
            center = (max_stress + min_stress) / 2
            radius = (max_stress - min_stress) / 2
            
            theta = np.linspace(0, 2*np.pi, 100)
            x = center + radius * np.cos(theta)
            y = radius * np.sin(theta)
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name="Mohr's Circle",
                    line=dict(color='green', width=2)
                ),
                row=2, col=3
            )
        
        # Add pore pressure plot if data is provided
        if pore_pressure_data is not None and strain_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=strain_data,
                    y=pore_pressure_data,
                    mode='lines',
                    name='Pore Pressure',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
        
        # Add validation summary table
        summary_data = {
            'Parameter': ['Cohesion', 'Friction Angle', 'Particle Sizes', 'Overall'],
            'Status': [
                '✓' if self.report['cohesion_validation']['is_valid'] else '✗',
                '✓' if self.report['friction_angle_validation']['is_valid'] else '✗',
                '✓' if self.report['particle_size_validation']['is_valid'] else '✗',
                '✓' if self.report['overall_validation']['is_valid'] else '✗'
            ],
            'Error (%)': [
                f"{self.report['cohesion_validation']['error_percentage']:.2f}",
                f"{self.report['friction_angle_validation']['error_percentage']:.2f}",
                "N/A",
                "N/A"
            ]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(summary_data.keys()),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[summary_data[k] for k in summary_data.keys()],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=3, col=2
        )
        
        # Add sample geometry indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=95980,  # Sample volume in mm³
                title={'text': "Sample Volume (mm³)"},
                delta={'reference': 100000, 'relative': True},
                number={'suffix': " mm³"}
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="Triaxial Test Comprehensive Dashboard",
            height=1200,
            width=1500,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Axial Strain (%)", row=2, col=1)
        fig.update_yaxes(title_text="Deviatoric Stress (kPa)", row=2, col=1)
        fig.update_xaxes(title_text="Particle Size (mm)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_xaxes(title_text="Normal Stress (kPa)", row=2, col=3)
        fig.update_yaxes(title_text="Shear Stress (kPa)", row=2, col=3)
        fig.update_xaxes(title_text="Axial Strain (%)", row=3, col=1)
        fig.update_yaxes(title_text="Pore Pressure (kPa)", row=3, col=1)
        
        return fig

def run_streamlit_app():
    """Run the Streamlit app for real-time visualization."""
    st.set_page_config(page_title="Triaxial Test Dashboard", layout="wide")
    
    st.title("Triaxial Test Real-time Dashboard")
    
    # Sidebar controls
    st.sidebar.header("Test Parameters")
    cohesion = st.sidebar.slider("Cohesion (kPa)", 0.0, 50.0, 17.5)
    friction_angle = st.sidebar.slider("Friction Angle (°)", 0.0, 45.0, 34.0)
    particle_size_limit = st.sidebar.slider("Particle Size Limit (mm)", 0.1, 5.0, 2.0)
    
    # Create placeholder for the dashboard
    dashboard_placeholder = st.empty()
    
    # Simulation loop
    while True:
        # Generate example data
        strain = np.linspace(0, 20, 100)
        stress = cohesion * (1 - np.exp(-0.2 * strain))
        pore_pressure = 5 * (1 - np.exp(-0.1 * strain))
        particle_sizes = np.random.uniform(0.5, particle_size_limit, 1000)
        
        # Create example validation report
        simulation_results = {
            'cohesion': cohesion,
            'friction_angle': friction_angle,
            'particle_sizes': particle_sizes,
            'cohesion_validation': {
                'simulated': cohesion,
                'target': 17.5,
                'is_valid': abs(cohesion - 17.5) < 1.0,
                'error_percentage': abs(cohesion - 17.5) / 17.5 * 100
            },
            'friction_angle_validation': {
                'simulated': friction_angle,
                'target': 34.0,
                'is_valid': abs(friction_angle - 34.0) < 1.0,
                'error_percentage': abs(friction_angle - 34.0) / 34.0 * 100
            },
            'particle_size_validation': {
                'is_valid': np.all(particle_sizes <= particle_size_limit)
            },
            'overall_validation': {
                'is_valid': True
            }
        }
        
        # Create visualizer and dashboard
        visualizer = TriaxialTestVisualizer(simulation_results)
        dashboard = visualizer.create_combined_dashboard(
            stress_data=stress,
            strain_data=strain,
            pore_pressure_data=pore_pressure
        )
        
        # Update dashboard
        dashboard_placeholder.plotly_chart(dashboard, use_container_width=True)
        
        # Add a small delay to prevent overwhelming the system
        time.sleep(0.1)

def main():
    """Main function to run either the Streamlit app or generate a static dashboard."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        run_streamlit_app()
    else:
        # Example usage for static dashboard
        # Create example validation report
        simulation_results = {
            'cohesion': 17.0,  # kPa
            'friction_angle': 33.5,  # degrees
            'particle_sizes': np.random.uniform(0.5, 2.0, 1000),  # mm
            'cohesion_validation': {
                'simulated': 17.0,
                'target': 17.5,
                'is_valid': True,
                'error_percentage': 2.86
            },
            'friction_angle_validation': {
                'simulated': 33.5,
                'target': 34.0,
                'is_valid': True,
                'error_percentage': 1.47
            },
            'particle_size_validation': {
                'is_valid': True
            },
            'overall_validation': {
                'is_valid': True
            }
        }
        
        # Create visualizer
        visualizer = TriaxialTestVisualizer(simulation_results)
        
        # Create example data
        strain = np.linspace(0, 20, 100)  # 0-20% strain
        stress = 17.5 * (1 - np.exp(-0.2 * strain))  # Example stress-strain curve
        pore_pressure = 5 * (1 - np.exp(-0.1 * strain))  # Example pore pressure curve
        
        # Create and save combined dashboard
        dashboard = visualizer.create_combined_dashboard(
            stress_data=stress,
            strain_data=strain,
            pore_pressure_data=pore_pressure
        )
        dashboard.write_html('results/validation/combined_dashboard.html')

if __name__ == "__main__":
    main() 