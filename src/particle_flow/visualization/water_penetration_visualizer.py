"""
Visualization module for water penetration test results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Optional
import logging


class WaterPenetrationVisualizer:
    def __init__(self, output_dir: str):
        """Initialize the visualizer.

        Args:
            output_dir (str): Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Use a default matplotlib style instead of seaborn
        plt.style.use('default')

        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

    def plot_test_results(self, results: Dict, test_params: Dict):
        """Create comprehensive plots of test results.

        Args:
            results (Dict): Test results from WaterPenetrationTest
            test_params (Dict): Test parameters
        """
        self.logger.info("Creating test result plots")

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)

        # Plot pressure and flow rate
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_pressure_flow(ax1, results)

        # Plot erosion rate and particle outflow
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_erosion(ax2, results)

        # Plot porosity evolution
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_porosity(ax3, results)

        # Plot displacement
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_displacement(ax4, results)

        # Plot failure criteria
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_failure_criteria(ax5, results, test_params)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / 'test_results.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pressure_flow(self, ax: plt.Axes, results: Dict):
        """Plot pressure and flow rate over time."""
        time = results['time']

        # Plot pressure
        ax.plot(time, results['pressure'], 'b-', label='Pressure')
        ax.set_ylabel('Pressure (Pa)')
        ax.set_xlabel('Time (s)')

        # Create second y-axis for flow rate
        ax2 = ax.twinx()
        ax2.plot(time, results['flow_rate'], 'r--', label='Flow Rate')
        ax2.set_ylabel('Flow Rate (m³/s)')

        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax.set_title('Pressure and Flow Rate Evolution')

    def _plot_erosion(self, ax: plt.Axes, results: Dict):
        """Plot erosion rate and particle outflow."""
        time = results['time']

        # Plot erosion rate
        ax.plot(time, results['erosion_rate'], 'g-', label='Erosion Rate')
        ax.set_ylabel('Erosion Rate (kg/s)')
        ax.set_xlabel('Time (s)')

        # Create second y-axis for particle outflow
        ax2 = ax.twinx()
        ax2.plot(time, results['particle_outflow'],
                 'm--', label='Particle Outflow')
        ax2.set_ylabel('Particle Outflow (kg)')

        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax.set_title('Erosion and Particle Outflow')

    def _plot_porosity(self, ax: plt.Axes, results: Dict):
        """Plot porosity evolution."""
        time = results['time']

        ax.plot(time, results['porosity'], 'c-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Porosity')
        ax.set_title('Porosity Evolution')

        # Add horizontal line for initial porosity
        ax.axhline(y=results['porosity'][0], color='r', linestyle='--',
                   label=f'Initial Porosity: {results["porosity"][0]:.3f}')
        ax.legend()

    def _plot_displacement(self, ax: plt.Axes, results: Dict):
        """Plot sample displacement."""
        time = results['time']

        ax.plot(time, results['displacement'], 'b-')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Displacement (m)')
        ax.set_title('Sample Displacement')

    def _plot_failure_criteria(self, ax: plt.Axes, results: Dict, test_params: Dict):
        """Plot failure criteria analysis."""
        time = results['time']

        # Calculate normalized values with default values if parameters are missing
        # 10% of sample height
        max_displacement = test_params['sample']['height'] * 0.1
        max_erosion = test_params['material'].get(
            'critical_erosion_rate', 0.1)  # Default 0.1 kg/s
        max_porosity = test_params['material'].get(
            'critical_porosity', 0.4)  # Default 0.4

        displacement_norm = np.array(
            results['displacement']) / max_displacement
        erosion_norm = np.array(results['erosion_rate']) / max_erosion
        porosity_norm = np.array(results['porosity']) / max_porosity

        # Plot normalized values
        ax.plot(time, displacement_norm, 'b-', label='Displacement')
        ax.plot(time, erosion_norm, 'r-', label='Erosion Rate')
        ax.plot(time, porosity_norm, 'g-', label='Porosity')

        # Add failure threshold line
        ax.axhline(y=1.0, color='k', linestyle='--', label='Failure Threshold')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Failure Criteria Analysis')
        ax.legend()

    def _plot_saturation_analysis(self, ax: plt.Axes, results: Dict):
        """Plot saturation analysis results."""
        time = results['time']
        saturation_degree = results['saturation_degree']
        water_content = results['water_content']
        stability_index = results['stability_index']

        # Plot saturation degree
        ax.plot(time, saturation_degree, 'b-', label='Saturation Degree')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Saturation Degree')
        ax.set_title('Saturation Analysis')
        ax.grid(True)

        # Add water content on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(time, water_content, 'r--', label='Water Content')
        ax2.set_ylabel('Water Content')

        # Add stability index on tertiary y-axis
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(time, stability_index, 'g:', label='Stability Index')
        ax3.set_ylabel('Stability Index')

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines1 + lines2 + lines3, labels1 +
                  labels2 + labels3, loc='upper right')

    def create_dashboard(self, results: Dict, test_params: Dict):
        """Create an interactive dashboard for test results.

        Args:
            results (Dict): Test results from WaterPenetrationTest
            test_params (Dict): Test parameters
        """
        try:
            import streamlit as st
        except ImportError:
            self.logger.error(
                "Streamlit not installed. Please install it to use the dashboard.")
            return

        st.title("Water Penetration Test Results")

        # Sidebar with test parameters
        st.sidebar.header("Test Parameters")
        st.sidebar.json(test_params)

        # Main content
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Pressure and Flow Rate")
            fig, ax = plt.subplots(figsize=(8, 4))
            self._plot_pressure_flow(ax, results)
            st.pyplot(fig)
            plt.close()

            st.subheader("Porosity Evolution")
            fig, ax = plt.subplots(figsize=(8, 4))
            self._plot_porosity(ax, results)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Erosion and Particle Outflow")
            fig, ax = plt.subplots(figsize=(8, 4))
            self._plot_erosion(ax, results)
            st.pyplot(fig)
            plt.close()

            st.subheader("Sample Displacement")
            fig, ax = plt.subplots(figsize=(8, 4))
            self._plot_displacement(ax, results)
            st.pyplot(fig)
            plt.close()

        # Failure criteria analysis
        st.subheader("Failure Criteria Analysis")
        fig, ax = plt.subplots(figsize=(12, 4))
        self._plot_failure_criteria(ax, results, test_params)
        st.pyplot(fig)
        plt.close()

        # Download data
        st.sidebar.header("Download Data")
        if st.sidebar.button("Download Results"):
            import pandas as pd
            df = pd.DataFrame(results)
            st.sidebar.download_button(
                label="Download CSV",
                data=df.to_csv(index=False),
                file_name="water_penetration_results.csv",
                mime="text/csv"
            )
