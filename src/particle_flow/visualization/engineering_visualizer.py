"""
Engineering-scale visualization module for monitoring tunnel excavation and water inrush.
This module provides specialized visualizations for:
1. Triaxial seepage test
2. Engineering-scale monitoring
3. Treatment measures effectiveness
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

logger = logging.getLogger(__name__)


class EngineeringVisualizer:
    def __init__(self, config: Dict):
        """Initialize the engineering visualizer."""
        self.config = config
        self.output_dir = Path(config['output']['directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize figure for real-time monitoring
        self.fig = None
        self.animation = None

        # Data storage
        self.monitoring_data = {
            'time': [],
            'displacement': {},
            'seepage_velocity': {},
            'particle_outflow': {},
            'porosity': {},
            'pressure': {},
            'treatment_effectiveness': {}
        }

        logger.info("Engineering visualizer initialized")

    def plot_triaxial_test(self, test_data: Dict):
        """Plot triaxial seepage test results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Pressure Evolution', 'Water Flow Pattern',
                            'Test Results Comparison', 'Erosion Rate')
        )

        # Pressure evolution
        fig.add_trace(
            go.Scatter(x=test_data['time'], y=test_data['pressure'],
                       name='Pressure'),
            row=1, col=1
        )

        # Water flow pattern
        fig.add_trace(
            go.Scatter(x=test_data['x'], y=test_data['y'],
                       mode='markers',
                       marker=dict(
                size=10,
                color=test_data['velocity_magnitude'],
                colorscale='Viridis',
                showscale=True
            ),
                name='Flow Pattern'),
            row=1, col=2
        )

        # Test results comparison
        fig.add_trace(
            go.Scatter(x=test_data['time'], y=test_data['experimental'],
                       name='Experimental'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=test_data['time'], y=test_data['simulated'],
                       name='Simulated'),
            row=2, col=1
        )

        # Erosion rate
        fig.add_trace(
            go.Scatter(x=test_data['time'], y=test_data['erosion_rate'],
                       name='Erosion Rate'),
            row=2, col=2
        )

        fig.update_layout(height=800, width=1200,
                          title_text="Triaxial Seepage Test Results")
        return fig

    def plot_engineering_monitoring(self, monitoring_data: Dict):
        """Plot engineering-scale monitoring data."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Displacement Curves', 'Seepage Velocity Evolution',
                            'Particle Outflow', 'Porosity Evolution')
        )

        # Displacement curves
        for point_id, displacement in monitoring_data['displacement'].items():
            fig.add_trace(
                go.Scatter(x=monitoring_data['time'], y=displacement,
                           name=f'Point {point_id}'),
                row=1, col=1
            )

        # Seepage velocity
        fig.add_trace(
            go.Scatter(x=monitoring_data['time'],
                       y=monitoring_data['seepage_velocity'],
                       name='Seepage Velocity'),
            row=1, col=2
        )

        # Particle outflow
        fig.add_trace(
            go.Scatter(x=monitoring_data['time'],
                       y=monitoring_data['particle_outflow'],
                       name='Particle Outflow'),
            row=2, col=1
        )

        # Porosity evolution
        fig.add_trace(
            go.Scatter(x=monitoring_data['time'],
                       y=monitoring_data['porosity'],
                       name='Porosity'),
            row=2, col=2
        )

        fig.update_layout(height=800, width=1200,
                          title_text="Engineering-Scale Monitoring")
        return fig

    def plot_treatment_measures(self, treatment_data: Dict):
        """Plot treatment measures effectiveness."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Advance Grouting', 'Face Drilling and Drainage',
                            'Face Spraying', 'Treatment Effectiveness')
        )

        # Advance grouting
        fig.add_trace(
            go.Scatter(x=treatment_data['time'],
                       y=treatment_data['grouting_pressure'],
                       name='Grouting Pressure'),
            row=1, col=1
        )

        # Face drilling and drainage
        fig.add_trace(
            go.Scatter(x=treatment_data['time'],
                       y=treatment_data['drainage_rate'],
                       name='Drainage Rate'),
            row=1, col=2
        )

        # Face spraying
        fig.add_trace(
            go.Scatter(x=treatment_data['time'],
                       y=treatment_data['spray_coverage'],
                       name='Spray Coverage'),
            row=2, col=1
        )

        # Treatment effectiveness
        fig.add_trace(
            go.Scatter(x=treatment_data['time'],
                       y=treatment_data['erosion_reduction'],
                       name='Erosion Reduction'),
            row=2, col=2
        )

        fig.update_layout(height=800, width=1200,
                          title_text="Treatment Measures Effectiveness")
        return fig

    def update_monitoring_data(self, new_data: Dict):
        """Update monitoring data with new measurements."""
        for key, value in new_data.items():
            if key in self.monitoring_data:
                if isinstance(value, dict):
                    self.monitoring_data[key].update(value)
                else:
                    self.monitoring_data[key].append(value)

    def save_plots(self, prefix: str = "monitoring"):
        """Save all monitoring plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save triaxial test results
        if 'test_data' in self.monitoring_data:
            fig = self.plot_triaxial_test(self.monitoring_data['test_data'])
            fig.write_html(self.output_dir /
                           f"{prefix}_triaxial_test_{timestamp}.html")

        # Save engineering monitoring
        fig = self.plot_engineering_monitoring(self.monitoring_data)
        fig.write_html(self.output_dir /
                       f"{prefix}_engineering_monitoring_{timestamp}.html")

        # Save treatment measures
        if 'treatment_data' in self.monitoring_data:
            fig = self.plot_treatment_measures(
                self.monitoring_data['treatment_data'])
            fig.write_html(self.output_dir /
                           f"{prefix}_treatment_measures_{timestamp}.html")

    def create_real_time_monitoring(self):
        """Create real-time monitoring dashboard."""
        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Displacement', 'Seepage Velocity',
                            'Particle Outflow', 'Porosity')
        )

        # Initialize traces
        for i in range(4):  # Number of monitoring points
            self.fig.add_trace(
                go.Scatter(x=[], y=[], name=f'Point {i+1}'),
                row=1, col=1
            )

        self.fig.add_trace(
            go.Scatter(x=[], y=[], name='Seepage Velocity'),
            row=1, col=2
        )

        self.fig.add_trace(
            go.Scatter(x=[], y=[], name='Particle Outflow'),
            row=2, col=1
        )

        self.fig.add_trace(
            go.Scatter(x=[], y=[], name='Porosity'),
            row=2, col=2
        )

        self.fig.update_layout(height=800, width=1200,
                               title_text="Real-time Monitoring")
        return self.fig

    def update_real_time_plot(self, new_data: Dict):
        """Update real-time monitoring plot with new data."""
        if self.fig is None:
            self.create_real_time_monitoring()

        # Update displacement curves
        for i, displacement in enumerate(new_data['displacement'].values()):
            self.fig.data[i].x = list(self.fig.data[i].x) + [new_data['time']]
            self.fig.data[i].y = list(self.fig.data[i].y) + [displacement]

        # Update seepage velocity
        self.fig.data[4].x = list(self.fig.data[4].x) + [new_data['time']]
        self.fig.data[4].y = list(self.fig.data[4].y) + \
            [new_data['seepage_velocity']]

        # Update particle outflow
        self.fig.data[5].x = list(self.fig.data[5].x) + [new_data['time']]
        self.fig.data[5].y = list(self.fig.data[5].y) + \
            [new_data['particle_outflow']]

        # Update porosity
        self.fig.data[6].x = list(self.fig.data[6].x) + [new_data['time']]
        self.fig.data[6].y = list(self.fig.data[6].y) + [new_data['porosity']]
