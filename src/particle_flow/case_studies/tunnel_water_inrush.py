"""
Tunnel water inrush case study module.

This module implements the tunnel water inrush scenario, including:
1. Initial condition setup
2. Boundary condition application
3. Erosion and failure analysis
4. Risk assessment
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from scipy import stats
import matplotlib.pyplot as plt
from particle_flow.cfd.coupling import CFDDEMCoupling
from particle_flow.visualization.real_time_visualizer import RealTimeVisualizer
from particle_flow.cfd.force_models import ForceModels
from particle_flow.case_studies.initialization_stage import InitializationStage
from particle_flow.case_studies.boundary_condition_stage import BoundaryConditionStage
from particle_flow.case_studies.simulation_stage import SimulationStage
from particle_flow.case_studies.erosion_analysis_stage import ErosionAnalysisStage
from particle_flow.case_studies.risk_assessment_stage import RiskAssessmentStage
from particle_flow.case_studies.validation_stage import ValidationStage

logger = logging.getLogger(__name__)


class TunnelWaterInrush:
    def __init__(self, config: Dict):
        """Initialize the tunnel water inrush case study."""
        self.config = config

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load case study parameters
        self.case_params = self._load_case_params()

        # Initialize output directory
        self.output_dir = Path(config['output']['directory']) / 'tunnel_inrush'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize coupling framework
        self.coupling = CFDDEMCoupling(config)

        # Initialize force models
        self.force_models = ForceModels(config)

        # Initialize real-time visualizer
        self.visualizer = RealTimeVisualizer(config)

        # Case study specific parameters
        self.tunnel_length = 80.0  # meters
        self.tunnel_diameter = 8.0  # meters
        self.water_pressure = config['case_study'].get(
            'water_pressure', 1e6)  # Pa
        self.mud_concentration = config['case_study'].get(
            'mud_concentration', 0.3)  # 30% by volume

        # Initialize tunnel geometry
        self._initialize_tunnel_geometry()

        # State tracking
        self.time_steps = []
        self.erosion_stats = []
        self.bond_health_stats = []
        self.particle_data = []
        self.fluid_data = []
        self.erosion_data = []

        # Use new stage modules
        self.initialization_stage = InitializationStage(
            self.config, self.case_params)
        self.boundary_condition_stage = BoundaryConditionStage(
            self.config, self.coupling, self.water_pressure, self.tunnel_length, self.tunnel_diameter)
        self.simulation_stage = SimulationStage(self.coupling, self.visualizer)
        self.erosion_analysis_stage = ErosionAnalysisStage()
        self.risk_assessment_stage = RiskAssessmentStage()
        self.validation_stage = ValidationStage(
            config_path='data/input/triaxial_test/test_parameters.yaml')

        self.logger.info("Tunnel water inrush case study initialized")

    def _load_case_params(self) -> Dict:
        """Load case study parameters from config."""
        return {
            'tunnel': {
                'center': np.array(self.config['case_study']['tunnel']['center']),
                'diameter': self.config['case_study']['tunnel']['diameter'],
                'length': self.config['case_study']['tunnel']['length']
            },
            'water_pressure': {
                'inlet': self.config['case_study']['water_pressure']['inlet'],
                'outlet': self.config['case_study']['water_pressure']['outlet']
            },
            'mud_concentration': self.config['case_study']['mud_concentration']
        }

    def _initialize_tunnel_geometry(self):
        """Initialize tunnel geometry and boundary conditions."""
        # Convert tunnel dimensions to simulation units
        length_scale = self.config['simulation']['domain_size'][0] / \
            self.tunnel_length
        diameter_scale = self.config['simulation']['domain_size'][1] / \
            self.tunnel_diameter

        # Set up tunnel walls with proper boundary checking
        self.tunnel_walls = {
            'x_min': max(0.0, 0.0),
            'x_max': min(self.tunnel_length * length_scale, self.config['simulation']['domain_size'][0]),
            'y_min': max(-self.tunnel_diameter/2 * diameter_scale, -self.config['simulation']['domain_size'][1]/2),
            'y_max': min(self.tunnel_diameter/2 * diameter_scale, self.config['simulation']['domain_size'][1]/2),
            'z_min': max(-self.tunnel_diameter/2 * diameter_scale, -self.config['simulation']['domain_size'][2]/2),
            'z_max': min(self.tunnel_diameter/2 * diameter_scale, self.config['simulation']['domain_size'][2]/2)
        }

        # Set up tunnel geometry for CFD solver
        tunnel_params = {
            'length': min(self.tunnel_length * length_scale, self.config['simulation']['domain_size'][0]),
            'diameter': min(self.tunnel_diameter * diameter_scale, min(self.config['simulation']['domain_size'][1:])),
            'center': np.array([
                min(self.tunnel_length * length_scale / 2,
                    self.config['simulation']['domain_size'][0] / 2),
                0.0,
                0.0
            ]),
            'roughness': 0.001  # 1mm roughness
        }

        # Set boundary conditions
        boundary_conditions = {
            'inlet': {
                'type': 'pressure',
                'value': self.water_pressure,
                'position': 0.0
            },
            'outlet': {
                'type': 'pressure',
                'value': 0.0,
                'position': min(self.tunnel_length * length_scale, self.config['simulation']['domain_size'][0])
            },
            'walls': {
                'type': 'no-slip',
                'roughness': 0.001
            }
        }

        # Update CFD solver with tunnel geometry and boundary conditions
        self.coupling.cfd_solver.set_tunnel_geometry(tunnel_params)
        self.coupling.cfd_solver.set_boundary_conditions(boundary_conditions)

        self.logger.info("Tunnel geometry and boundary conditions initialized")

    def setup_initial_conditions(self, domain_size: Tuple[float, float, float],
                                 particle_radius: float) -> Dict:
        """Set up initial conditions for the tunnel water inrush scenario."""
        # Create particle positions
        positions = self._create_particle_positions(
            domain_size, particle_radius)

        # Create initial velocities
        velocities = np.zeros_like(positions)

        # Create initial bond health
        bond_health = np.ones(len(positions))

        # Apply tunnel geometry
        self._apply_tunnel_geometry(positions, bond_health)

        # Apply initial water pressure
        pressure_field = self._create_pressure_field(domain_size)

        # Create initial velocity field
        velocity_field = self._create_velocity_field(domain_size)

        return {
            'positions': positions,
            'velocities': velocities,
            'bond_health': bond_health,
            'pressure_field': pressure_field,
            'velocity_field': velocity_field
        }

    def _create_particle_positions(self, domain_size: Tuple[float, float, float],
                                   particle_radius: float) -> np.ndarray:
        """Create initial particle positions."""
        # Compute number of particles
        volume = domain_size[0] * domain_size[1] * domain_size[2]
        particle_volume = 4/3 * np.pi * particle_radius**3
        num_particles = int(volume / particle_volume *
                            0.6)  # 60% packing density

        # Create random positions
        positions = np.random.rand(num_particles, 3)
        positions[:, 0] *= domain_size[0]
        positions[:, 1] *= domain_size[1]
        positions[:, 2] *= domain_size[2]

        return positions

    def _apply_tunnel_geometry(self, positions: np.ndarray,
                               bond_health: np.ndarray):
        """Apply tunnel geometry to particle positions and bond health."""
        tunnel_center = self.case_params['tunnel']['center']
        tunnel_radius = self.case_params['tunnel']['diameter'] / 2

        # Compute distances from tunnel center
        distances = np.linalg.norm(positions - tunnel_center, axis=1)

        # Remove particles inside tunnel
        mask = distances > tunnel_radius
        positions[:] = positions[mask]
        bond_health[:] = bond_health[mask]

        # Reduce bond health near tunnel surface
        surface_distance = 2 * tunnel_radius
        surface_mask = distances < surface_distance
        bond_health[surface_mask] *= 0.8

    def _create_pressure_field(self, domain_size: Tuple[float, float, float]) -> np.ndarray:
        """Create initial pressure field."""
        # Create grid
        nx, ny, nz = 100, 100, 100
        x = np.linspace(0, domain_size[0], nx)
        y = np.linspace(0, domain_size[1], ny)
        z = np.linspace(0, domain_size[2], nz)

        # Create pressure field
        pressure = np.zeros((nx, ny, nz))

        # Apply pressure gradient
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Compute distance from tunnel
                    pos = np.array([x[i], y[j], z[k]])
                    distance = np.linalg.norm(
                        pos - self.case_params['tunnel']['center'])

                    if distance > self.case_params['tunnel']['diameter'] / 2:
                        # Linear pressure gradient
                        pressure[i, j, k] = self.case_params['water_pressure']['inlet'] - \
                            (self.case_params['water_pressure']['inlet'] -
                             self.case_params['water_pressure']['outlet']) * \
                            distance / domain_size[0]

        return pressure

    def _create_velocity_field(self, domain_size: Tuple[float, float, float]) -> np.ndarray:
        """Create initial velocity field."""
        # Create grid
        nx, ny, nz = 100, 100, 100
        x = np.linspace(0, domain_size[0], nx)
        y = np.linspace(0, domain_size[1], ny)
        z = np.linspace(0, domain_size[2], nz)

        # Create velocity field
        velocity = np.zeros((nx, ny, nz, 3))

        # Apply initial velocity based on pressure gradient
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Compute distance from tunnel
                    pos = np.array([x[i], y[j], z[k]])
                    distance = np.linalg.norm(
                        pos - self.case_params['tunnel']['center'])

                    if distance > self.case_params['tunnel']['diameter'] / 2:
                        # Compute velocity based on pressure gradient
                        pressure_gradient = (self.case_params['water_pressure']['inlet'] -
                                             self.case_params['water_pressure']['outlet']) / \
                            domain_size[0]
                        velocity[i, j, k, 0] = -pressure_gradient / \
                            self.config['cfd']['fluid_viscosity']

        return velocity

    def run(self, num_steps: int, domain_size, particle_radius):
        # 1. Initialization
        initial_conditions = self.initialization_stage.setup_initial_conditions(
            domain_size, particle_radius)
        # 2. Boundary Conditions
        self.boundary_condition_stage.initialize_tunnel_geometry()
        # 3. Simulation
        simulation_results = self.simulation_stage.run_simulation(
            num_steps, initial_conditions)
        # 4. Erosion Analysis
        erosion_results = self.erosion_analysis_stage.analyze_erosion_progress(
            simulation_results)
        # 5. Risk Assessment
        risk_results = self.risk_assessment_stage.compute_failure_risk(
            eroded_particles=simulation_results.get('eroded_particles', []),
            bond_health=simulation_results.get('bond_health', []),
            positions=simulation_results.get('positions', [])
        )
        # 6. Validation (including triaxial)
        validation_report = self.validation_stage.run_validation(
            simulation_results)
        return {
            'initial_conditions': initial_conditions,
            'simulation_results': simulation_results,
            'erosion_results': erosion_results,
            'risk_results': risk_results,
            'validation_report': validation_report
        }

    def run_simulation(self, num_steps: int):
        """Run the tunnel water inrush simulation."""
        self.logger.info(f"Starting simulation for {num_steps} steps")

        for step in range(num_steps):
            try:
                # Update fluid state
                self.coupling.cfd_solver.update_fluid_state()

                # Compute fluid forces on particles
                fluid_forces = self.coupling.cfd_solver.compute_fluid_forces(
                    [p['position'] for p in self.coupling.dem_solver.particles]
                )

                # Update particle states
                self.coupling.dem_solver.update_particle_states(fluid_forces)

                # Update erosion
                self._update_erosion()

                # Update visualization
                self._update_visualization(step)

                # Store statistics
                self._store_statistics(step)

                if step % 100 == 0:
                    self.logger.info(f"Completed step {step}/{num_steps}")
            except Exception as e:
                self.logger.error(f"Error during simulation step {step}: {e}")
                continue

    def _update_erosion(self):
        """Update erosion state based on fluid forces and particle properties."""
        # Get current fluid velocity and pressure
        fluid_data = self.coupling.cfd_solver.get_fluid_data()
        velocity_field = fluid_data['velocity_field']
        pressure_field = fluid_data['pressure_field']

        # Update erosion for each particle
        for particle in self.coupling.dem_solver.particles:
            # Check if particle position is within bounds
            if not self._is_position_in_bounds(particle['position']):
                continue

            # Interpolate fluid properties at particle position
            try:
                velocity = self.coupling.cfd_solver._interpolate_field(
                    particle['position'], velocity_field
                )
                pressure = self.coupling.cfd_solver._interpolate_field(
                    particle['position'], pressure_field
                )

                # Compute shear stress
                shear_stress = self._compute_shear_stress(
                    velocity, particle['radius'])

                # Update particle erosion state
                self._update_particle_erosion(particle, shear_stress)
            except IndexError as e:
                self.logger.warning(f"Index error during interpolation: {e}")
                continue

    def _is_position_in_bounds(self, position: np.ndarray) -> bool:
        """Check if a position is within the simulation domain bounds."""
        return (
            self.tunnel_walls['x_min'] <= position[0] <= self.tunnel_walls['x_max'] and
            self.tunnel_walls['y_min'] <= position[1] <= self.tunnel_walls['y_max'] and
            self.tunnel_walls['z_min'] <= position[2] <= self.tunnel_walls['z_max']
        )

    def _compute_shear_stress(self, velocity: np.ndarray, particle_radius: float) -> float:
        """Compute shear stress on particle surface."""
        velocity_magnitude = np.linalg.norm(velocity)
        reynolds = 2 * particle_radius * velocity_magnitude * \
            self.config['cfd']['fluid_density'] / \
            self.config['cfd']['fluid_viscosity']

        if reynolds < 1:
            # Stokes flow
            return 3 * self.config['cfd']['fluid_viscosity'] * velocity_magnitude / particle_radius
        else:
            # Turbulent flow
            return 0.5 * self.config['cfd']['fluid_density'] * velocity_magnitude**2

    def _update_particle_erosion(self, particle: Dict, shear_stress: float):
        """Update particle erosion state based on shear stress."""
        critical_shear_stress = self.config['erosion']['critical_shear_stress']
        erosion_rate = self.config['erosion']['erosion_rate_coefficient']

        if shear_stress > critical_shear_stress:
            # Compute erosion rate
            erosion = erosion_rate * (shear_stress - critical_shear_stress)

            # Update particle properties with boundary checking
            new_radius = particle['radius'] * (1 - erosion)
            if new_radius >= 0.1 * self.config['dem']['particle_radius']:
                particle['radius'] = new_radius
            else:
                particle['eroded'] = True

    def _update_visualization(self, step: int):
        """Update visualization with current simulation state."""
        if step % self.config['output']['visualization_interval'] == 0:
            try:
                # Get current particle positions
                particle_positions = [p['position']
                                      for p in self.coupling.dem_solver.particles]

                # Get current fluid velocity
                fluid_data = self.coupling.cfd_solver.get_fluid_data()
                fluid_velocity = fluid_data['velocity_field']

                # Update visualization
                self.visualizer.update_data(
                    particle_positions=particle_positions,
                    fluid_velocity=fluid_velocity,
                    time_step=step * self.config['simulation']['time_step']
                )
            except Exception as e:
                self.logger.warning(f"Error during visualization update: {e}")

    def _store_statistics(self, step: int):
        """Store simulation statistics."""
        try:
            # Store time step
            self.time_steps.append(
                step * self.config['simulation']['time_step'])

            # Compute erosion statistics
            eroded_particles = sum(
                1 for p in self.coupling.dem_solver.particles if p.get('eroded', False))
            self.erosion_stats.append(eroded_particles)

            # Compute bond health statistics
            if hasattr(self.coupling, 'bond_model'):
                bond_health = self.coupling.bond_model.get_bond_health()
                self.bond_health_stats.append(bond_health)

            # Store particle data
            self.particle_data.append(
                [p['position'] for p in self.coupling.dem_solver.particles])

            # Store fluid data
            fluid_data = self.coupling.cfd_solver.get_fluid_data()
            self.fluid_data.append(fluid_data)
        except Exception as e:
            self.logger.warning(f"Error during statistics storage: {e}")

    def save_results(self, filename: str = None):
        """Save simulation results."""
        if filename is None:
            filename = f"tunnel_water_inrush_{len(self.time_steps)}.npz"

        try:
            np.savez(
                filename,
                time_steps=np.array(self.time_steps),
                erosion_stats=np.array(self.erosion_stats),
                bond_health_stats=np.array(self.bond_health_stats),
                particle_data=np.array(self.particle_data),
                fluid_data=np.array(self.fluid_data)
            )

            self.logger.info(f"Results saved to {filename}")

            # Save visualization
            self.visualizer.save_animation()
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def close(self):
        """Clean up resources."""
        try:
            self.visualizer.close()
            self.logger.info("Resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def analyze_erosion_progress(self, simulation_results: Dict) -> Dict:
        """Analyze erosion progress and failure risk."""
        # Extract data
        time = np.array(simulation_results['time'])
        eroded_particles = np.array(simulation_results['eroded_particles'])
        bond_health = np.array(simulation_results['bond_health'])
        positions = np.array(simulation_results['particle_positions'])

        # Compute erosion rate
        erosion_rate = np.gradient(eroded_particles, time)

        # Compute failure risk
        risk = self._compute_failure_risk(
            eroded_particles, bond_health, positions)

        # Create analysis plots
        self._create_analysis_plots(time, eroded_particles, bond_health,
                                    erosion_rate, risk)

        return {
            'erosion_rate': erosion_rate,
            'failure_risk': risk,
            'critical_time': self._find_critical_time(erosion_rate, risk)
        }

    def _compute_failure_risk(self, eroded_particles: np.ndarray,
                              bond_health: np.ndarray,
                              positions: np.ndarray) -> np.ndarray:
        """Compute failure risk based on erosion and bond health."""
        # Normalize inputs
        eroded_norm = eroded_particles / np.max(eroded_particles)
        health_norm = 1 - bond_health  # Convert to damage

        # Compute distance from tunnel
        tunnel_center = self.case_params['tunnel']['center']
        distances = np.linalg.norm(positions - tunnel_center, axis=1)
        distance_norm = distances / np.max(distances)

        # Compute risk factors
        erosion_risk = eroded_norm
        bond_risk = health_norm
        geometry_risk = 1 - distance_norm

        # Combine risk factors
        risk = 0.4 * erosion_risk + 0.4 * bond_risk + 0.2 * geometry_risk

        return risk

    def _find_critical_time(self, erosion_rate: np.ndarray,
                            risk: np.ndarray) -> float:
        """Find critical time for failure."""
        # Find time when erosion rate exceeds threshold
        erosion_threshold = 0.1 * np.max(erosion_rate)
        erosion_critical = np.where(erosion_rate > erosion_threshold)[0]

        # Find time when risk exceeds threshold
        risk_threshold = 0.8
        risk_critical = np.where(risk > risk_threshold)[0]

        # Take the earlier of the two
        if len(erosion_critical) > 0 and len(risk_critical) > 0:
            return min(erosion_critical[0], risk_critical[0])
        elif len(erosion_critical) > 0:
            return erosion_critical[0]
        elif len(risk_critical) > 0:
            return risk_critical[0]
        else:
            return -1

    def _create_analysis_plots(self, time: np.ndarray,
                               eroded_particles: np.ndarray,
                               bond_health: np.ndarray,
                               erosion_rate: np.ndarray,
                               risk: np.ndarray):
        """Create analysis plots."""
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot erosion progress
        ax1.plot(time, eroded_particles, 'b-')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Number of Eroded Particles')
        ax1.set_title('Erosion Progress')
        ax1.grid(True, alpha=0.3)

        # Plot bond health
        ax2.plot(time, bond_health, 'r-')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Bond Health')
        ax2.set_title('Bond Degradation')
        ax2.grid(True, alpha=0.3)

        # Plot erosion rate
        ax3.plot(time, erosion_rate, 'g-')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Erosion Rate (particles/s)')
        ax3.set_title('Erosion Rate')
        ax3.grid(True, alpha=0.3)

        # Plot failure risk
        ax4.plot(time, risk, 'k-')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Failure Risk')
        ax4.set_title('Failure Risk Assessment')
        ax4.grid(True, alpha=0.3)

        # Save figure
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tunnel_inrush_analysis.png')
        plt.close()

    def generate_case_study_report(self, analysis_results: Dict):
        """Generate a comprehensive case study report."""
        report = {
            'timestamp': str(np.datetime64('now')),
            'case_parameters': self.case_params,
            'analysis_results': analysis_results
        }
        # Save report
        with open(self.output_dir / 'tunnel_inrush_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        self.logger.info("Case study report generated successfully")
