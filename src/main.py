"""
Main entry point for the DEM-based Internal Erosion Model.
"""

import logging
import yaml
from pathlib import Path
import numpy as np
from src.dem.solver import DEMSolver
from src.cfd.solver import CFDSolver
from src.dem.constitutive.bond_model import SeepageErosionBondModel
from src.visualization.visualizer import SimulationVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def initialize_simulation(config: dict):
    """Initialize simulation components based on configuration."""
    logger.info("Initializing simulation components...")

    # Initialize solvers
    dem_solver = DEMSolver(config)
    cfd_solver = CFDSolver(config)
    erosion_model = SeepageErosionBondModel(config)

    # Initialize particles
    num_particles = config['simulation']['initial_particles']
    dem_solver.initialize_particles(num_particles)

    # Initialize fluid fields
    cfd_solver.initialize_fields()

    return dem_solver, cfd_solver, erosion_model


def run_simulation(dem_solver: DEMSolver,
                   cfd_solver: CFDSolver,
                   erosion_model: SeepageErosionBondModel,
                   config: dict):
    """Run the coupled DEM-CFD-erosion simulation."""
    time_steps = config['simulation']['time_steps']
    output_interval = config['simulation']['output_interval']

    # Initialize output directory
    output_dir = Path(config['output']['directory'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    visualizer = SimulationVisualizer(config)

    # Initialize results storage
    results = {
        'time': [],
        'eroded_particles': [],
        'bond_health': [],
        'fluid_forces': [],
        'particle_positions': [],
        'fluid_velocity': [],
        'fluid_pressure': []
    }

    for step in range(time_steps):
        logger.info(f"Simulation step {step + 1}/{time_steps}")

        # Update fluid state
        cfd_solver.update_fluid_state()

        # Get current particle data
        particle_data = dem_solver.get_particle_data()
        particle_positions = particle_data['positions']
        particle_velocities = particle_data['velocities']

        # Compute fluid forces on particles
        fluid_forces = cfd_solver.compute_fluid_forces(particle_positions)

        # Update particle states with fluid forces
        dem_solver.apply_forces(fluid_forces)
        dem_solver.step()

        # Compute erosion and update particle bonds
        erosion_rate = erosion_model.compute_erosion_rate(
            particle_positions,
            particle_velocities,
            cfd_solver.get_fluid_data()
        )
        erosion_model.update_bonds(
            erosion_rate, config['simulation']['time_step'])

        # Store results
        current_time = step * config['simulation']['time_step']
        results['time'].append(current_time)
        results['eroded_particles'].append(
            erosion_model.get_eroded_particles())
        results['bond_health'].append(erosion_model.get_bond_health())
        results['fluid_forces'].append(fluid_forces)
        results['particle_positions'].append(particle_positions)

        fluid_data = cfd_solver.get_fluid_data()
        results['fluid_velocity'].append(fluid_data['velocity_field'])
        results['fluid_pressure'].append(fluid_data['pressure_field'])

        # Output results if needed
        if (step + 1) % output_interval == 0:
            logger.info(f"Outputting results at step {step + 1}")

            # Save numerical results
            np.savez(
                output_dir / f'results_step_{step+1}.npz',
                time=current_time,
                eroded_particles=results['eroded_particles'][-1],
                bond_health=results['bond_health'][-1],
                fluid_forces=results['fluid_forces'][-1],
                particle_positions=results['particle_positions'][-1],
                fluid_velocity=results['fluid_velocity'][-1],
                fluid_pressure=results['fluid_pressure'][-1]
            )

            # Generate visualizations
            visualizer.plot_particle_distribution(
                particle_positions,
                erosion_model.get_bond_health(),
                output_dir / f'particles_step_{step+1}.png'
            )

            visualizer.plot_fluid_field(
                fluid_data['velocity_field'],
                fluid_data['pressure_field'],
                output_dir / f'fluid_step_{step+1}.png'
            )

            visualizer.plot_erosion_progress(
                results['time'],
                results['eroded_particles'],
                results['bond_health'],
                output_dir / f'erosion_step_{step+1}.png'
            )

    # Save final results
    np.savez(
        output_dir / 'final_results.npz',
        time=np.array(results['time']),
        eroded_particles=np.array(results['eroded_particles']),
        bond_health=np.array(results['bond_health']),
        fluid_forces=np.array(results['fluid_forces']),
        particle_positions=np.array(results['particle_positions']),
        fluid_velocity=np.array(results['fluid_velocity']),
        fluid_pressure=np.array(results['fluid_pressure'])
    )

    logger.info("Simulation completed successfully")
    return results


def main():
    """Main entry point for the simulation."""
    try:
        logger.info("Starting DEM-based Internal Erosion Model...")

        # Load configuration
        config_path = Path("config.yaml")
        config = load_config(config_path)

        # Initialize simulation
        dem_solver, cfd_solver, erosion_model = initialize_simulation(config)

        # Run simulation
        results = run_simulation(dem_solver, cfd_solver, erosion_model, config)

        logger.info("Simulation completed successfully.")

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()
