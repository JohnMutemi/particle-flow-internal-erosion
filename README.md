# DEM-based Internal Erosion Model

This project implements a comprehensive model for simulating internal erosion processes in granular materials by coupling Discrete Element Method (DEM) with Computational Fluid Dynamics (CFD).

## Overview

The model simulates internal erosion processes in granular materials using a customized approach that:
- Implements a new constitutive model for particle bonding under seepage erosion
- Uses coarse-grained technology for large-scale simulations
- Provides CFD-DEM coupling for fluid-particle interactions
- Validates against experimental data and real-world case studies

## Key Deliverables

1. New Constitutive Model
   - Parallel bond model with seepage erosion effects
   - Bond degradation under fluid action
   - Implementation in OpenFOAM-PFC

2. Coarse-grained Technology
   - Scaled-up particle interactions
   - Parameter calibration for large-scale simulations
   - Validation against fine-scale models

3. CFD-DEM Coupling Framework
   - Fluid flow simulation using CFD
   - Particle interactions using DEM
   - Erosion and bond degradation mechanisms

4. Validation and Case Studies
   - Experimental verification through seepage tests
   - Tunnel water inrush simulation
   - Engineering-scale simulations
   - Mitigation measure analysis

## Features

- 3D particle flow simulation
- Fluid-particle coupling using CFD-DEM
- Multiple erosion criteria implementation:
  - Grading criteria (particle size distribution)
  - Hydraulic criteria (seepage forces)
  - Stress criteria (microscopic interactions)
- Simulation of both suffosion and suffusion processes
- Heterogeneity development tracking
- Mechanical behavior evolution
- Time-dependent process simulation
- Support for different pressure gradients and boundary conditions

## Requirements

- Python 3.8+
- OpenFOAM
- PFC (Particle Flow Code)
- NumPy
- SciPy
- Matplotlib
- PyVista (for 3D visualization)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Configure simulation parameters in `config.yaml`
2. Run the simulation:
```bash
python src/main.py
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── main.py
│   ├── dem/
│   │   ├── particle.py
│   │   ├── bonds.py
│   │   └── coarse_grained.py
│   ├── cfd/
│   │   ├── fluid.py
│   │   └── coupling.py
│   ├── erosion/
│   │   ├── constitutive/
│   │   │   ├── parallel_bond.py
│   │   │   └── degradation.py
│   │   ├── criteria/
│   │   │   ├── grading.py
│   │   │   ├── hydraulic.py
│   │   │   └── stress.py
│   │   └── process/
│   │       ├── suffosion.py
│   │       └── suffusion.py
│   └── validation/
│       ├── experimental.py
│       └── case_studies.py
└── tests/
    └── test_erosion.py
```

## Implementation Approach

1. Constitutive Model Development
   - Implement parallel bond model
   - Add seepage erosion effects
   - Integrate with OpenFOAM-PFC

2. Coarse-grained Implementation
   - Develop scaling algorithms
   - Implement parameter calibration
   - Create validation framework

3. CFD-DEM Coupling
   - Set up fluid simulation
   - Implement particle interactions
   - Add erosion mechanisms

4. Validation Framework
   - Create experimental test suite
   - Implement case study scenarios
   - Add mitigation analysis tools

## Citation

If you use this code in your research, please cite:
```
Liu, F., Singh, J., Chen, C., Li, Y., & Wang, G. (2025). Hydrological and mechanical behavior of granular materials subjected to internal erosion: A review. KSCE Journal of Civil Engineering, 29, 100047.
```

## License

MIT License
