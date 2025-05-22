# CFD-DEM Framework for Geotechnical Applications

## Overview
This project implements an advanced CFD-DEM (Computational Fluid Dynamics-Discrete Element Method) coupling framework for simulating fluid-particle interactions in geotechnical applications. The framework features a novel constitutive model for particle bonding that accounts for seepage erosion effects, along with a coarse-grained approach for large-scale simulations.

## Technology Stack
- **Python**: Chosen for its extensive scientific computing ecosystem, ease of integration with numerical libraries, and scalability through parallel processing capabilities.
- **Key Libraries**:
  - NumPy: For numerical computations
  - Streamlit: For interactive visualization and parameter studies
  - Plotly: For advanced data visualization
  - PyYAML: For configuration management
  - Pandas: For data analysis and manipulation

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- pip (Python package manager)

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/JohnMutemi/particle_flow_code.git
   cd particle_flow_code
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On Unix or MacOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
particle_flow_code/
├── src/
│   ├── models/
│   │   └── bond_model.py
│   ├── coarse_grained/
│   │   └── coarse_grained_model.py
│   ├── coupling/
│   │   └── coupling_manager.py
│   ├── validation/
│   │   └── validation_manager.py
│   └── visualization/
│       └── visualizer.py
├── tests/
├── data/
├── results/
├── demo.py
├── dashboard.py
├── config.yaml
└── requirements.txt
```

## Usage

### Running the Demo
```bash
python demo.py
```
This will:
- Demonstrate the bond model
- Show coarse-grained modeling
- Run CFD-DEM coupling
- Perform validation

### Running the Dashboard
```bash
streamlit run dashboard.py
```
The dashboard provides:
- Interactive parameter modification
- Real-time visualization
- Detailed analysis of fluidity effects
- Validation results

## Theoretical Background

### 1. Constitutive Model Development
- **Innovation**: New constitutive model for particle bonding
- **Features**:
  - Seepage erosion effects
  - Bond degradation under fluid flow
  - Parallel bond model integration

### 2. Coarse-Grained Technology
- **Purpose**: Enable large-scale engineering simulations
- **Features**:
  - Scaled particle interactions
  - Parameter calibration
  - Validation against fine-scale models

### 3. CFD-DEM Coupling
- **Framework**: Integrated fluid-particle interaction
- **Components**:
  - Fluid flow simulation (CFD)
  - Particle interaction (DEM)
  - Erosion and degradation mechanisms

### 4. Validation and Case Studies
- **Experimental Verification**:
  - Seepage erosion tests
  - Comparison with numerical results
- **Case Studies**:
  - Tunnel water inrush simulation
  - Engineering-scale modeling
  - Mitigation analysis

## Key Features
1. **Interactive Parameter Studies**
   - Real-time parameter modification
   - Immediate visualization of effects
   - Comprehensive analysis tools

2. **Advanced Visualization**
   - Flow patterns
   - Particle behavior
   - Bond degradation
   - Statistical analysis

3. **Validation Framework**
   - Experimental data comparison
   - Statistical validation
   - Sensitivity analysis

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


