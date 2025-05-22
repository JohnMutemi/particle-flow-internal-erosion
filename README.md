# CFD-DEM Simulation Project

A comprehensive Computational Fluid Dynamics (CFD) and Discrete Element Method (DEM) coupling framework for simulating particle-fluid interactions, with a focus on tunnel water inrush scenarios. This project implements an advanced CFD-DEM coupling framework for simulating fluid-particle interactions in geotechnical applications, featuring a novel constitutive model for particle bonding that accounts for seepage erosion effects, along with a coarse-grained approach for large-scale simulations.

## Project Structure

```
particle_flow_code/
├── src/                           # Source code directory
│   ├── cfd/                       # CFD-related modules
│   ├── dem/                       # DEM-related modules
│   ├── coupling/                  # CFD-DEM coupling algorithms
│   ├── coarse_grained/           # Coarse-grained simulation methods
│   ├── models/                    # Physical models and equations
│   ├── validation/               # Validation and verification tools
│   ├── mitigation/               # Risk mitigation strategies
│   ├── parallel/                 # Parallel computing implementations
│   ├── experimental/             # Experimental data processing
│   ├── calibration/              # Model calibration tools
│   ├── case_studies/            # Case study implementations
│   ├── visualization/            # Visualization tools
│   ├── erosion/                  # Erosion modeling
│   └── main.py                   # Main simulation entry point
│
├── tests/                        # Test suite
│   ├── test_cfd.py              # CFD module tests
│   ├── test_dem.py              # DEM module tests
│   ├── test_coupling.py         # Coupling algorithm tests
│   ├── test_erosion.py          # Erosion model tests
│   ├── test_force_models.py     # Force model tests
│   ├── test_main.py             # Main module tests
│   ├── test_real_time_visualizer.py  # Visualization tests
│   ├── test_tunnel_water_inrush.py   # Tunnel water inrush tests
│   ├── test_visualization.py     # Visualization module tests
│   └── conftest.py              # Test configuration
│
├── data/                         # Data directory
│   ├── input/                    # Input data files
│   └── output/                   # Simulation output files
│
├── results/                      # Simulation results
│   ├── visualizations/           # Generated visualizations
│   └── analysis/                 # Analysis results
│
├── dashboard.py                  # Interactive dashboard
├── demo.py                       # Demo simulation script
├── create_presentation.py        # Presentation generation script
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup file
├── pytest.ini                    # PyTest configuration
└── README.md                     # This file
```

## Technology Stack

- **Python**: Chosen for its extensive scientific computing ecosystem, ease of integration with numerical libraries, and scalability through parallel processing capabilities.
- **Key Libraries**:
  - NumPy: For numerical computations
  - Streamlit: For interactive visualization and parameter studies
  - Plotly: For advanced data visualization
  - PyYAML: For configuration management
  - Pandas: For data analysis and manipulation
  - SciPy: For scientific computing
  - Matplotlib: For plotting and visualization
  - scikit-learn: For machine learning and clustering
  - PyVista: For 3D visualization

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- pip (Python package manager)

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/JohnMutemi/particle-flow-internal-erosion.git
   cd particle-flow-internal-erosion
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

4. **Install the Package in Development Mode**
   ```bash
   pip install -e .
   ```

## Usage

### Running Simulations

1. **Basic Simulation**
   ```bash
   python demo.py
   ```
   This will:
   - Demonstrate the bond model
   - Show coarse-grained modeling
   - Run CFD-DEM coupling
   - Perform validation

2. **Interactive Dashboard**
   ```bash
   streamlit run dashboard.py
   ```
   The dashboard provides:
   - Interactive parameter modification
   - Real-time visualization
   - Detailed analysis of fluidity effects
   - Validation results

3. **Custom Simulation**
   ```python
   from src.main import Simulation
   
   sim = Simulation(config_path='config.yaml')
   sim.run()
   ```

### Visualization

The project includes several visualization options:

1. **Real-time Visualization**
   - Available through the dashboard interface
   - Supports particle tracking and flow field visualization

2. **Results Analysis**
   - Access through the dashboard's analysis section
   - Includes fluidity parameters, interaction analysis, and stability metrics

3. **Presentation Generation**
   ```bash
   python create_presentation.py
   ```

### Dashboard Deployment

#### Local Deployment
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

#### Streamlit Cloud Deployment
1. Create a GitHub repository and push your code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file path (`dashboard.py`)
6. Click "Deploy"

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

4. **Parallel Computing Support**
   - Large-scale simulations
   - Distributed processing
   - Performance optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your license here]

## Contact

[Add contact information]


