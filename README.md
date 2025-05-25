# CFD-DEM Simulation Project | CFD-DEM 模拟项目

A comprehensive Computational Fluid Dynamics (CFD) and Discrete Element Method (DEM) coupling framework for simulating particle-fluid interactions, with a focus on tunnel water inrush scenarios. This project implements an advanced CFD-DEM coupling framework for simulating fluid-particle interactions in geotechnical applications, featuring a novel constitutive model for particle bonding that accounts for seepage erosion effects, along with a coarse-grained approach for large-scale simulations.

一个全面的计算流体动力学(CFD)和离散元方法(DEM)耦合框架，用于模拟颗粒-流体相互作用，特别关注隧道突水场景。本项目实现了一个先进的 CFD-DEM 耦合框架，用于模拟岩土工程应用中的流体-颗粒相互作用，具有考虑渗流侵蚀效应的新型颗粒粘结本构模型，以及用于大规模模拟的粗粒化方法。

## Project Structure | 项目结构

```
particle_flow_code/
├── src/                           # Source code directory | 源代码目录
│   ├── models/                    # Physical models and equations | 物理模型和方程
│   │   └── bond_model.py         # Seepage bond model implementation | 渗流粘结模型实现
│   ├── coarse_grained/           # Coarse-grained simulation methods | 粗粒化模拟方法
│   │   └── coarse_grained_model.py
│   ├── coupling/                 # CFD-DEM coupling algorithms | CFD-DEM耦合算法
│   │   └── coupling_manager.py
│   ├── validation/              # Validation and verification tools | 验证和确认工具
│   │   └── validation_manager.py
│   ├── case_studies/           # Case study implementations | 案例研究实现
│   │   └── tunnel_water_inrush.py
│   └── visualization/           # Visualization tools | 可视化工具
│       └── visualizer.py
│
├── tests/                        # Test suite | 测试套件
│   ├── test_bond_model.py       # Bond model tests | 粘结模型测试
│   ├── test_coarse_grained.py   # Coarse-grained model tests | 粗粒化模型测试
│   ├── test_coupling.py         # Coupling algorithm tests | 耦合算法测试
│   └── test_validation.py       # Validation framework tests | 验证框架测试
│
├── data/                         # Data directory | 数据目录
│   ├── input/                    # Input data files | 输入数据文件
│   └── output/                   # Simulation output files | 模拟输出文件
│
├── results/                      # Simulation results | 模拟结果
│   ├── visualizations/           # Generated visualizations | 生成的可视化
│   └── analysis/                 # Analysis results | 分析结果
│
├── dashboard.py                  # Interactive dashboard | 交互式仪表板
├── demo.py                       # Demo simulation script | 演示模拟脚本
├── config.yaml                   # Configuration file | 配置文件
├── requirements.txt              # Python dependencies | Python依赖
└── README.md                     # This file | 本文件
```

## Technology Stack | 技术栈

- **Python**: Chosen for its extensive scientific computing ecosystem, ease of integration with numerical libraries, and scalability through parallel processing capabilities.
- **Key Libraries**:
  - NumPy: For numerical computations | 用于数值计算
  - Streamlit: For interactive visualization and parameter studies | 用于交互式可视化和参数研究
  - Plotly: For advanced data visualization | 用于高级数据可视化
  - PyYAML: For configuration management | 用于配置管理
  - Pandas: For data analysis and manipulation | 用于数据分析和操作
  - SciPy: For scientific computing | 用于科学计算
  - Matplotlib: For plotting and visualization | 用于绘图和可视化
  - scikit-learn: For machine learning and clustering | 用于机器学习和聚类
  - PyVista: For 3D visualization | 用于 3D 可视化

## Installation | 安装

### Prerequisites | 前提条件

- Python 3.8 or higher | Python 3.8 或更高版本
- Git
- pip (Python package manager) | pip (Python 包管理器)

### Setup Steps | 设置步骤

1. **Clone the Repository | 克隆仓库**

   ```bash
   git clone https://github.com/JohnMutemi/particle-flow-internal-erosion.git
   cd particle-flow-internal-erosion
   ```

2. **Create and Activate Virtual Environment | 创建并激活虚拟环境**

   ```bash
   # Create virtual environment | 创建虚拟环境
   python -m venv .venv

   # Activate virtual environment | 激活虚拟环境
   # On Windows:
   .venv\Scripts\activate
   # On Unix or MacOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies | 安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install the Package in Development Mode | 以开发模式安装包**
   ```bash
   pip install -e .
   ```

## Usage | 使用方法

### Running Simulations | 运行模拟

1. **Basic Simulation | 基本模拟**

   ```bash
   python scripts/demo.py
   ```

   This will: | 这将：

   - Demonstrate the bond model | 演示粘结模型
   - Show coarse-grained modeling | 展示粗粒化建模
   - Run CFD-DEM coupling | 运行 CFD-DEM 耦合
   - Perform validation | 执行验证

2. **Interactive Dashboard | 交互式仪表板**

   ```bash
   streamlit run scripts/dashboard.py
   ```

   The dashboard provides: | 仪表板提供：

   - Interactive parameter modification | 交互式参数修改
   - Real-time visualization | 实时可视化
   - Detailed analysis of fluidity effects | 流动性效应详细分析
   - Validation results | 验证结果

3. **Custom Simulation | 自定义模拟**

   ```python
   from src.main import Simulation

   sim = Simulation(config_path='config.yaml')
   sim.run()
   ```

### Visualization | 可视化

The project includes several visualization options: | 项目包含多种可视化选项：

1. **Real-time Visualization | 实时可视化**

   - Available through the dashboard interface | 通过仪表板界面可用
   - Supports particle tracking and flow field visualization | 支持颗粒追踪和流场可视化

2. **Results Analysis | 结果分析**

   - Access through the dashboard's analysis section | 通过仪表板的分析部分访问
   - Includes fluidity parameters, interaction analysis, and stability metrics | 包括流动性参数、相互作用分析和稳定性指标

3. **Presentation Generation | 演示生成**
   ```bash
   python create_presentation.py
   ```

### Dashboard Deployment | 仪表板部署

#### Local Deployment | 本地部署

1. Install the required dependencies: | 安装所需依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. Run the dashboard: | 运行仪表板：
   ```bash
   streamlit run dashboard.py
   ```

#### Streamlit Cloud Deployment | Streamlit 云部署

1. Create a GitHub repository and push your code: | 创建 GitHub 仓库并推送代码：

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

## Theoretical Background | 理论基础

### 1. Constitutive Model Development | 本构模型开发

- **Innovation**: New constitutive model for particle bonding | 创新：新型颗粒粘结本构模型
- **Features**: | 特点：
  - Seepage erosion effects | 渗流侵蚀效应
  - Bond degradation under fluid flow | 流体流动下的粘结退化
  - Parallel bond model integration | 平行粘结模型集成

### 2. Coarse-Grained Technology | 粗粒化技术

- **Purpose**: Enable large-scale engineering simulations | 目的：实现大规模工程模拟
- **Features**: | 特点：
  - Scaled particle interactions | 缩放颗粒相互作用
  - Parameter calibration | 参数校准
  - Validation against fine-scale models | 与精细尺度模型验证

### 3. CFD-DEM Coupling | CFD-DEM 耦合

- **Framework**: Integrated fluid-particle interaction | 框架：集成流体-颗粒相互作用
- **Components**: | 组件：
  - Fluid flow simulation (CFD) | 流体流动模拟(CFD)
  - Particle interaction (DEM) | 颗粒相互作用(DEM)
  - Erosion and degradation mechanisms | 侵蚀和退化机制

### 4. Validation and Case Studies | 验证和案例研究

- **Experimental Verification**: | 实验验证：
  - Seepage erosion tests | 渗流侵蚀测试
  - Three-axis seepage simulation | 三轴渗流模拟
  - Tunnel water inrush scenarios | 隧道突水场景

### 5. Three-Axis Seepage Simulation | 三轴渗流模拟

- **Purpose**: | 目的：

  - Comprehensive analysis of seepage effects in three dimensions | 三维渗流效应的综合分析
  - Study of anisotropic flow behavior | 各向异性流动行为研究
  - Evaluation of structural stability under multi-directional flow | 多向流动下的结构稳定性评估

- **Features**: | 特点：

  - Multi-directional flow simulation | 多向流动模拟
  - Anisotropic permeability analysis | 各向异性渗透性分析
  - Stress-strain relationship under seepage | 渗流条件下的应力-应变关系
  - Pore pressure distribution visualization | 孔隙压力分布可视化

- **Applications**: | 应用：
  - Geotechnical engineering | 岩土工程
  - Underground construction | 地下工程
  - Slope stability analysis | 边坡稳定性分析
  - Dam safety assessment | 大坝安全评估

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

## Acknowledgments

- Wang et al. (2020) for CFD-DEM coupling research
- Streamlit for the dashboard framework
- Plotly for visualization components

## Contributing | 贡献

We welcome contributions to this project! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

欢迎对本项目做出贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解更多详情。

## License | 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

本项目采用 MIT 许可证 - 详情请查看[LICENSE](LICENSE)文件。

## Contact | 联系方式

For questions and support, please open an issue in the GitHub repository or contact the maintainers.

如有问题和需要支持，请在 GitHub 仓库中提出 issue 或联系维护者。
