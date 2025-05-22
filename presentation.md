# Particle Flow and Internal Erosion Simulation
## Project Demonstration

### 1. Introduction (5 minutes)
- **Problem Statement**
  - Water flow can cause soil/rock structures to fail
  - Understanding this process is crucial for engineering safety
  - Real-world impact: Tunnel collapses, dam failures, etc.

- **Project Goals**
  - Develop new models for particle bonding
  - Create efficient simulation tools
  - Validate against real-world data
  - Apply to engineering problems

### 2. New Constitutive Model (10 minutes)
- **What is it?**
  - Mathematical model describing how particles stick together
  - Accounts for water flow effects
  - More accurate than previous models

- **Demonstration**
  - Show bond degradation visualization
  - Explain how water flow affects bonds
  - Compare with traditional models

### 3. Coarse-Grained Model (10 minutes)
- **What is it?**
  - Makes large-scale simulations possible
  - Reduces computational cost
  - Maintains accuracy

- **Demonstration**
  - Show fine vs. coarse scale comparison
  - Explain scaling process
  - Demonstrate efficiency gains

### 4. CFD-DEM Coupling (10 minutes)
- **What is it?**
  - Combines fluid flow and particle motion
  - Simulates real-world conditions
  - Accounts for all important forces

- **Demonstration**
  - Show fluid-particle interaction
  - Visualize forces
  - Explain coupling process

### 5. Validation Framework (10 minutes)
- **What is it?**
  - Ensures model accuracy
  - Compares with experimental data
  - Provides confidence in results

- **Demonstration**
  - Show validation metrics
  - Compare with experimental data
  - Explain error analysis

### 6. Case Study: Tunnel Water Inrush (15 minutes)
- **What is it?**
  - Real-world application
  - Simulates tunnel flooding
  - Helps prevent disasters

- **Demonstration**
  - Show simulation results
  - Explain key findings
  - Discuss practical applications

### 7. Q&A Session (10 minutes)
- Open floor for questions
- Technical details
- Applications
- Future work

### Running the Demonstration
1. Ensure all dependencies are installed
2. Run the demonstration script:
   ```bash
   python demo.py
   ```
3. Results will be saved in the 'results' directory
4. Use the visualizations to explain each component

### Key Points to Emphasize
1. **Innovation**
   - New constitutive model
   - Efficient coarse-grained approach
   - Comprehensive validation

2. **Practical Impact**
   - Engineering applications
   - Safety improvements
   - Cost savings

3. **Technical Achievement**
   - Complex physics
   - Computational efficiency
   - Validation accuracy

### Tips for Presentation
1. Start with the big picture
2. Use visualizations effectively
3. Explain technical terms
4. Connect to real-world examples
5. Encourage questions 