"""
Mitigation measures module for simulating various mitigation strategies
and their effectiveness in preventing tunnel water inrush.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json

class MitigationManager:
    def __init__(self, config: Dict):
        """Initialize the mitigation manager.
        
        Args:
            config: Configuration dictionary containing mitigation settings
        """
        self.config = config
        self.output_dir = Path(config['output']['directory']) / 'mitigation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load mitigation strategies
        self.strategies = self._load_strategies()
        
        # Initialize cost tracking
        self.costs = {}
        self.effectiveness = {}
        
        self.logger.info("Initialized mitigation manager")
    
    def _load_strategies(self) -> Dict:
        """Load mitigation strategies from configuration."""
        strategies = {}
        for strategy in self.config['mitigation']['strategies']:
            strategies[strategy['name']] = {
                'type': strategy['type'],
                'parameters': strategy['parameters'],
                'cost_model': strategy['cost_model'],
                'effectiveness_model': strategy['effectiveness_model']
            }
        return strategies
    
    def apply_strategy(self, strategy_name: str, 
                      current_state: Dict) -> Dict:
        """Apply a mitigation strategy to the current state.
        
        Args:
            strategy_name: Name of the strategy to apply
            current_state: Current simulation state
            
        Returns:
            Updated state after applying the strategy
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        updated_state = current_state.copy()
        
        # Apply strategy-specific modifications
        if strategy['type'] == 'grouting':
            updated_state = self._apply_grouting(
                updated_state, strategy['parameters'])
        elif strategy['type'] == 'drainage':
            updated_state = self._apply_drainage(
                updated_state, strategy['parameters'])
        elif strategy['type'] == 'support':
            updated_state = self._apply_support(
                updated_state, strategy['parameters'])
        
        # Update costs and effectiveness
        self._update_metrics(strategy_name, updated_state)
        
        return updated_state
    
    def _apply_grouting(self, state: Dict, 
                       parameters: Dict) -> Dict:
        """Apply grouting mitigation strategy.
        
        Args:
            state: Current simulation state
            parameters: Grouting parameters
            
        Returns:
            Updated state after grouting
        """
        # Modify permeability
        state['permeability'] *= parameters['permeability_reduction']
        
        # Add grout strength
        state['strength'] += parameters['strength_increase']
        
        # Update porosity
        state['porosity'] *= (1 - parameters['porosity_reduction'])
        
        return state
    
    def _apply_drainage(self, state: Dict, 
                       parameters: Dict) -> Dict:
        """Apply drainage mitigation strategy.
        
        Args:
            state: Current simulation state
            parameters: Drainage parameters
            
        Returns:
            Updated state after drainage
        """
        # Add drainage channels
        state['drainage_channels'] = parameters['channel_positions']
        
        # Modify pressure gradient
        state['pressure_gradient'] *= parameters['pressure_reduction']
        
        # Update flow rate
        state['flow_rate'] *= parameters['flow_reduction']
        
        return state
    
    def _apply_support(self, state: Dict, 
                      parameters: Dict) -> Dict:
        """Apply support mitigation strategy.
        
        Args:
            state: Current simulation state
            parameters: Support parameters
            
        Returns:
            Updated state after support
        """
        # Add support elements
        state['support_elements'] = parameters['element_positions']
        
        # Modify stress distribution
        state['stress'] *= parameters['stress_reduction']
        
        # Update deformation
        state['deformation'] *= parameters['deformation_reduction']
        
        return state
    
    def _update_metrics(self, strategy_name: str, 
                       state: Dict) -> None:
        """Update cost and effectiveness metrics.
        
        Args:
            strategy_name: Name of the strategy
            state: Current simulation state
        """
        strategy = self.strategies[strategy_name]
        
        # Calculate costs
        self.costs[strategy_name] = self._calculate_costs(
            strategy['cost_model'], state)
        
        # Calculate effectiveness
        self.effectiveness[strategy_name] = self._calculate_effectiveness(
            strategy['effectiveness_model'], state)
    
    def _calculate_costs(self, cost_model: Dict, 
                        state: Dict) -> float:
        """Calculate implementation costs.
        
        Args:
            cost_model: Cost model parameters
            state: Current simulation state
            
        Returns:
            Total implementation cost
        """
        total_cost = 0.0
        
        # Material costs
        total_cost += cost_model['material_cost'] * state.get('volume', 1.0)
        
        # Labor costs
        total_cost += cost_model['labor_cost'] * state.get('duration', 1.0)
        
        # Equipment costs
        total_cost += cost_model['equipment_cost']
        
        # Maintenance costs
        total_cost += cost_model['maintenance_cost'] * state.get('lifetime', 1.0)
        
        return total_cost
    
    def _calculate_effectiveness(self, effectiveness_model: Dict, 
                               state: Dict) -> float:
        """Calculate strategy effectiveness.
        
        Args:
            effectiveness_model: Effectiveness model parameters
            state: Current simulation state
            
        Returns:
            Effectiveness score (0-1)
        """
        effectiveness = 0.0
        
        # Pressure reduction effectiveness
        pressure_effect = 1 - (state.get('pressure', 0.0) / 
                             effectiveness_model['base_pressure'])
        effectiveness += pressure_effect * effectiveness_model['pressure_weight']
        
        # Flow reduction effectiveness
        flow_effect = 1 - (state.get('flow_rate', 0.0) / 
                          effectiveness_model['base_flow'])
        effectiveness += flow_effect * effectiveness_model['flow_weight']
        
        # Stability improvement effectiveness
        stability_effect = (state.get('stability_factor', 0.0) / 
                          effectiveness_model['target_stability'])
        effectiveness += stability_effect * effectiveness_model['stability_weight']
        
        return min(max(effectiveness, 0.0), 1.0)
    
    def optimize_strategy(self, constraints: Dict) -> Dict:
        """Optimize mitigation strategy based on constraints.
        
        Args:
            constraints: Optimization constraints
            
        Returns:
            Optimal strategy parameters
        """
        best_strategy = None
        best_score = float('-inf')
        
        for strategy_name, strategy in self.strategies.items():
            # Check if strategy meets constraints
            if not self._check_constraints(strategy, constraints):
                continue
            
            # Calculate optimization score
            score = self._calculate_optimization_score(
                strategy, constraints)
            
            if score > best_score:
                best_score = score
                best_strategy = {
                    'name': strategy_name,
                    'parameters': strategy['parameters'],
                    'score': score
                }
        
        return best_strategy
    
    def _check_constraints(self, strategy: Dict, 
                          constraints: Dict) -> bool:
        """Check if strategy meets constraints.
        
        Args:
            strategy: Strategy parameters
            constraints: Optimization constraints
            
        Returns:
            True if constraints are met
        """
        # Check cost constraints
        if self.costs[strategy['name']] > constraints['max_cost']:
            return False
        
        # Check effectiveness constraints
        if self.effectiveness[strategy['name']] < constraints['min_effectiveness']:
            return False
        
        # Check implementation time constraints
        if strategy['parameters'].get('implementation_time', 0) > constraints['max_time']:
            return False
        
        return True
    
    def _calculate_optimization_score(self, strategy: Dict, 
                                    constraints: Dict) -> float:
        """Calculate optimization score for strategy.
        
        Args:
            strategy: Strategy parameters
            constraints: Optimization constraints
            
        Returns:
            Optimization score
        """
        # Cost component
        cost_score = 1 - (self.costs[strategy['name']] / constraints['max_cost'])
        
        # Effectiveness component
        effectiveness_score = self.effectiveness[strategy['name']]
        
        # Time component
        time_score = 1 - (strategy['parameters'].get('implementation_time', 0) / 
                         constraints['max_time'])
        
        # Weighted combination
        score = (constraints['cost_weight'] * cost_score +
                constraints['effectiveness_weight'] * effectiveness_score +
                constraints['time_weight'] * time_score)
        
        return score
    
    def plot_results(self) -> None:
        """Plot mitigation results."""
        # Create cost-effectiveness plot
        plt.figure(figsize=(10, 6))
        for strategy_name in self.strategies:
            plt.scatter(self.costs[strategy_name],
                       self.effectiveness[strategy_name],
                       label=strategy_name)
        
        plt.xlabel('Cost')
        plt.ylabel('Effectiveness')
        plt.title('Cost-Effectiveness Analysis')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'cost_effectiveness.png')
        plt.close()
        
        # Create comparison plots for each metric
        metrics = ['pressure', 'flow_rate', 'stability']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for strategy_name in self.strategies:
                plt.bar(strategy_name,
                       self.effectiveness[strategy_name])
            
            plt.xlabel('Strategy')
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} Comparison')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{metric}_comparison.png')
            plt.close()
    
    def save_results(self) -> None:
        """Save mitigation results to file."""
        results = {
            'costs': self.costs,
            'effectiveness': self.effectiveness,
            'strategies': self.strategies
        }
        
        with open(self.output_dir / 'mitigation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        self.logger.info("Saved mitigation results") 