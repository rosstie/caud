import numpy as np
import pandas as pd
from typing import Dict, Any

class MetricsCollector:
    def __init__(self, num_agents: int, num_time_steps: int):
        self.num_agents = num_agents
        self.num_time_steps = num_time_steps
        
        # Initialize storage arrays
        self.payoffs = np.zeros((num_agents, num_time_steps + 1))
        self.items = np.zeros((num_agents, num_time_steps + 1), dtype=int)
        self.unique_items = np.zeros(num_time_steps + 1, dtype=int)
        self.learned_solutions = np.zeros(num_time_steps + 1, dtype=int)
        self.innovated_solutions = np.zeros(num_time_steps + 1, dtype=int)
        
        # Network metrics
        self.network_metrics = {}
        
    def collect_step(self, t: int, payoffs: np.ndarray, items: np.ndarray,
                    learned: int, innovated: int):
        """Collect metrics for a single time step"""
        self.payoffs[:, t] = payoffs
        self.items[:, t] = items
        self.unique_items[t] = len(np.unique(items))
        self.learned_solutions[t] = learned
        self.innovated_solutions[t] = innovated
        
    def collect_network_metrics(self, metrics: Dict[str, Any]):
        """Collect network metrics"""
        self.network_metrics.update(metrics)
        
    def get_results(self) -> Dict[str, Any]:
        """Convert collected metrics to results dictionary"""
        # Calculate time series metrics
        time_series = {
            'payoff_mean': np.mean(self.payoffs, axis=0),
            'payoff_std': np.std(self.payoffs, axis=0),
            'payoff_median': np.median(self.payoffs, axis=0),
            'unique_items': self.unique_items,
            'learned_solutions': self.learned_solutions,
            'innovated_solutions': self.innovated_solutions
        }
        
        # Calculate summary metrics
        summary = {
            'final_payoff_mean': np.mean(self.payoffs[:, -1]),
            'final_payoff_std': np.std(self.payoffs[:, -1]),
            'total_learned': np.sum(self.learned_solutions),
            'total_innovated': np.sum(self.innovated_solutions),
            'avg_unique_items': np.mean(self.unique_items)
        }
        
        return {
            'time_series': time_series,
            'network': self.network_metrics,
            'summary': summary
        } 