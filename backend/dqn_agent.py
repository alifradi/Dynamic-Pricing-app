"""
Deep Q-Network (DQN) Agent for Strategic Hotel Ranking
Implements reinforcement learning for adaptive policy selection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import json
import os
from typing import List, Dict, Tuple, Optional

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Deep Q-Network architecture for policy selection"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """Deep Q-Network agent for strategic hotel ranking decisions"""
    
    def __init__(self, 
                 state_size: int = 6,  # market_demand, days_to_go, competition_density, etc.
                 action_size: int = 4,  # number of policies
                 hidden_size: int = 64,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update: int = 100):
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Policy definitions
        self.policies = {
            0: {"name": "High-Trust Policy", "alpha": 0.2, "beta": 0.6, "gamma": 0.2},
            1: {"name": "Balanced Policy", "alpha": 0.4, "beta": 0.3, "gamma": 0.3},
            2: {"name": "High-Revenue Policy", "alpha": 0.6, "beta": 0.2, "gamma": 0.2},
            3: {"name": "Partner-Focused Policy", "alpha": 0.3, "beta": 0.2, "gamma": 0.5}
        }
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Training counters
        self.step_count = 0
        self.episode_count = 0
        
        # Initialize target network
        self.update_target_network()
    
    def get_state_vector(self, market_data: Dict) -> np.ndarray:
        """Convert market data to state vector"""
        # Normalize market demand (0-1)
        market_demand = min(market_data.get('market_demand', 0) / 100, 1.0)
        
        # Normalize days to go (0-1, where 0 = immediate, 1 = far future)
        days_to_go = min(market_data.get('days_to_go', 30) / 365, 1.0)
        
        # Competition density (0-1)
        competition_density = min(market_data.get('competition_density', 0) / 20, 1.0)
        
        # Price volatility (0-1)
        price_volatility = min(market_data.get('price_volatility', 0) / 0.5, 1.0)
        
        # User satisfaction trend (-1 to 1, normalized to 0-1)
        satisfaction_trend = (market_data.get('satisfaction_trend', 0) + 1) / 2
        
        # Budget utilization (0-1)
        budget_utilization = min(market_data.get('budget_utilization', 0) / 100, 1.0)
        
        return np.array([
            market_demand,
            days_to_go,
            competition_density,
            price_volatility,
            satisfaction_trend,
            budget_utilization
        ], dtype=np.float32)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def get_policy_weights(self, action: int) -> Dict[str, float]:
        """Get policy weights for the selected action"""
        return self.policies[action]
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def replay(self) -> Optional[float]:
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> Optional[float]:
        """Take a step in the environment and train if needed"""
        self.remember(state, action, reward, next_state, done)
        
        loss = None
        if len(self.memory) > self.batch_size:
            loss = self.replay()
        
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target_network()
        
        return loss
    
    def get_policy_heatmap(self, market_scenarios: List[Dict]) -> Dict:
        """Generate policy heatmap for visualization"""
        heatmap_data = []
        
        for scenario in market_scenarios:
            state = self.get_state_vector(scenario)
            q_values = self.q_network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            q_values = q_values.detach().cpu().numpy()[0]
            
            heatmap_data.append({
                "market_demand": scenario.get('market_demand', 0),
                "days_to_go": scenario.get('days_to_go', 30),
                "competition_density": scenario.get('competition_density', 0),
                "q_values": q_values.tolist(),
                "best_policy": self.policies[q_values.argmax()]["name"]
            })
        
        return {
            "heatmap_data": heatmap_data,
            "policies": self.policies,
            "epsilon": self.epsilon
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.episode_count = checkpoint['episode_count']
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")

# Global DQN agent instance
dqn_agent = None

def get_dqn_agent() -> DQNAgent:
    """Get or create the global DQN agent instance"""
    global dqn_agent
    if dqn_agent is None:
        dqn_agent = DQNAgent()
        # Try to load existing model
        model_path = "/data/dqn_model.pth"  # Use Docker volume path
        dqn_agent.load_model(model_path)
    return dqn_agent

def calculate_reward(optimization_results: Dict) -> float:
    """Calculate reward based on optimization results"""
    # Multi-objective reward function
    trivago_income = optimization_results.get('trivago_income', 0)
    user_satisfaction = optimization_results.get('user_satisfaction', 0)
    partner_conversion_value = optimization_results.get('partner_conversion_value', 0)
    
    # Normalize and combine objectives
    reward = (
        0.4 * trivago_income / 1000 +  # Normalize by expected max
        0.3 * user_satisfaction / 10 +  # Normalize by max satisfaction
        0.3 * partner_conversion_value / 10  # Normalize by max conversion
    )
    
    return reward 