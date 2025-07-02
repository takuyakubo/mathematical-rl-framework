# API Interfaces and Core Classes

This document defines the core interfaces and base classes for the Mathematical Reinforcement Learning Framework.

## Core Data Models

### State and Observation Types

```python
from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
import numpy as np

class PublicState(BaseModel):
    """Information visible to all participants"""
    data: Dict[str, Any]
    timestamp: int

class PrivateAgentState(BaseModel):
    """Agent-specific private information"""
    agent_id: str
    data: Dict[str, Any]

class PrivateEnvironmentState(BaseModel):
    """Environment's internal state"""
    data: Dict[str, Any]

class AgentObservation(BaseModel):
    """Agent's observation of the environment"""
    # Noisy/filtered environment signals
    env_observation: Dict[str, Any]
    # Private information (e.g., hand cards)
    private_info: Dict[str, Any]
    timestamp: int

class ObservationSnapshot(BaseModel):
    """Immutable snapshot of observable information for policy creation"""
    public_state: PublicState
    agent_observation: AgentObservation
    available_actions: List['Action']
    timestamp: int
    
    class Config:
        frozen = True
```

### Action System

```python
class ActionSpace(BaseModel):
    """Defines the space of possible actions"""
    action_type: str  # "discrete", "continuous", "mixed"
    parameters: Dict[str, Any]

class Action(BaseModel):
    """Represents a single action"""
    agent_id: str
    action_type: str
    value: Union[int, float, np.ndarray, Dict[str, Any]]
    timestamp: int

class EnvironmentResponse(BaseModel):
    """Response from environment after processing actions"""
    new_public_state: PublicState
    agent_observations: Dict[str, AgentObservation]
    rewards: Dict[str, float]
    done: bool
    info: Dict[str, Any]
```

## Core Interfaces

### Policy System

```python
class Policy(ABC):
    """Abstract base class for action decision policies"""
    
    @abstractmethod
    def decide(self, 
              agent_observation: AgentObservation,
              public_state: PublicState,
              available_actions: List[Action]) -> Action:
        """
        Make an action decision based on available information.
        
        Args:
            agent_observation: Agent's private observation
            public_state: Publicly visible state
            available_actions: List of valid actions
            
        Returns:
            Selected action
        """
        pass

class PolicyStrategy(ABC):
    """Abstract base class for policy creation strategies"""
    
    @abstractmethod
    def get_policy(self, 
                   agent_id: str, 
                   observation_snapshot: ObservationSnapshot,
                   context: Optional[Dict[str, Any]] = None) -> Policy:
        """
        Create a policy based on current observations and context.
        
        Args:
            agent_id: Identifier of the agent
            observation_snapshot: Current observable information
            context: Additional context information
            
        Returns:
            Policy instance for action decisions
        """
        pass
```

### Environment Interface

```python
class Dynamics(ABC):
    """Abstract interface for environment dynamics computation"""
    
    @abstractmethod
    def transition(self, 
                   current_state: PrivateEnvironmentState, 
                   actions: Dict[str, Action]) -> PrivateEnvironmentState:
        """Compute state transition given actions"""
        pass
    
    @abstractmethod
    def compute_rewards(self, 
                       state: PrivateEnvironmentState, 
                       actions: Dict[str, Action]) -> Dict[str, float]:
        """Compute rewards for each agent"""
        pass

class Environment(ABC):
    """Abstract base class for environments"""
    
    def __init__(self):
        self.private_state: PrivateEnvironmentState = None
        self.dynamics: Dynamics = None
    
    @abstractmethod
    def step(self, actions: Dict[str, Action]) -> EnvironmentResponse:
        """
        Process actions and return new state and observations.
        
        Args:
            actions: Dictionary mapping agent_id to their actions
            
        Returns:
            Environment response with new state and observations
        """
        pass
    
    @abstractmethod
    def get_available_actions(self, agent_id: str) -> List[Action]:
        """
        Get list of valid actions for a specific agent.
        
        Args:
            agent_id: Identifier of the agent
            
        Returns:
            List of valid actions
        """
        pass
    
    @abstractmethod
    def get_observation_for_agent(self, agent_id: str) -> AgentObservation:
        """
        Generate observation for a specific agent.
        
        Args:
            agent_id: Identifier of the agent
            
        Returns:
            Agent's observation of the environment
        """
        pass
    
    @abstractmethod
    def reset(self) -> EnvironmentResponse:
        """Reset environment to initial state"""
        pass
```

### Agent Interface

```python
class Agent(BaseModel):
    """Represents a learning agent"""
    
    agent_id: str
    private_state: PrivateAgentState
    policy_strategy: PolicyStrategy
    current_policy: Optional[Policy] = None
    observation: Optional[AgentObservation] = None
    
    def get_action(self, 
                   public_state: PublicState, 
                   available_actions: List[Action]) -> Action:
        """
        Get action based on current policy and observations.
        
        Args:
            public_state: Current public state
            available_actions: List of valid actions
            
        Returns:
            Selected action
        """
        if self.current_policy is None:
            # Create observation snapshot for policy creation
            snapshot = ObservationSnapshot(
                public_state=public_state,
                agent_observation=self.observation,
                available_actions=available_actions,
                timestamp=public_state.timestamp
            )
            self.current_policy = self.policy_strategy.get_policy(
                self.agent_id, 
                snapshot
            )
        
        return self.current_policy.decide(
            self.observation, 
            public_state, 
            available_actions
        )
    
    def update_observation(self, new_observation: AgentObservation):
        """Update agent's observation"""
        self.observation = new_observation
        # Optionally invalidate current policy to force recreation
        self.current_policy = None
```

### PlayGround Interface

```python
class StepResult(BaseModel):
    """Result of a single step execution"""
    step_number: int
    actions_taken: Dict[str, Action]
    environment_response: EnvironmentResponse
    success: bool
    info: Dict[str, Any]

class PlayGround(BaseModel):
    """Central orchestrator for the simulation"""
    
    public_state: PublicState
    agents: List[Agent]
    environment: Environment
    current_step: int = 0
    
    def step(self) -> StepResult:
        """
        Execute one step of the simulation.
        
        Returns:
            Step execution result
        """
        actions = {}
        
        # Collect actions from all agents
        for agent in self.agents:
            available_actions = self.environment.get_available_actions(agent.agent_id)
            action = agent.get_action(self.public_state, available_actions)
            actions[agent.agent_id] = action
        
        # Send actions to environment
        env_response = self.environment.step(actions)
        
        # Update public state
        self.public_state = env_response.new_public_state
        
        # Update agent observations
        for agent in self.agents:
            if agent.agent_id in env_response.agent_observations:
                agent.update_observation(
                    env_response.agent_observations[agent.agent_id]
                )
        
        self.current_step += 1
        
        return StepResult(
            step_number=self.current_step,
            actions_taken=actions,
            environment_response=env_response,
            success=True,
            info={}
        )
    
    def reset(self) -> EnvironmentResponse:
        """Reset the entire simulation"""
        self.current_step = 0
        env_response = self.environment.reset()
        self.public_state = env_response.new_public_state
        
        # Reset all agents
        for agent in self.agents:
            agent.current_policy = None
            if agent.agent_id in env_response.agent_observations:
                agent.update_observation(
                    env_response.agent_observations[agent.agent_id]
                )
        
        return env_response
```

## Utility Interfaces

### Belief and Value Models

```python
class PossibleState(BaseModel):
    """Represents a possible state in belief space"""
    state_data: Dict[str, Any]
    probability: float

class BeliefModel(ABC):
    """Abstract interface for state estimation from observations"""
    
    @abstractmethod
    def estimate_states(self, observation: ObservationSnapshot) -> List[PossibleState]:
        """
        Estimate possible states from observation.
        
        Args:
            observation: Current observation snapshot
            
        Returns:
            List of possible states with probabilities
        """
        pass

class ValueFunction(ABC):
    """Abstract interface for value functions"""
    
    @abstractmethod
    def evaluate(self, state: Dict[str, Any], action: Action) -> float:
        """
        Evaluate state-action pair.
        
        Args:
            state: State representation
            action: Action to evaluate
            
        Returns:
            Estimated value
        """
        pass
```

## Example Implementations

### Simple Random Policy

```python
import random

class RandomPolicy(Policy):
    """Simple random action selection policy"""
    
    def decide(self, 
              agent_observation: AgentObservation,
              public_state: PublicState,
              available_actions: List[Action]) -> Action:
        """Randomly select from available actions"""
        return random.choice(available_actions)

class RandomPolicyStrategy(PolicyStrategy):
    """Strategy that always returns random policies"""
    
    def get_policy(self, 
                   agent_id: str, 
                   observation_snapshot: ObservationSnapshot,
                   context: Optional[Dict[str, Any]] = None) -> Policy:
        return RandomPolicy()
```

### Value-Based Policy

```python
class EpsilonGreedyPolicy(Policy):
    """Epsilon-greedy policy based on value function"""
    
    def __init__(self, value_function: ValueFunction, epsilon: float = 0.1):
        self.value_function = value_function
        self.epsilon = epsilon
    
    def decide(self, 
              agent_observation: AgentObservation,
              public_state: PublicState,
              available_actions: List[Action]) -> Action:
        """Select action using epsilon-greedy strategy"""
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Greedy action selection
        state_repr = {**public_state.data, **agent_observation.env_observation}
        best_action = max(
            available_actions,
            key=lambda a: self.value_function.evaluate(state_repr, a)
        )
        return best_action

class ValueBasedPolicyStrategy(PolicyStrategy):
    """Strategy for value-based policies"""
    
    def __init__(self, value_function: ValueFunction, epsilon: float = 0.1):
        self.value_function = value_function
        self.epsilon = epsilon
    
    def get_policy(self, 
                   agent_id: str, 
                   observation_snapshot: ObservationSnapshot,
                   context: Optional[Dict[str, Any]] = None) -> Policy:
        return EpsilonGreedyPolicy(self.value_function, self.epsilon)
```

## Configuration and Factory Patterns

```python
class FrameworkConfig(BaseModel):
    """Configuration for the entire framework"""
    
    # Environment configuration
    environment_type: str
    environment_params: Dict[str, Any]
    
    # Agent configurations
    agent_configs: List[Dict[str, Any]]
    
    # Simulation parameters
    max_steps: int = 1000
    random_seed: Optional[int] = None

class FrameworkFactory:
    """Factory for creating framework components"""
    
    @staticmethod
    def create_playground(config: FrameworkConfig) -> PlayGround:
        """Create a complete playground from configuration"""
        # Implementation would instantiate environment, agents, etc.
        pass
    
    @staticmethod
    def create_environment(env_type: str, params: Dict[str, Any]) -> Environment:
        """Create environment from configuration"""
        pass
    
    @staticmethod
    def create_agent(agent_config: Dict[str, Any]) -> Agent:
        """Create agent from configuration"""
        pass
```

This API design provides a clean, type-safe foundation for implementing various reinforcement learning algorithms while maintaining the mathematical rigor and information hiding principles established in the architecture.