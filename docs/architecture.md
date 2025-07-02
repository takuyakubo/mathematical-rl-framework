# Architecture Design Document

## Overview

This document details the architectural decisions and design patterns used in the Mathematical Reinforcement Learning Framework.

## Core Components

### PlayGround

The central orchestrator that manages the step progression and coordination between agents and environment.

```python
class PlayGround(BaseModel):
    public_state: PublicState
    agents: List[Agent]
    environment: Environment
    current_step: int = 0
    
    def step(self) -> StepResult:
        """Execute one step of the simulation"""
        # 1. Get observation snapshots for each agent
        # 2. Collect actions from all agents
        # 3. Send actions to environment
        # 4. Update public state
        # 5. Notify agents of new observations
```

**Responsibilities:**
- Step progression coordination
- Public information management
- Agent-Environment communication facilitation

### Agent

Represents a learning agent with private observations and decision-making capabilities.

```python
class Agent(BaseModel):
    agent_id: str
    private_state: PrivateAgentState
    policy_strategy: PolicyStrategy
    current_policy: Optional[Policy] = None
    observation: AgentObservation
    
    def get_action(self, 
                   public_state: PublicState, 
                   available_actions: List[Action]) -> Action:
        """Get action based on current policy"""
```

**Responsibilities:**
- Maintaining private observations (noisy environment signals + private info)
- Policy management and action selection
- Learning state updates

### Environment

Manages the underlying game/simulation state and dynamics.

```python
class Environment(ABC):
    private_state: PrivateEnvironmentState
    dynamics: Dynamics  # Dependency injection point
    
    @abstractmethod
    def step(self, actions: Dict[str, Action]) -> EnvironmentResponse:
        """Process actions and return new state"""
    
    @abstractmethod
    def get_available_actions(self, agent_id: str) -> List[Action]:
        """Get valid actions for an agent"""
    
    @abstractmethod
    def get_observation_for_agent(self, agent_id: str) -> dict:
        """Get observable information for specific agent"""
```

**Responsibilities:**
- Game state management
- Rule enforcement
- Observation generation (with appropriate noise/filtering)

## Policy System Design

### Two-Level Policy Architecture

The framework separates policy creation from policy execution:

1. **PolicyStrategy**: Determines how to create policies based on current situation
2. **Policy**: Executes immediate action decisions

This separation enables:
- Dynamic policy switching during execution
- Context-dependent policy creation
- Clean separation between learning algorithms and decision logic

### Example Policy Strategies

```python
class ValueBasedPolicyStrategy(PolicyStrategy):
    """Creates policies based on value function estimation"""
    value_function: ValueFunction
    exploration_rate: float
    
    def get_policy(self, agent_id: str, observation_snapshot: ObservationSnapshot, context=None) -> Policy:
        return EpsilonGreedyPolicy(
            value_function=self.value_function,
            epsilon=self.exploration_rate
        )

class MonteCarloTreeSearchStrategy(PolicyStrategy):
    """Creates MCTS policies for planning"""
    belief_model: BeliefModel
    simulation_env_factory: Callable[[], SimulationEnvironment]
    
    def get_policy(self, agent_id: str, observation_snapshot: ObservationSnapshot, context=None) -> Policy:
        # Estimate possible states from observations
        belief_states = self.belief_model.estimate_states(observation_snapshot)
        
        return MCTSPolicy(
            belief_states=belief_states,
            simulation_factory=self.simulation_env_factory
        )
```

## Information Flow and Observability

### Information Hierarchy

```
Environment Private State (hidden)
    ↓ (with noise/filtering)
Agent Observations (private to each agent)
    ↓ (aggregated/filtered)
Public State (visible to all)
```

### Observation Snapshot System

To maintain information hiding principles, the framework uses observation snapshots:

```python
class ObservationSnapshot(BaseModel):
    """Immutable snapshot of observable information"""
    public_state: PublicState
    agent_observation: AgentObservation
    available_actions: List[Action]
    timestamp: int
    
    class Config:
        frozen = True  # Immutable to prevent accidental modification
```

This ensures that:
- Policy strategies cannot access hidden information
- Monte Carlo simulations must work with uncertain state information
- State estimation becomes part of the learning problem

## Dependency Injection Points

### Environment Dynamics

Different environments require different computational approaches:

```python
class Dynamics(ABC):
    """Abstract interface for environment dynamics"""
    @abstractmethod
    def transition(self, state: State, actions: Dict[str, Action]) -> State:
        pass

class MatrixDynamics(Dynamics):
    """For small MDPs with explicit transition matrices"""
    transition_matrix: np.ndarray
    reward_matrix: np.ndarray

class SimulationDynamics(Dynamics):
    """For complex rule-based environments"""
    rule_engine: RuleEngine
```

### Value Functions and Models

Learning components can be injected as needed:

```python
class ValueFunction(ABC):
    @abstractmethod
    def evaluate(self, state: State, action: Action) -> float:
        pass

class BeliefModel(ABC):
    @abstractmethod
    def estimate_states(self, observation: ObservationSnapshot) -> List[PossibleState]:
        pass
```

## Design Patterns Used

### Strategy Pattern
- PolicyStrategy for different policy creation approaches
- Dynamics for different environment computation methods

### Factory Pattern
- PolicyStrategy creates Policy instances
- Environment factories for simulation environments

### Observer Pattern (Future)
- Logging and monitoring systems (separate from core logic)

### Dependency Injection
- Computational paradigm switching
- Model and algorithm substitution

## Computational Paradigm Support

### Matrix-Based Computation
For environments with small, discrete state spaces:
- Explicit transition and reward matrices
- Vectorized operations
- Exact value function computation

### Simulation-Based Computation
For complex environments:
- Rule-based state transitions
- Monte Carlo sampling
- Approximate value estimation

### Hybrid Approaches
Combining analytical and simulation methods:
- Analytical computation where possible
- Simulation for complex interactions
- Caching and memoization for efficiency

## Extension Points

### Adding New Environments
1. Implement `Environment` interface
2. Create appropriate `Dynamics` implementation
3. Define state and action spaces
4. Implement observation generation logic

### Adding New Algorithms
1. Create `PolicyStrategy` implementation
2. Implement corresponding `Policy` class
3. Add any required models (value functions, belief models)
4. Integrate with dependency injection system

### Adding Computational Methods
1. Implement `Dynamics` interface for new computation paradigm
2. Create factory methods for environment instantiation
3. Ensure compatibility with existing policy strategies

## Performance Considerations

### Memory Management
- Immutable snapshots prevent accidental state corruption
- Efficient copying mechanisms for simulation environments
- Garbage collection considerations for long-running simulations

### Computational Efficiency
- Lazy evaluation where appropriate
- Caching of expensive computations
- Parallel simulation support (future enhancement)

### Scalability
- Batch processing capabilities
- Distributed computation support (future enhancement)
- Memory-efficient history management

## Testing Strategy

### Unit Testing
- Individual component testing with mocked dependencies
- Policy strategy testing with controlled observations
- Environment dynamics validation

### Integration Testing
- Full simulation runs with known outcomes
- Multi-agent interaction testing
- Information hiding validation

### Performance Testing
- Computational paradigm comparison
- Memory usage profiling
- Simulation speed benchmarking