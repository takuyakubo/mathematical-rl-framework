# Mathematical Reinforcement Learning Framework

A mathematically tractable reinforcement learning framework designed for multi-agent environments with partial observability.

## Design Philosophy

This framework prioritizes mathematical rigor and clean abstractions while supporting both matrix-based computations (for simple MDPs) and simulation-based approaches (for complex environments like board games). The architecture separates concerns clearly and uses dependency injection to support different computational paradigms.

## Core Architecture

### Three-Layer Design

The framework is built around three main components:

- **PlayGround**: Orchestrates the step progression and manages public information
- **Agent**: Represents learning agents with private observations and policy strategies
- **Environment**: Manages game/simulation state and dynamics

### Information Visibility

- **PlayGround**: Holds only publicly observable information
- **Agent**: Maintains private observations (including noisy environment signals and private information like hand cards)
- **Environment**: Keeps private game state necessary for progression

## Key Components

### Policy and Strategy Pattern

```python
class Policy(BaseModel):
    """Handles immediate action decisions"""
    @abstractmethod
    def decide(self, 
              agent_observation: AgentObservation,
              public_state: PublicState,
              available_actions: List[Action]) -> Action:
        pass

class PolicyStrategy(ABC):
    """Factory for creating/updating policies"""
    @abstractmethod
    def get_policy(self, 
                   agent_id: str, 
                   observation_snapshot: ObservationSnapshot,
                   context: Optional[dict] = None) -> Policy:
        pass
```

The separation allows for:
- Static policies (pre-trained models)
- Adaptive policies (learning during execution)
- Context-dependent policies (exploration vs exploitation phases)

### Partial Observability Support

The framework respects information constraints through observation snapshots:

```python
class ObservationSnapshot(BaseModel):
    """Contains only information available to the agent"""
    public_state: PublicState
    agent_observation: AgentObservation
    available_actions: List[Action]
    timestamp: int
```

This ensures that policy strategies (including Monte Carlo methods) cannot access hidden information, making state estimation part of the learning problem.

### Dependency Injection for Computational Paradigms

Different environments require different computational approaches:

- **Matrix-based**: For small state spaces with explicit transition matrices
- **Simulation-based**: For complex rule-based environments (chess, Go, etc.)
- **Hybrid approaches**: Combining analytical and simulation methods

The framework uses dependency injection to swap between these paradigms without changing the core logic.

## Use Cases

### Supported Scenarios

- Single and multi-agent reinforcement learning
- Perfect and imperfect information games
- Turn-based and simultaneous action environments
- Model-free and model-based approaches
- Online and offline learning

### Example Applications

- Board games (chess, poker)
- Multi-agent coordination problems
- Financial trading simulations
- Resource allocation problems

## Technical Stack

- **Python**: Core implementation language
- **Pydantic**: Type safety and validation
- **Dependency Injection**: Computational paradigm switching

## Design Principles

1. **Mathematical Tractability**: Clear correspondence between code and mathematical concepts
2. **Type Safety**: Extensive use of Pydantic for runtime validation
3. **Separation of Concerns**: Clean boundaries between components
4. **Information Hiding**: Proper handling of partial observability
5. **Extensibility**: Easy to add new algorithms and environments

## Next Steps

- Implement core interfaces and base classes
- Create example environments (simple MDP, board game)
- Develop standard policy strategies (random, value-based, MCTS)
- Add comprehensive testing framework
- Performance optimization for large-scale simulations

## Contributing

This framework is designed to be mathematically sound and practically useful. Contributions should maintain the clean separation of concerns and respect the information hiding principles.