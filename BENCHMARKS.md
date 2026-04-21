# cpp-pettingzoo Performance Benchmarks

Benchmarks comparing C++ implementation (cpp-pettingzoo) vs pure Python implementation (mpe2).

**Hardware:** M1 MacBook Pro
**Test Configuration:** 1M resets, 1M steps, 100K episodes (25 cycles each)

## Results Summary

| Environment | Metric | C++ (ops/sec) | Python (ops/sec) | Speedup |
|-------------|--------|---------------|------------------|---------|
| **Simple** | Resets | 204,778 | 102,490 | **2.00x** |
| | Steps | 84,133 | 36,350 | **2.31x** |
| | Episodes | 3,378 | 1,462 | **2.31x** |
| **SimpleSpread** | Resets | 72,245 | 36,925 | **1.96x** |
| | Steps | 30,148 | 8,040 | **3.75x** |
| | Episodes | 1,206 | 318 | **3.79x** |
| **SimpleReference** | Resets | 98,140 | 36,027 | **2.72x** |
| | Steps | 44,249 | 16,160 | **2.74x** |
| | Episodes | 1,775 | 656 | **2.71x** |
| **SimpleSpeakerListener** | Resets | 126,333 | 42,139 | **3.00x** |
| | Steps | 50,100 | 19,542 | **2.56x** |
| | Episodes | 2,004 | 792 | **2.53x** |

## Key Findings

### Overall Performance
- **Average speedup: 2.70x faster** than pure Python MPE2
- **Range: 1.96x - 3.79x** depending on environment and operation

### Environment-Specific Analysis

**Simple (1 agent, 1 landmark):**
- Consistent ~2.3x speedup for simulation operations
- Simpler physics benefits less from C++ optimization
- Reset overhead similar between implementations

**SimpleSpread (3 agents, 3 landmarks):**
- **Best performance**: 3.79x faster for full episodes
- Physics-heavy environment with collision detection
- C++ shines with more agents and collision calculations
- Episode speedup significantly higher than Simple

**SimpleReference (2 agents, 3 landmarks, communication):**
- Solid 2.7x speedup across all operations
- Communication overhead handled efficiently in C++
- Composite action space (Discrete(50)) decomposition benefits from C++

**SimpleSpeakerListener (2 agents, 3 landmarks, asymmetric communication):**
- **Best reset performance**: 3.00x faster than Python
- Asymmetric agents (speaker: non-movable/communicative, listener: silent/movable)
- Asymmetric action spaces handled efficiently (speaker: Discrete(3), listener: Discrete(5))
- Sequential action decomposition adds negligible overhead

### Performance Insights

1. **Complexity matters**: More complex environments (SimpleSpread with 3 agents) see larger speedups
2. **Physics computation**: Collision detection and integration benefit most from C++
3. **Consistent gains**: All operations show 2-4x improvement across the board
4. **Multi-agent scaling**: C++ advantage grows with number of agents

## Running Benchmarks

```bash
# Simple environment
uv run python cpp_pettingzoo/benchmark_simple.py

# SimpleSpread environment
uv run python cpp_pettingzoo/benchmark_simple_spread.py

# SimpleReference environment
uv run python cpp_pettingzoo/benchmark_simple_reference.py

# SimpleSpeakerListener environment
uv run python cpp_pettingzoo/benchmark_simple_speaker_listener.py
```

## Benchmark Details

Each benchmark measures three operations:

1. **Resets**: Creating new episodes (1M resets)
2. **Steps**: Environment dynamics with random actions (1M steps)
3. **Episodes**: Complete episodes with 25 steps each (100K episodes)

All benchmarks use discrete action spaces. Communication in SimpleReference uses the default Discrete(50) action space (10 communication words × 5 movement actions). SimpleSpeakerListener uses asymmetric discrete action spaces: speaker has Discrete(3) for communication only, listener has Discrete(5) for movement only.
