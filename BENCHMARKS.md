# cpp-pettingzoo Performance Benchmarks

Benchmarks comparing C++ implementation (cpp-pettingzoo) vs pure Python implementation (mpe2).

**Hardware:** M1 MacBook Pro  
**Build:** Release (`-O2`, CMake default Release flags)  
**Test Configuration:** 1M resets, 1M steps, 100K episodes (25 cycles each)

## Results Summary

| Environment | Metric | C++ (ops/sec) | Python (ops/sec) | Speedup |
|-------------|--------|---------------|------------------|---------|
| **Simple** | Resets | 982,152 | 100,763 | **9.75x** |
| | Steps | 538,869 | 35,623 | **15.13x** |
| | Episodes | 22,488 | 1,438 | **15.64x** |
| **SimpleSpread** | Resets | 315,470 | 35,936 | **8.78x** |
| | Steps | 161,036 | 7,709 | **20.89x** |
| | Episodes | 6,502 | 317 | **20.53x** |
| **SimpleReference** | Resets | 433,989 | 36,194 | **11.99x** |
| | Steps | 248,318 | 16,342 | **15.20x** |
| | Episodes | 10,205 | 641 | **15.92x** |
| **SimpleSpeakerListener** | Resets | 591,300 | 42,495 | **13.91x** |
| | Steps | 319,597 | 19,966 | **16.01x** |
| | Episodes | 13,040 | 799 | **16.32x** |
| **SimpleAdversary** | Resets | 477,898 | 35,164 | **13.59x** |
| | Steps | 172,841 | 11,580 | **14.93x** |
| | Episodes | 7,094 | 464 | **15.28x** |
| **SimpleTag** | Resets | 197,755 | 31,725 | **6.23x** |
| | Steps | 92,890 | 5,182 | **17.93x** |
| | Episodes | 3,853 | 195 | **19.73x** |
| **SimplePush** | Resets | 537,179 | 43,469 | **12.36x** |
| | Steps | 220,386 | 15,960 | **13.81x** |
| | Episodes | 9,127 | 645 | **14.14x** |
| **SimpleFormation** | Resets | 399,064 | 39,599 | **10.08x** |
| | Steps | 136,006 | 6,297 | **21.60x** |
| | Episodes | 5,515 | 253 | **21.83x** |

## Key Findings

### Overall Performance
- **Average speedup: 15.07x faster** than pure Python MPE2
- **Range: 6.23x - 21.83x** depending on environment and operation

### Environment-Specific Analysis

**Simple (1 agent, 1 landmark):**
- Consistent ~15x speedup for simulation operations
- Single-agent physics with no collision detection
- Reset is ~10x — Python overhead amortized across more work in steps/episodes

**SimpleSpread (3 agents, 3 landmarks):**
- **Best step/episode speedup**: 20.89x and 20.53x
- Physics-heavy with collision detection; C++ benefits compound with more agents
- Global reward computation (min-distance per landmark) is particularly fast in C++

**SimpleReference (2 agents, 3 landmarks, 10-word communication):**
- Solid ~15x speedup across all operations
- Composite action space (Discrete(50): 10 comm × 5 movement) decomposition is efficient in C++

**SimpleSpeakerListener (2 agents, 3 landmarks, 3-word communication):**
- **Best reset performance**: 591K resets/sec (13.91x)
- Asymmetric agents and action spaces handled efficiently
- Strong 16x episode speedup from efficient asymmetric action decomposition

**SimpleAdversary (1 adversary + N good agents, N landmarks):**
- Consistent ~14-15x speedup
- Reward caching eliminates redundant sqrt computations: good agent rewards (min_dist + adv_dist) computed once per step rather than once per good agent
- Adversarial reward structure (asymmetric goals) has no measurable overhead vs cooperative envs

**SimpleTag (3 adversaries + 1 good agent, 2 obstacles):**
- **Lowest reset speedup: 6.23x** — 4 agents means heavier per-reset Python overhead (observation dict construction, numpy conversions) relative to the C++ reset work
- **Strong step/episode speedup: 17.93x / 19.73x** — collision detection across all (adversary, good) pairs and the `bound()` penalty are cheap in C++
- The reset/step gap is the widest of any environment: Python wrapper overhead dominates resets but is amortized away over a 25-step episode

**SimplePush (1 adversary + 1 good agent, 2 landmarks):**
- Tight 12–14x speedup across all operations — the most consistent ratio in the suite
- Smallest world (2 agents, 2 landmarks, no collision detection on landmarks) keeps per-step C++ work low, so Python wrapper overhead stays proportionally significant
- The asymmetric observation (good agent encodes goal color; adversary does not) adds negligible C++ cost vs the Python equivalent

**SimpleFormation (4 cooperative agents, 1 central landmark):**
- **Best step/episode speedup in the suite: 21.60x / 21.83x** — Hungarian matching via munkres-cpp runs in microseconds; Python equivalent calls scipy's linear_sum_assignment each step
- Reset speedup (10.08x) is moderate: the C++ reset is lightweight (random positions only) so Python wrapper overhead is proportionally larger
- Pure global reward (local_ratio=0.0) means all 4 agents share the same scalar per step; the mutable cache (`cache_valid_` flag) ensures the Munkres solve runs exactly once per step even though global_reward() is called N times

### Performance Insights

1. **Release build matters significantly**: Prior Debug-mode numbers showed 2-4x; Release shows 9-21x
2. **Step/episode speedups exceed reset speedups**: Python wrapper overhead is proportionally larger for resets (one-time Python object construction) vs steps (pure physics loop)
3. **More agents = larger step speedup**: SimpleSpread (3 agents) achieves the highest step/episode speedup at 20.89x; its reset speedup (8.78x) is the lowest because reset cost is dominated by Python wrapper overhead rather than physics computation
4. **Reward caching pays off**: SimpleAdversary's per-step reward cache (mutable members, invalidated once per step) eliminated O(N) redundant sqrt calls with no correctness tradeoff

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

# SimpleAdversary environment
uv run python cpp_pettingzoo/benchmark_simple_adversary.py

# SimpleTag environment
uv run python cpp_pettingzoo/benchmark_simple_tag.py

# SimplePush environment
uv run python cpp_pettingzoo/benchmark_simple_push.py

# SimpleFormation environment
uv run python cpp_pettingzoo/benchmark_simple_formation.py
```

## Benchmark Details

Each benchmark measures three operations:

1. **Resets**: Creating new episodes (1M resets)
2. **Steps**: Environment dynamics with random actions (1M steps, auto-reset on done)
3. **Episodes**: Complete episodes with 25 steps each (100K episodes)

All benchmarks use discrete action spaces. Communication in SimpleReference uses Discrete(50) (10 communication words × 5 movement actions). SimpleSpeakerListener uses asymmetric discrete action spaces: speaker Discrete(3), listener Discrete(5). SimpleAdversary uses Discrete(5) for all agents (movement only, no communication despite dim_c=2). SimpleTag uses Discrete(5) for all agents (3 adversaries + 1 good agent) with default full observability (no partial observability neighbors set). SimplePush uses Discrete(5) for both agents; the good agent's observation encodes goal identity via landmark colors. SimpleFormation uses Discrete(5) for all 4 agents with a single central landmark; optimal agent-to-slot matching uses the Munkres algorithm (munkres-cpp) with results cached per step.
