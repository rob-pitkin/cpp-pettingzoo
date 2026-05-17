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
| **SimpleLine** | Resets | 358,930 | 19,817 | **18.11x** |
| | Steps | 130,706 | 6,586 | **19.85x** |
| | Episodes | 5,286 | 263 | **20.09x** |
| **CollectTreasure** | Resets | 40,690 | 1,992 | **20.42x** |
| | Steps | 29,200 | 735 | **39.72x** |
| | Episodes | 1,172 | 29 | **39.84x** |
| **SimpleWorldComm** | Resets | 115,432 | 2,861 | **40.34x** |
| | Steps | 59,585 | 1,016 | **58.64x** |
| | Episodes | 2,382 | 41 | **58.13x** |

## Key Findings

### Overall Performance
- **Average speedup: 20.51x faster** than pure Python MPE2
- **Range: 6.23x - 58.64x** depending on environment and operation

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

**SimpleLine (4 cooperative agents, 2 line-endpoint landmarks):**
- Strong consistent speedup: **18.11x / 19.85x / 20.09x** across all operations
- Target positions are fixed at reset (the line doesn't move), so `compute_line` only solves Munkres each step — no per-step target geometry recomputation unlike SimpleFormation's rotating circle
- mpe2's reset is heavier than Formation's (placing lm1 via angular search loop in Python) which inflates the reset speedup to 18x despite 2 agents vs Formation's 1 landmark reset

**CollectTreasure (6 collectors + 2 deposits, 6 treasure landmarks):**
- **Highest step/episode speedup in the suite: 39.72x / 39.84x** — the gap comes from mpe2's Python loop over all (collector, treasure) and (collector, deposit) pairs for pickup/delivery/reward, plus distance-sorted observation construction every step; all of this is O(agents × treasures) Python iteration vs tight C++ loops
- Reset speedup (20.42x) is also strong: mpe2 allocates numpy arrays per-agent per-reset; C++ reuses pre-allocated vectors
- The `post_step` hook (pickup → respawn → delivery, all in C++ before reward computation) is the key architectural win — mpe2 had to override `_execute_world_step` entirely for this; our virtual no-op in BaseEnv adds zero overhead to all other environments

**SimpleWorldComm (1 leader + 3 adversaries + 2 good agents, 1 obstacle + 2 food + 2 forests):**
- **Highest speedup in the entire suite: 40.34x / 58.64x / 58.13x** — by a wide margin, this is the env where Python's overhead hurts most
- mpe2's `observation()` is the bottleneck: for every (self, other) agent pair it computes forest-membership flags by iterating all forests with numpy `is_collision` calls, then runs a `for/else` Python control-flow pattern to decide whether to include real positions or zero-pad. With 6 agents × 5 other-agents × 2 forests per observation, that's hundreds of numpy ops per step in pure Python; C++ replaces all of it with tight nested loops over plain `float`s
- 4-channel communication (`dim_c=4`) is also broadcast to every adversary every step; mpe2 builds a per-other-agent `comm` list then unconditionally overwrites it with `[world.agents[0].state.c]` — the C++ version skips the wasted loop entirely
- Reset speedup (40.34x) is also the highest in the suite — 9 entities (6 agents + 3 landmark categories), each currently requiring numpy array allocations in mpe2 vs pre-allocated `std::array<float, 2>` slots in C++
- The asymmetric action space (leader has Discrete(20), others have Discrete(5)) added zero new C++ code — `BaseEnv::step` already decomposed `action % 5` (movement) and `action / 5` (comm one-hot) for any non-silent agent, originally built for SimpleReference

### Performance Insights

1. **Release build matters significantly**: Prior Debug-mode numbers showed 2-4x; Release shows 9-40x
2. **Step/episode speedups exceed reset speedups**: Python wrapper overhead is proportionally larger for resets (one-time Python object construction) vs steps (pure physics loop)
3. **Algorithmic complexity matters most**: CollectTreasure's 40x speedup vs SimpleSpread's 21x — both have 8 agents, but CollectTreasure's O(agents × treasures) pickup/reward loops in Python are far slower than SimpleSpread's simpler reward structure
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

# SimpleLine environment
uv run python cpp_pettingzoo/benchmark_simple_line.py

# CollectTreasure environment
uv run python cpp_pettingzoo/benchmark_collect_treasure.py

# SimpleWorldComm environment
uv run python cpp_pettingzoo/benchmark_simple_world_comm.py
```

## Benchmark Details

Each benchmark measures three operations:

1. **Resets**: Creating new episodes (1M resets)
2. **Steps**: Environment dynamics with random actions (1M steps, auto-reset on done)
3. **Episodes**: Complete episodes with 25 steps each (100K episodes)

All benchmarks use discrete action spaces. Communication in SimpleReference uses Discrete(50) (10 communication words × 5 movement actions). SimpleSpeakerListener uses asymmetric discrete action spaces: speaker Discrete(3), listener Discrete(5). SimpleAdversary uses Discrete(5) for all agents (movement only, no communication despite dim_c=2). SimpleTag uses Discrete(5) for all agents (3 adversaries + 1 good agent) with default full observability (no partial observability neighbors set). SimplePush uses Discrete(5) for both agents; the good agent's observation encodes goal identity via landmark colors. SimpleFormation uses Discrete(5) for all 4 agents with a single central landmark; optimal agent-to-slot matching uses the Munkres algorithm (munkres-cpp) with results cached per step. SimpleLine uses Discrete(5) for all 4 agents with 2 line-endpoint landmarks; target positions are fixed at reset and reused across all steps of the episode. CollectTreasure uses Discrete(5) for all 8 agents (6 collectors + 2 deposits) with 6 treasure landmarks; pickup/delivery/respawn logic runs in a post_step hook between physics and reward computation. SimpleWorldComm uses asymmetric discrete action spaces: the leader (`leadadversary_0`) has Discrete(20) = 5 movement × 4 communication words, while the 3 normal adversaries and 2 good agents each have Discrete(5) (movement only). Observations are also asymmetric (34 for adversaries including the 4-channel leader comm, 28 for good agents). Three landmark subtypes (1 obstacle, 2 food, 2 forests) share the `world.landmarks` vector and are distinguished by a `subtype` tag on `core::Landmark`. Forest-based partial observability: agents only see others if they share a forest, neither is in any forest, or self is the leader.
