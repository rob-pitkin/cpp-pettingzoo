# cpp-pettingzoo

Fast C++ implementations of PettingZoo MPE environments with Python bindings.

## Goal

Provide high-performance C++ implementations of Multi-Agent Particle Environments (MPE) from PettingZoo, while maintaining compatibility with the PettingZoo API through Python bindings.

## Environments

- [x] Simple - Single agent reaching a landmark (starting point)
- [ ] Simple Spread - Multiple agents coordinating to cover landmarks
- [ ] Simple Reference - Communication-based coordination

## Structure

```
cpp-pettingzoo/
├── cpp_pettingzoo/          # Python package
│   ├── simple/              # Simple environment
│   │   ├── simple.py        # Python PettingZoo wrapper
│   │   └── cpp/             # C++ implementation
│   │       ├── simple.cpp
│   │       └── simple.h
│   └── bindings/            # pybind11 bindings
└── setup.py                 # Build configuration
```

## Dependencies

- Python 3.10+
- pybind11
- PettingZoo
- CMake (for building)
- C++17 compiler

## Installation

```bash
pip install -e .
```

## Usage

```python
from cpp_pettingzoo.simple import SimpleEnv

env = SimpleEnv()
observations = env.reset()
# ... use like any PettingZoo environment
```

## Performance

TBD - Benchmark results comparing C++ vs Python implementation
