"""Shared wrapper-stack helpers matching mpe2's make_env / parallel_wrapper_fn.

mpe2's flow is:
    raw_env (AEC) -> make_env -> Clip/Assert + OrderEnforcing (AEC)
                              -> parallel_wrapper_fn -> aec_to_parallel (parallel)

Our raw_env classes are native ParallelEnv subclasses, so we apply checks
inline at step() time (parallel path) and via the wrapper stack at env() time
(AEC path).
"""

import numpy as np
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_to_aec_wrapper

# Cached for hot-path isinstance() check.
_INT_TYPES = (int, np.integer)


def check_and_maybe_clip_actions(actions, action_space_fn, continuous_actions):
    """Validate (discrete) or clip (continuous) per-agent actions in-place.

    Matches AssertOutOfBoundsWrapper / ClipOutOfBoundsWrapper semantics from
    pettingzoo, but applied at the parallel-env level. Returns a dict of
    (possibly clipped) actions; raises AssertionError for invalid discrete
    actions.

    Hot-path optimized: skips space.contains() and uses a direct bounds
    check (`0 <= val < space.n`) for Discrete spaces.
    """
    if continuous_actions:
        # Clip to [low, high] for each agent's Box space.
        result = {}
        for agent, action in actions.items():
            space = action_space_fn(agent)
            result[agent] = np.clip(action, space.low, space.high)
        return result
    # Discrete fast path: avoid space.contains() (which constructs a numpy int
    # and dispatches through gymnasium's space machinery) — just check bounds.
    for agent, action in actions.items():
        # Unwrap a 1-element sequence to its scalar.
        if isinstance(action, _INT_TYPES):
            val = action
        else:
            # numpy array, list, etc. — pull element 0 if non-empty.
            val = int(action[0]) if len(action) > 0 else action
        n = action_space_fn(agent).n
        if not (0 <= val < n):
            raise AssertionError(
                f"action {action} for agent {agent} is not in action space "
                f"Discrete({n})"
            )
    return actions


def make_aec_env(parallel_env_instance):
    """Wrap a ParallelEnv with the standard mpe2 AEC wrapper stack.

    Applies parallel_to_aec_wrapper, then ClipOutOfBoundsWrapper (for continuous)
    or AssertOutOfBoundsWrapper (for discrete), then OrderEnforcingWrapper.
    """
    aec = parallel_to_aec_wrapper(parallel_env_instance)
    if parallel_env_instance.continuous_actions:
        aec = wrappers.ClipOutOfBoundsWrapper(aec)
    else:
        aec = wrappers.AssertOutOfBoundsWrapper(aec)
    aec = wrappers.OrderEnforcingWrapper(aec)
    return aec
