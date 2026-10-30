"""A large Bloch cache must fit by streaming rather than relaxing its budget."""
import json

import numpy as np
import pytest

from mandacaru import _native


def uniform_crystal(dtype):
    shape = (40, 40, 40)
    n = np.prod(shape)
    arrays = {}

    def hold(value, dtype=float):
        key = f"a{len(arrays)}"
        arrays[key] = np.ascontiguousarray(value, dtype=dtype).ravel()
        return key

    points = [dict(k=[0.] * 3, fractional=[0.] * 3, weight=1 / 16,
                   psi=hold(np.full(n, 1 / np.sqrt(n)), dtype),
                   overlap=hold([1.], complex), fixed=hold([-.5], complex),
                   projections=hold([], complex)) for _ in range(16)]
    payload = dict(schema_version=1, representation="periodic-paw-snapshot",
                   energy_unit="Ha", length_unit="bohr",
                   grid=dict(shape=shape, origin=[0.] * 3, step=np.eye(3).ravel().tolist()),
                   m=1, p=0, channels=0, channel_labels=[], kpoints=points,
                   blocks=hold([], complex), core=hold(np.zeros(n)),
                   initial=hold(np.full(n, 1.1 / n)), initial_m=hold(np.zeros(n)),
                   initial_q=hold([], complex), kernel=hold(np.zeros(n)),
                   xc_wave=hold(np.zeros(3 * n)), phase=hold(np.ones(n), complex),
                   kerker=hold(np.ones(n)), ion=hold(np.zeros(n), complex),
                   hats=hold([], complex), u=hold([], complex), uion=hold([], complex),
                   linear=hold([], complex), eion=0., ion_constant=0., constant=.13,
                   reference=0., electrons=1., spins=1, functional="lda",
                   relativistic=False, permutations=[], qsym=hold([], complex),
                   precision="float32" if dtype == np.complex64 else "float64")
    return json.dumps(payload), arrays


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_streamed_bloch_cache_and_mixing_history_preserve_the_resident_solution(dtype):
    metadata, arrays = uniform_crystal(dtype)
    states = []
    for budget in (64 << 20, 256 << 20):
        context = _native.Context(threads=1, memory_limit_bytes=budget)
        problem = context.periodic_problem(metadata, arrays)
        result = problem.solve(width_ha=.06)
        state = result.snapshot(include_matrices=False)
        assert state["converged"]
        assert context.memory_peak_bytes <= budget
        assert "coefficients" not in state
        explicit = result.snapshot()
        assert explicit["density_matrices"].shape == (1, 16, 1, 1)
        assert np.allclose(explicit["occupations"], 1., atol=1e-11)
        states.append(state)
        del result, problem
        assert context.memory_used_bytes == 0
    assert abs(states[0]["free_energy"] - states[1]["free_energy"]) < 1e-12
    assert np.array_equal(states[0]["density"], states[1]["density"])
