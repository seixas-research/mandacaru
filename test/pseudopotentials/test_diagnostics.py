# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_diagnostics.py

# This code is part of Mandacaru.
# MIT License

"""The checks of a dataset through the native engine (:mod:`mandacaru.pseudopotentials.diagnostics`):
a clean hydrogen dataset passes them, a damaged one does not (the negative control), and a
check that cannot be evaluated raises."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mandacaru.pseudopotentials import diagnostics as dx
from mandacaru.pseudopotentials.generation import generate_paw


@pytest.fixture(scope="module")
def hydrogen():
    return generate_paw("H")


def damaged(pp, scale):
    """The same dataset with the l=0 screened coupling multiplied by ``scale``."""
    s = pp.channels[0]
    channels = dict(pp.channels)
    channels[0] = dataclasses.replace(s, coupling_screened=s.coupling_screened * scale)
    return dataclasses.replace(pp, channels=channels,
                               coupling_screened={**pp.coupling_screened, 0: s.coupling_screened * scale})


class TestDiagnose:
    def test_a_clean_dataset_has_no_defect(self, hydrogen):
        found = dx.diagnose(hydrogen, levels=True)
        assert found["ghosts"] == {} and found["missing"] == {} and found["branch_ghosts"] == {}
        assert found["wrong_phases"] == {}
        assert found["defects"]["ghosts"] == {} and found["defects"]["phases"] == {}
        assert set(found["phases"]) == {0, 1}
        # The frozen numbers of the generator this replaced (A0 freeze, lda-sr H, attempt 0).
        assert found["phases"][0] == pytest.approx((3.5248043763e-05, 2.1727727445e-04), rel=1e-6)
        assert all(len(v) == 2 for v in found["phases"].values())
        assert 0 <= found["s_miss"] < 1.0
        assert set(found["levels"]) == {0, 1}

    def test_the_levels_are_the_spectrum(self, hydrogen):
        found = dx.diagnose(hydrogen, levels=True)
        assert np.allclose(found["levels"][0], dx.paw_spectrum(hydrogen, 0, n_states=2), atol=1e-12)
        # The bound reference is the lowest level; a ghost would sit below it.
        assert found["levels"][0][0] == pytest.approx(hydrogen.channels[0].reference_energies[0], abs=1e-3)

    def test_a_dataset_that_scatters_wrongly_is_found(self, hydrogen):
        found = dx.diagnose(damaged(hydrogen, 0.0))      # the s projectors stop acting
        assert list(found["wrong_phases"]) == [0] and found["wrong_phases"][0][0] > 0.1
        assert found["missing"] == {0: pytest.approx(0.0556, abs=1e-3)}
        assert found["defects"]["phases"] == found["wrong_phases"]
        assert found["defects"]["missing"] == found["missing"]


class TestRefusals:
    def test_a_number_that_json_cannot_hold_raises_before_the_engine(self, hydrogen):
        broken = damaged(hydrogen, float("nan"))
        with pytest.raises(ValueError, match="Out of range float values"):
            dx.diagnose(broken)


class TestSpectrumAndLogDerivative:
    def test_another_stride_is_not_available(self, hydrogen):
        with pytest.raises(NotImplementedError, match="stride 2"):
            dx.paw_spectrum(hydrogen, 0, stride=1)

    def test_a_channel_that_does_not_exist_raises(self, hydrogen):
        with pytest.raises(ValueError, match="no channel l=5"):
            dx.paw_spectrum(hydrogen, 5)

    def test_the_smooth_log_derivative_is_the_all_electron_one_at_the_reference(self, hydrogen):
        channel = hydrogen.channels[0]
        e0 = channel.reference_energies[0]
        smooth = dx.log_derivative_paw(hydrogen, 0, e0)
        r = hydrogen.r
        i = int(np.searchsorted(r, hydrogen.projector_radius(0)))
        ae = channel.ae_waves[0]
        exact = (ae[i + 1] - ae[i - 1]) / (r[i + 1] - r[i - 1]) / ae[i]    # R'/R of the stored R(r)
        assert smooth == pytest.approx(exact, abs=5e-3 * max(1.0, abs(exact)))

    def test_a_channel_without_projectors_needs_a_radius(self, hydrogen):
        no_p = dataclasses.replace(hydrogen, projectors={})
        with pytest.raises(ValueError, match="has no projectors: pass r_cut"):
            dx.log_derivative_paw(no_p, 1, -0.1)


class TestReconstruct:
    def test_a_zero_function_stays_zero_and_the_shape_is_kept(self, hydrogen):
        r = np.linspace(0.05, 3.0, 400)
        out = dx.reconstruct_ae(hydrogen, 0, r, np.zeros_like(r))
        assert out.shape == r.shape and np.all(out == 0.0)

    def test_the_smooth_wave_goes_back_to_the_all_electron_wave(self, hydrogen):
        channel = hydrogen.channels[0]
        r = hydrogen.r
        u = channel.pseudo_waves[0] * r
        ae = dx.reconstruct_ae(hydrogen, 0, r, u)
        reference = channel.ae_waves[0] * r
        assert np.max(np.abs(ae - reference)) < 5e-3 * np.max(np.abs(reference))
