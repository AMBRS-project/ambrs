"""Guard numerical failures without changing the existing composition policy."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from ambrs import aerosol, ppe


class FixedSamples:
    """Return specified member values through either public sampling interface."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def rvs(self, size):
        assert size == self.values.size
        return self.values.copy()

    def ppf(self, quantiles):
        assert np.shape(quantiles) == self.values.shape
        return self.values.copy()


def make_specification(fractions, *, number=None, extra_modes=()):
    species = tuple(
        aerosol.AerosolSpecies(name, molar_mass=12.0, density=1000.0)
        for name in ("BC", "OC", "H2O")[: len(fractions)]
    )
    return ppe.EnsembleSpecification(
        name="normalization regression",
        aerosols=species,
        gases=(),
        size=aerosol.AerosolModalSizeDistribution(
            modes=(
                aerosol.AerosolModeDistribution(
                    name="test mode",
                    species=species,
                    number=stats.uniform(1.0e8, 1.0e8) if number is None else number,
                    geom_mean_diam=stats.uniform(1.0e-7, 1.0e-7),
                    log10_geom_std_dev=stats.uniform(0.1, 0.1),
                    mass_fractions=fractions,
                ),
                *extra_modes,
            )
        ),
        gas_concs=(),
        flux=stats.uniform(0.0, 1.0),
        relative_humidity=stats.uniform(0.1, 0.5),
        temperature=stats.uniform(280.0, 20.0),
        pressure=stats.uniform(100000.0, 1000.0),
        height=500.0,
    )


def run_sampler(sampler, specification, n=4):
    if sampler == "lhs":
        return ppe.lhs(specification, n, seed=123)
    return ppe.sample(specification, n)


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
@pytest.mark.parametrize(
    "bad_value", [0.0, np.nan, np.inf, -np.inf, np.finfo(float).max]
)
def test_invalid_total_identifies_the_mode_and_member(sampler, bad_value):
    # Member 2 is invalid; the other members have an ordinary positive total.
    specification = make_specification(
        (
            FixedSamples([1.0, bad_value, 1.0, 1.0]),
            FixedSamples([1.0, bad_value, 1.0, 1.0]),
        )
    )

    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match=r"test mode.*member.*\[2\]"):
            run_sampler(sampler, specification)


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
def test_zero_total_from_cancelling_weights_is_rejected(sampler):
    specification = make_specification(
        (
            FixedSamples([1.0, 1.0, 1.0, 1.0]),
            FixedSamples([1.0, -1.0, 1.0, 1.0]),
        )
    )

    with pytest.raises(ValueError, match=r"test mode.*member.*\[2\]"):
        run_sampler(sampler, specification)


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
def test_zero_total_is_rejected_even_when_particle_number_is_zero(sampler):
    specification = make_specification(
        (FixedSamples([0.0] * 4),), number=FixedSamples([0.0] * 4)
    )

    with pytest.raises(
        ValueError,
        match=r"test mode.*zero or non-finite total.*member.*\[1, 2, 3, 4\]",
    ):
        run_sampler(sampler, specification)


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
def test_invalid_second_mode_identifies_its_name_and_member(sampler):
    first_mode = make_specification((FixedSamples([1.0] * 4),)).size.modes[0]
    second_mode = replace(
        first_mode,
        name="second mode",
        mass_fractions=(FixedSamples([1.0, 1.0, 0.0, 1.0]),),
    )
    specification = make_specification(
        first_mode.mass_fractions, extra_modes=(second_mode,)
    )

    with pytest.raises(ValueError, match=r"second mode.*member.*\[3\]"):
        run_sampler(sampler, specification)


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
@pytest.mark.parametrize("failure", ["zero total", "non-finite result"])
def test_invalid_member_diagnostic_shows_only_the_first_ten(sampler, failure):
    n = 24
    weights = np.ones(n)
    if failure == "zero total":
        weights[1::2] = 0.0
        fractions = (FixedSamples(weights),)
    else:
        weights[1::2] = np.nextafter(0.0, 1.0)
        fractions = (
            FixedSamples(np.ones(n)),
            FixedSamples(-np.ones(n)),
            FixedSamples(weights),
        )
    specification = make_specification(fractions)

    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="test mode") as error:
            run_sampler(sampler, specification, n)

    message = str(error.value)
    # Twelve members fail; report the first ten using 1-based member indices.
    expected_members = list(range(2, 21, 2))
    assert f"ensemble member(s) {expected_members} " in message
    assert "(1-based indices, first 10 shown)." in message
    if failure == "zero total":
        assert "zero or non-finite total" in message
    else:
        assert "non-finite result" in message


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
def test_overflow_after_finite_nonzero_total_is_rejected(sampler):
    tiny = np.nextafter(0.0, 1.0)
    specification = make_specification(
        (
            FixedSamples([1.0] * 4),
            FixedSamples([-1.0] * 4),
            FixedSamples([tiny] * 4),
        )
    )

    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match=r"test mode.*member.*\[1, 2, 3, 4\]"):
            run_sampler(sampler, specification)


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
@pytest.mark.parametrize(
    "weights",
    [
        (0.0, 2.0),
        (0.2, 0.6),
        (-1.0, 3.0),
        (-1.0, -3.0),
        (1.0e-300, 3.0e-300),
        (1.0e300, 3.0e300),
    ],
)
def test_existing_finite_normalization_is_preserved(sampler, weights):
    specification = make_specification(
        tuple(FixedSamples([weight] * 4) for weight in weights)
    )

    ensemble = run_sampler(sampler, specification)

    for values, weight in zip(ensemble.size.modes[0].mass_fractions, weights):
        np.testing.assert_array_equal(values, np.full(4, weight / sum(weights)))


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
def test_legacy_zero_width_uniform_still_means_zero(sampler):
    specification = make_specification(
        (
            stats.uniform(0.0, 0.0),
            FixedSamples([2.0] * 4),
        )
    )

    ensemble = run_sampler(sampler, specification)

    zero, nonzero = ensemble.size.modes[0].mass_fractions
    np.testing.assert_array_equal(zero, np.zeros(4))
    np.testing.assert_array_equal(nonzero, np.ones(4))


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
def test_legacy_zero_width_uniform_with_zero_total_is_rejected(sampler):
    specification = make_specification((stats.uniform(0.0, 0.0),))

    with pytest.raises(ValueError, match=r"test mode.*member.*\[1, 2, 3, 4\]"):
        run_sampler(sampler, specification)


def test_lhs_keeps_its_existing_nan_to_zero_behavior():
    specification = make_specification(
        (
            FixedSamples([np.nan, 1.0, np.nan, 1.0]),
            FixedSamples([2.0] * 4),
        )
    )

    ensemble = ppe.lhs(specification, 4, seed=123)

    fractions = ensemble.size.modes[0].mass_fractions
    np.testing.assert_array_equal(fractions[0], [0.0, 1.0 / 3.0, 0.0, 1.0 / 3.0])
    np.testing.assert_array_equal(fractions[1], [1.0, 2.0 / 3.0, 1.0, 2.0 / 3.0])


def test_seeded_lhs_restores_rng_state_when_normalization_fails():
    specification = make_specification((FixedSamples([0.0] * 4),))
    original_state = np.random.get_state()
    try:
        np.random.seed(987)
        before = np.random.get_state()

        with pytest.raises(ValueError, match="test mode"):
            ppe.lhs(specification, 4, seed=123)

        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]
    finally:
        np.random.set_state(original_state)


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
def test_empty_composition_keeps_its_existing_behavior(sampler):
    ensemble = run_sampler(sampler, make_specification(()))

    assert ensemble.size.modes[0].mass_fractions == ()


def test_sample_of_zero_members_keeps_its_existing_behavior():
    specification = make_specification((FixedSamples([]),))

    ensemble = ppe.sample(specification, 0)

    assert len(ensemble) == 0
    assert ensemble.size.modes[0].mass_fractions[0].shape == (0,)


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
def test_near_cancellation_rejects_huge_but_finite_fractions(sampler):
    # Member 2 has a nonzero but roundoff-sized sum: 0.1 + 0.2 - 0.3.
    specification = make_specification(
        (
            FixedSamples([0.8, 0.1 + 0.2, 0.8, 0.8]),
            FixedSamples([0.2, -0.3, 0.2, 0.2]),
        )
    )
    with pytest.raises(ValueError, match=r"test mode.*severe cancellation.*\[2\]"):
        run_sampler(sampler, specification)


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
def test_small_finite_positive_weights_are_not_near_cancellation(sampler):
    # The check is relative: small total mass alone is not a failure.
    specification = make_specification(
        (FixedSamples([1e-200] * 4), FixedSamples([3e-200] * 4))
    )
    ensemble = run_sampler(sampler, specification)
    fractions = ensemble.size.modes[0].mass_fractions
    np.testing.assert_allclose(fractions[0], [0.25] * 4)
    np.testing.assert_allclose(fractions[1], [0.75] * 4)


@pytest.mark.parametrize("sampler", ["sample", "lhs"])
def test_finite_negative_weights_without_cancellation_remain_allowed(sampler):
    specification = make_specification(
        (FixedSamples([-1.0] * 4), FixedSamples([3.0] * 4))
    )
    ensemble = run_sampler(sampler, specification)
    fractions = ensemble.size.modes[0].mass_fractions
    np.testing.assert_allclose(fractions[0], [-0.5] * 4)
    np.testing.assert_allclose(fractions[1], [1.5] * 4)
