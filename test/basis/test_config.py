"""Electron configurations: aufbau filling, the valence set and the
single-electron rearrangements a reference atom is chosen among."""
from mandacaru.basis._config import (ground_state_config, rearrangements,
                                     valence_subshells)


class TestRearrangements:
    def test_the_first_candidate_is_aufbau(self):
        for Z in (8, 26, 57, 90):
            assert rearrangements(Z)[0] == ground_state_config(Z)

    def test_a_main_group_atom_has_nothing_to_rearrange(self):
        """Only s, d and f compete; an s -> p move is an excitation."""
        assert rearrangements(8) == [ground_state_config(8)]

    def test_every_candidate_keeps_the_electron_count(self):
        for Z in (26, 57, 89, 90):
            for candidate in rearrangements(Z):
                assert sum(candidate.values()) == Z

    def test_lanthanum_can_move_its_4f_electron_to_5d(self):
        moved = [c for c in rearrangements(57)
                 if c.get((5, 2), 0) == 1 and c.get((4, 3), 0) == 0]
        assert moved

    def test_no_destination_opens_a_new_principal_shell(self):
        """Lanthanum's 4f -> 7s would leave a one-electron valence."""
        for Z in (57, 89, 90):
            n_outer = max(n for n, _l in ground_state_config(Z))
            for candidate in rearrangements(Z):
                assert max(n for n, _l in candidate) <= n_outer

    def test_thorium_still_reaches_6d(self):
        """Only two empty subshells remain in the filling table at Z = 90."""
        assert any(c.get((6, 2), 0) > 0 for c in rearrangements(90))


class TestValenceSubshells:
    def test_the_valence_follows_the_configuration_given(self):
        config = dict(ground_state_config(57))
        config[(4, 3)] = 0
        config[(5, 2)] = 1
        valence = valence_subshells(57, configuration=config)
        assert (5, 2) in valence
        assert (4, 3) not in valence

    def test_without_a_configuration_it_is_aufbau(self):
        assert (valence_subshells(26)
                == valence_subshells(26, configuration=ground_state_config(26)))
