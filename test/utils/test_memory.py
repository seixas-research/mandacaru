# -*- coding: utf-8 -*-
# file: test_memory.py

"""The memory the process may hold and the memory it can take now."""

from mandacaru.utils import memory
from mandacaru.utils.memory import cgroup_memory


class TestAvailableMemory:
    """The guards read what can be allocated now, not the installed memory."""

    @staticmethod
    def _cgroup(tmp_path, v2=True, limit="max", usage="0"):
        membership = tmp_path / "cgroup"
        root = tmp_path / "fs"
        if v2:
            membership.write_text("0::/job/step\n")
            group = root / "job" / "step"
            group.mkdir(parents=True)
            (group / "memory.max").write_text("max\n")
            (group / "memory.current").write_text(f"{usage}\n")
            (root / "job" / "memory.max").write_text(f"{limit}\n")
        else:
            membership.write_text("4:memory:/job\n3:cpu:/job\n")
            group = root / "memory" / "job"
            group.mkdir(parents=True)
            (group / "memory.limit_in_bytes").write_text(f"{limit}\n")
            (group / "memory.usage_in_bytes").write_text(f"{usage}\n")
        return str(root), str(membership)

    def test_a_v2_limit_is_the_tightest_ancestor(self, tmp_path):
        root, membership = self._cgroup(tmp_path, limit=str(4 * 2 ** 30),
                                        usage=str(2 ** 30))
        assert cgroup_memory(root, membership) == (4 * 2 ** 30, 2 ** 30)

    def test_v1_and_unlimited_groups(self, tmp_path):
        root, membership = self._cgroup(tmp_path, v2=False,
                                        limit=str(2 ** 33), usage="5")
        assert cgroup_memory(root, membership) == (2 ** 33, 5)
        unlimited = tmp_path / "unlimited"
        unlimited.mkdir()
        root, membership = self._cgroup(unlimited, v2=False,
                                        limit=str(2 ** 63 - 4096), usage="5")
        assert cgroup_memory(root, membership) == (None, 5)
        assert cgroup_memory(str(tmp_path), str(tmp_path / "absent")) == (
            None, None)

    def test_available_is_capped_by_the_group_and_meminfo(self, monkeypatch):
        monkeypatch.setattr(memory, "cgroup_memory",
                            lambda: (3 * 2 ** 30, 2 ** 30))
        available = memory.available_memory_bytes()
        assert 0 < available <= 2 * 2 ** 30
        assert memory.memory_limit_bytes() <= 3 * 2 ** 30
        monkeypatch.setattr(memory, "cgroup_memory", lambda: (None, None))
        # MemAvailable never exceeds the installed memory.
        assert (memory.available_memory_bytes()
                <= memory.memory_limit_bytes())
