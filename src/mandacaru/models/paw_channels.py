"""Layout and validation of PAW projector metadata."""
import numpy as np

def projector_blocks(projectors) -> dict:
    """Group projector positions by channel: ``{(atom, l, m): [p, ...]}``.

    Positions within a block are ordered by the projector's radial ``index``
    (0 for a single Kleinman-Bylander projector); a duplicate index inside a
    channel is an error, since the block's rows would be ambiguous.
    """
    groups: dict = {}
    for position, projector in enumerate(projectors):
        key = (int(projector.atom_index), *(int(v) for v in projector.channel))
        groups.setdefault(key, []).append(
            (int(getattr(projector, "index", 0)), position))
    ordered = {}
    for key, entries in groups.items():
        indices = [i for i, _p in entries]
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"projectors of channel (atom, l, m)={key} carry duplicate "
                f"radial indices {sorted(indices)}")
        ordered[key] = [p for _i, p in sorted(entries)]
    return ordered

def assemble_block_matrix(projectors, blocks, diagonal=None) -> np.ndarray:
    r"""Assemble a ``(P, P)`` block-diagonal matrix over the projector channels.

    ``blocks`` maps ``(atom_index, l, m)`` to an ``(n, n)`` array for the ``n``
    radial projectors of that channel (every channel must be supplied, and no
    key may name a channel without projectors).  With ``blocks=None`` the
    matrix is ``diag(diagonal)`` -- the Kleinman-Bylander case of one energy
    per projector.
    """
    P = len(projectors)
    dtype = np.result_type(float, *(np.asarray(b) for b in
                                    (blocks or {}).values()),
                           *(() if diagonal is None
                             else (np.asarray(list(diagonal)),)))
    out = np.zeros((P, P), dtype=dtype)
    if blocks is None:
        if diagonal is None:
            raise ValueError("either blocks or a diagonal must be given")
        diagonal = np.asarray(list(diagonal), dtype=dtype)
        if diagonal.shape != (P,):
            raise ValueError(f"expected {P} diagonal entries, got "
                             f"{diagonal.shape}")
        out[np.diag_indices(P)] = diagonal
        return out
    groups = projector_blocks(projectors)
    normalized = {tuple(int(v) for v in key): value
                  for key, value in blocks.items()}
    missing = sorted(set(groups) - set(normalized))
    extra = sorted(set(normalized) - set(groups))
    if missing or extra:
        raise ValueError(
            "nonlocal blocks do not match the projector channels: "
            f"missing {missing}, unmatched {extra}")
    for key, positions in groups.items():
        block = np.asarray(normalized[key], dtype=dtype)
        n = len(positions)
        if block.shape != (n, n):
            raise ValueError(
                f"block for channel {key} must be ({n}, {n}), got {block.shape}")
        out[np.ix_(positions, positions)] = block
    return out
