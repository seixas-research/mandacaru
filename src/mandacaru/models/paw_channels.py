"""Layout and validation of PAW projector metadata."""
from .._native import (
    paw_assemble_block_matrix,
    paw_projector_blocks,
)


def projector_blocks(projectors) -> dict:
    """Group projector positions by channel: ``{(atom, l, m): [p, ...]}``.

    Positions within a block are ordered by the projector's radial ``index``
    (0 for a single Kleinman-Bylander projector); a duplicate index inside a
    channel is an error, since the block's rows would be ambiguous.
    """
    return paw_projector_blocks(projectors)


def assemble_block_matrix(projectors, blocks, diagonal=None):
    r"""Assemble a ``(P, P)`` block-diagonal matrix over the projector channels.

    ``blocks`` maps ``(atom_index, l, m)`` to an ``(n, n)`` array for the ``n``
    radial projectors of that channel (every channel must be supplied, and no
    key may name a channel without projectors).  With ``blocks=None`` the
    matrix is ``diag(diagonal)`` -- the Kleinman-Bylander case of one energy
    per projector.
    """
    return paw_assemble_block_matrix(projectors, blocks, diagonal)
