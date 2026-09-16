"""Use British spelling in rendered prose, preserving the public Python API.

Autodoc includes upstream docstrings with mixed spelling. Apply this small
presentation-only glossary to prose nodes, never to code, signatures, maths,
URLs or reference targets. Authored manual pages use British spelling directly.
"""

import re

from docutils import nodes
from sphinx import addnodes

WORDS = {
    "behavior": "behaviour", "behaviors": "behaviours",
    "color": "colour", "colors": "colours", "colored": "coloured",
    "coloring": "colouring", "center": "centre", "centers": "centres",
    "centered": "centred", "centering": "centring",
    "license": "licence", "licenses": "licences",
    "artifact": "artefact", "artifacts": "artefacts",
    "neighbor": "neighbour", "neighbors": "neighbours",
    "neighboring": "neighbouring", "analyze": "analyse", "analyzes": "analyses",
    "analyzed": "analysed", "analyzing": "analysing",
    "honor": "honour", "honors": "honours", "honored": "honoured",
    "honoring": "honouring", "modeling": "modelling", "modeled": "modelled",
    "labeled": "labelled", "labeling": "labelling", "canceled": "cancelled",
    "utilizing": "using", "toward": "towards",
}
for stem in (
    "optim", "minim", "maxim", "normal", "orthonormal", "orthogonal",
    "diagonal", "serial", "polar", "local", "delocal", "discret",
    "parallel", "vector", "general", "initial", "final", "standard",
    "parametr", "parameter", "factor", "real", "material", "quant",
    "regular", "symmetr", "random", "summar", "organ", "recogn", "pseud", "linear",
    "special", "neutral", "synthes",
):
    for ending in ("ize", "izes", "ized", "izing", "ization", "izations", "izer", "izers"):
        WORDS[stem + ending] = stem + ending.replace("z", "s")
PATTERN = re.compile(r"\b(?:" + "|".join(WORDS) + r")\b", re.IGNORECASE)


def british(text):
    """Convert words without changing identifiers containing underscores."""
    def replace(match):
        original = match.group()
        word = WORDS[original.lower()]
        if original.isupper():
            return word.upper()
        if original[0].isupper():
            return word.capitalize()
        return word
    return PATTERN.sub(replace, text)


def translate_prose(app, doctree, docname):
    protected = (nodes.literal, nodes.literal_block, nodes.math, nodes.math_block,
                 nodes.raw, nodes.reference, addnodes.desc_signature)
    for node in list(doctree.findall(nodes.Text)):
        parent = node.parent
        while parent is not None and not isinstance(parent, protected):
            parent = parent.parent
        if parent is not None:
            continue
        text = british(str(node))
        if text != str(node):
            node.parent.replace(node, nodes.Text(text))


def setup(app):
    app.connect("doctree-resolved", translate_prose)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
