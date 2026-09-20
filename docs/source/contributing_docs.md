# Building and editing the documentation

The manual uses Sphinx, MyST Markdown, reStructuredText and Furo, as does the
[Poraquê reference manual](https://poraque.readthedocs.io/en/latest/).
Run the following commands from the repository root:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html -W --keep-going docs/source docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open `http://localhost:8000`. A build warning is treated as a failure both locally
and on Read the Docs. Scientific figures are stored in `_static`; the build does
not execute simulations. See [the LiH scan](tutorial/pes_scan.md) to regenerate
its PNG, CSV and calculation metadata.

## American English and readable source

Write prose, headings, captions, alternative text and code comments in American
English: *behavior*, *color*, *optimize*, *normalize*, *center* and *license*.

Python identifiers, API arguments, dictionary keys and test names follow the same
rule: `optimizer`, `centers`, `neighbor`.

Use four spaces for Python indentation. Surround block directives and lists with
blank lines. In reStructuredText, indent directive options and content by three
spaces, and leave a blank line before the content. Check that heading underlines
are at least as long as their headings.

## Mathematics

In Markdown, use `$...$` for inline mathematics and a fenced MyST directive for
a displayed equation:

````markdown
The trial energy is $E(\boldsymbol{\theta})$.

```{math}
E(\boldsymbol{\theta}) =
\langle\psi(\boldsymbol{\theta})|\hat H|\psi(\boldsymbol{\theta})\rangle.
```
````

In reStructuredText, use `:math:` for inline mathematics and `.. math::` for a
block. Do not place mathematics in a plain code fence. Python docstrings with
LaTeX backslashes should use a raw string (`r"""..."""`) so sequences such as
`\alpha` and `\beta` do not become control characters before Sphinx sees them.

Sphinx's MathJax extension renders the generated maths nodes in the browser.
The MyST `dollarmath` and `amsmath` extensions also support existing dollar
delimiters and LaTeX environments. `myst_dmath_double_inline` does not enable
`\(...\)` shortcuts and is not needed here.

The `colon_fence` extension renders the existing `:::{note}` and
`:::{warning}` blocks in the guides. A plain Markdown code fence would display
their contents as code rather than as an admonition.

## Theme and diagrams

Keep Furo's color variables and link rules. The former custom stylesheet
forced dark blue links and targeted `html[data-theme="dark"]`; Furo places
`data-theme` on `body`. Automatic mode also follows `prefers-color-scheme`.
Removing those overrides restores the same native theme behavior used by
Poraquê, including nested, current-page and hovered links.

The homepage workflow is a semantic ordered list styled with theme variables.
It wraps on small screens and needs no external diagram renderer. The generated
PNG has its own white background so its labels remain legible in either theme.

After editing, check explicit light and dark mode, automatic mode with both
system preferences, nested navigation, the current page, hover and keyboard
focus. Check the homepage and LiH figure at a narrow screen width, and confirm
that equations render rather than displaying raw LaTeX.
