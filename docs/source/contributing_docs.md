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
and on Read the Docs (`fail_on_warning: true` in `.readthedocs.yaml`), so a single
unresolved cross-reference fails the whole build and the site keeps serving the
previous version. The hosted build installs the package, which compiles the Rust
extension, so `.readthedocs.yaml` asks for a Rust toolchain (`build.tools.rust`).
Scientific figures are stored in `_static`; the build does
not execute simulations. See [the LiH scan](tutorial/pes_scan.md) to regenerate
its PNG, CSV and calculation metadata.

`make -C docs html SPHINXOPTS="-W"` runs the same build. The API reference imports the
package, so the extension must be built first (`pip install -e '.[dev]'`).

## How the manual is organized

Six kinds of page are kept apart, each with its own place in `docs/source/`:

| Kind | Folder | What belongs there |
| --- | --- | --- |
| Start | `installation.md`, `start/` | Installing, and one first calculation. |
| Guides | `guide/` | How to do one task. Every code block runs. |
| Tutorials | `tutorial/` | One calculation followed from start to finish. |
| Concepts | `theory/` | Why a method works and what it does not guarantee. |
| Reference | `api.md`, `reference/` | The options (the `Mandacaru` docstring), the methods table, the command line, the run log, and the not-yet-available table. |
| Engineering | `development/`, this page | Architecture, the one live status page, the history. |

Two rules keep the pages honest.

**One owner per fact.** A default or an option is stated once, in the options
reference, and the guides link to it. A feature the program does not implement
appears in [the not-yet-available table](reference/not_yet_available.md) with what
raises and where it stands, never in a guide as if it were usable. When a Rust
capability changes, update the guide that owns it, the methods table if a method
changed, and [the cutover page](development/native_cutover.md); do not describe a
cutover item as done before that page's coverage table says so.

**Examples that run.** Every `Mandacaru(...)` call in a guide must pass options the
constructor takes, and every `examples/` script a page names must exist.
`test/test_docs_links.py` checks the first by parsing the code blocks and the second by
looking for the file.

## What the link test checks

`python -m pytest test/test_docs_links.py` runs in milliseconds and needs no Sphinx
build. It fails on:

- a `#target` that no `(target)=` line defines, a link to a heading slug, and a link
  to a page that does not exist (next section);
- a keyword in a `Mandacaru(...)` call inside a Python code block that the constructor
  does not take;
- a dotted `mandacaru.` name in the text that does not import, in any page except the
  historical ones (`development/history.md`, `development/native_cutover.md`), which
  may name removed modules;
- a row of [the methods table](reference/methods.md) that is missing, extra or
  misspelled against the code;
- a constructor parameter without a `name : type` entry in the `Mandacaru`
  docstring;
- a script path under `examples/` that is not a file.

Write the name of a removed or planned feature as plain words or in a text block,
never as a dotted package path or as a Python call.

## Linking to a section

`myst_heading_anchors` is **not** enabled, so MyST generates no anchors from
headings and `[text](page.md#some-heading-slug)` resolves to nothing. Give the
section an explicit target instead, on the line above its heading, and link to
that name from anywhere in the manual:

```markdown
(run-log-forces)=
## The `[FORCES]` block
```

```markdown
... as the run log shows (see [the forces block](#run-log-forces)).
```

`test/test_docs_links.py` checks this without building: it fails on a link to a
heading slug, on a `#target` nothing defines, and on a link to a page that does
not exist.

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
