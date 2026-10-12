# Copilot review instructions

`photon-mosaic` is a calcium-imaging analysis package: ROI extraction, neuropil and background
estimation, fluorescence and dF/F traces, deconvolution. Much of it ports numerical methods from
Suite2p and CaImAn. Review accordingly — the expensive bugs here are numerical and silent, not
stylistic.

## Branching: `dev` is the only baseline

The default branch is `dev`. `main` is not the development baseline and is often far behind.

Features are developed as **stacked pull requests**: a chain of small PRs where each targets the
previous one rather than `dev`, so each reviewable step is a single idea. A typical chain looks
like `feature-a → feature-b → feature-c → dev`.

When reviewing a PR whose base is **not** `dev`, that base branch is unreleased work in progress.
Therefore:

- **Do not raise backward-compatibility findings against it.** An API, parameter, data key or
  on-disk layout that exists only on an unmerged parent branch has never shipped. Changing or
  removing it breaks nothing.
- **Do not request data migrations** for formats introduced earlier in the same stack. No saved
  analyzer on any user's disk can contain them.
- **Do not flag functionality as missing** when it is scheduled for a later PR in the stack.
  Staging a feature across a chain is the intended workflow, not an oversight. A docstring that
  describes the finished feature while the integration lands two PRs later is deliberate.
- **But a reference that is broken on this branch is still a defect.** A `:func:`/`:class:`
  cross-reference, import, attribute or call whose target does not exist in this PR's tree breaks
  the docs build or the code *here*, whichever later PR introduces the target. Report those — the
  exemption above covers behaviour that is deferred, not names that do not resolve yet.

To judge whether something is genuinely a compatibility concern, compare against `dev`, not against
the PR's base branch.

## Project invariants worth checking against

These are real constraints. Code that violates one is a genuine finding.

**Determinism.** Extension results are cached on disk and compared across runs, so every
computation must be reproducible bit-for-bit. No unseeded RNG, no unordered iteration, no
dependence on thread or chunk scheduling. Where a library call has a hidden random component
(e.g. an ARPACK start vector), it must be pinned explicitly. Several functions document this
requirement inline — treat those comments as binding.

**Numerical precision.** The movie is float32, but fit objectives are differences of large terms
that cancel to something orders of magnitude smaller. Reductions against movie data are
accumulated in float64 on purpose, in bounded-memory blocks. Flag a float32 reduction in an
objective or Gram computation; do not "simplify" an explicit float64 upcast away.

**Memory.** Movies do not fit in RAM. Anything touching the full movie must stream in chunks.
Flag code that materializes `(n_frames, n_pixels)` densely.

**Ridge regularization.** `ridge` is applied relative to **each diagonal entry** of a Gram matrix,
never to its mean diagonal. This is deliberate: in a joint `[A, b]` Gram a background component's
diagonal is orders of magnitude larger than an ROI's, so a mean-scaled ridge would crush the ROI
block. Do not suggest mean-scaling it.

**Non-negativity.** Clipping an unconstrained least-squares solution at zero is *not* the
non-negative least-squares minimizer unless the Gram is diagonal. Overlapping ROIs and multiple
background components both produce non-diagonal Grams. Flag clipping used as a stand-in for NNLS,
and flag an NNLS solve that silently drops regularization its unconstrained counterpart applies.

## Extension architecture

Analysis steps are `AnalyzerExtension` subclasses in `core/roianalyzer_core_extensions.py`,
registered via `register_result_extension`. Three registration points must stay in sync, and they
live in different files:

- `depend_on` / `get_required_dependencies()` — drives recompute invalidation through
  `_extension_children`.
- `get_optional_dependencies()` — affects batch ordering only. **Optional dependencies are not
  wired into `_extension_children`**, so recomputing an optional parent does not invalidate its
  children. This is a known pre-existing gap, not something a given PR introduced; report it once
  against `core/roianalyzer.py` rather than on every PR that adds an optional dependency.
- `_builtin_extensions` in `core/roianalyzer.py` — backs `get_computable_extensions()` and lazy
  auto-import. A new extension must be added here or it stays invisible to the public API.

## What is most useful to report

Ranked by how much the maintainers want it:

1. **Code that contradicts its own docstring or inline comment.** This repo documents its numerical
   reasoning heavily, which makes the documentation a usable oracle. A function whose docstring
   states an invariant the body violates is the highest-value finding available, and these have
   been correct in past reviews.
2. **Prose in `examples/*.ipynb` that contradicts the source.** The notebooks are the package's
   public tutorial and drift badly: past reviews found them documenting methods and data keys that
   do not exist, and describing two code paths as interchangeable when the API rejects one of them.
   Check every API name a markdown cell mentions against the source, and check its claims against
   the code cells around it — the notebook frequently contradicts itself before it contradicts the
   source. A PR that changes the code cells but not the surrounding prose is a common failure here.
3. **Silently wrong numerics** — unvalidated parameters that corrupt a solve rather than raising,
   inconsistent regularization between branches, precision loss in a cancelling sum, a convergence
   test that can never fire.
4. **Public docstrings that are stale relative to the same PR** — a parameter list that omits a
   method the PR adds, or describes scaling the implementation does not use.
5. **Registration and dependency wiring** that is incomplete across the three points above.

Lower value: style, naming, and type-annotation completeness. Do not report these unless they
affect correctness.

## Conventions

- Line length 120, enforced by ruff (`I`, `E`, `F`); `pre-commit` and `mypy` are configured.
- NumPy-style docstrings.
- Tests are `pytest`, under `src/photon_mosaic/core/tests/` and `tests/`.
- Scientific spelling in prose and docstrings is inconsistent across the codebase
  (`regularisation`/`regularization`); do not report it.
