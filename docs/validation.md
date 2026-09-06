# Validation record

Local validation on 2026-09-06, macOS arm64, Python 3.12. Source archive commit: `356e505`.

- **25 regression cases passed:** local/global GP updates, integer maps, revisits, reset, empty and invalid observations, supported expert fusion, standard-deviation oracle, variable observation shapes, and faithful archive export.
- **Real environment smoke check passed:** reset plus two steps, three agents, original 100×100 navigation grid, local GPs and the legacy moving-user generator. All observations and rewards were finite; observations were `(6, 100, 100)` with dtype `float32`. The `50` scenario contained 51 actual users. This check does not test learned-policy performance.
- **Browser checks passed:** all 45 archived runs selectable and scrubbed to their final record; playback advances; JSON export matches the selected experiment; no page exceptions; no horizontal overflow at 390px. Desktop and mobile screenshots were inspected. Browser QA used the already installed Playwright runtime, not a new project dependency.
- **Final exporter check:** 45 runs successfully embedded in a self-contained offline HTML document with a source-hash manifest. After the default overview adjustment, all five archive-export tests passed again.
- **Whitespace check:** `git diff --check` passed.

The exact local scientific/test dependencies are captured in `requirements-verified.txt`. The new GitHub Actions matrix has been configured but not executed remotely. Training, GPU execution, long-horizon collision safety, learning convergence and paper-level service metrics were not validated in this patch. SciPy reported deprecation warnings for legacy distance helpers; these were not test failures.

## New MPS pipeline validation

The final new pipeline passed 63 tests on the actual MPS-capable host, including forward/backward, PPO updates, checkpoint restore, CPU bit-exact continuation, resumed best-checkpoint handling, and common-horizon service accounting. Three training seeds completed 16,384 steps each. The full Chinese experiment record and limitations are in [mps-2026-09-06.md](experiments/mps-2026-09-06.md). Both the historical viewer and new actual-episode viewer passed browser interaction and responsive-layout checks. Remote CI remains configured but has not been run from this local branch.
