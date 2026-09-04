# Releasing `aether-context`

`aether-context` ships to **two registries from one commit**: the Python package to
[PyPI](https://pypi.org/project/aether-context/), and an npm launcher that installs that exact
PyPI release. Both publishes are **manual workflow dispatches** — tagging and publishing are
separate decisions, so cutting a tag never pretends to ship a package.

- [`.github/workflows/publish.yml`](.github/workflows/publish.yml) — PyPI, via **OIDC Trusted
  Publishing**. No API token is stored in the repo or in GitHub secrets.
- [`.github/workflows/publish-npm.yml`](.github/workflows/publish-npm.yml) — npm, via an
  `NPM_TOKEN` automation token in the `npm-production` environment, published with
  `--provenance`.

Source of truth for the repo: <https://github.com/AetherAI3/Unlimited-Context-LLM>

> **Order matters: PyPI first, npm second.** The npm launcher pins
> `aether-context==<its own version>`, so publishing it before that version exists on PyPI ships
> a launcher that fails on every first run. The npm workflow has a preflight that refuses to
> publish in that state.

---

## One-time PyPI setup (do this BEFORE the first publish)

The publish workflow authenticates to PyPI via OIDC. PyPI rejects the upload unless a matching
**Trusted Publisher** exists.

1. Sign in at <https://pypi.org>.
2. The project does not exist yet, so add a **pending** publisher under
   **Your account → Publishing → Add a pending publisher**. (For an existing project it is
   **Manage → Publishing → Add a new publisher**.)
3. Enter exactly:

   | Field | Value |
   | --- | --- |
   | **PyPI project name** | `aether-context` |
   | **Owner** | `AetherAI3` |
   | **Repository name** | `Unlimited-Context-LLM` |
   | **Workflow name** | `publish.yml` |
   | **Environment name** | `pypi-production` |

   > **The project-name field is not the repo name.** It must equal the distribution name in
   > `pyproject.toml` (`aether-context`). Getting this wrong does *not* fail the OIDC exchange —
   > authentication succeeds and the **upload** is then rejected with
   > `400 Non-user identities cannot create new projects`, which reads like a permissions problem
   > but is a name mismatch. (Hyphen vs underscore is fine; PyPI normalizes those.)

   > **Owner must be the repo's owner at publish time.** OIDC matches it literally and does
   > **not** follow GitHub's transfer redirect — this repo moved from `DBarr3` to `AetherAI3`, so
   > a publisher naming `DBarr3` will never match.

4. (GitHub) Confirm a repo **Environment** named `pypi-production` exists
   (**Settings → Environments**). The workflow's `environment: pypi-production` references it;
   attach required reviewers there if you want a manual approval gate before each publish.

## One-time npm setup

1. Create the **`npm-production`** environment (**Settings → Environments**).
2. Generate an **Automation** token at npmjs.com → **Access Tokens** (automation tokens bypass
   2FA, which CI cannot answer), and add it to that environment as the secret **`NPM_TOKEN`**:

   ```bash
   gh secret set NPM_TOKEN --env npm-production --repo AetherAI3/Unlimited-Context-LLM
   ```

---

## Cutting a release

1. **Bump the version** in [`aether_context/__init__.py`](aether_context/__init__.py):

   ```python
   __version__ = "X.Y.Z"
   ```

   That is the single source of truth. `pyproject.toml` declares `dynamic = ["version"]` and
   reads the attribute, so there is no second literal to keep in sync.

2. **Bump the npm launcher** in
   [`packages/npm-cli/package.json`](packages/npm-cli/package.json) to the *same* `X.Y.Z`. The
   launcher installs the PyPI release matching its own version, and
   `tests/test_release_parity.py` fails the build if the two drift.

3. **Update [`CHANGELOG.md`](CHANGELOG.md):** move items out of `## [Unreleased]` into a new
   `## [X.Y.Z] — YYYY-MM-DD` section. The format follows
   [Keep a Changelog](https://keepachangelog.com/) and
   [SemVer](https://semver.org/).

4. **Commit and merge** to `main` via PR, and let CI go green.

5. **Tag the merge commit and push:**

   ```bash
   git tag -a vX.Y.Z -m "aether-context vX.Y.Z"
   git push origin vX.Y.Z
   ```

   The tag does not trigger anything — it is the auditable ref you publish *from*.

6. **Publish to PyPI**, passing the tag as the `ref` input:

   ```bash
   gh workflow run publish.yml -f ref=vX.Y.Z
   ```

   The job builds an sdist and wheel, asserts the wheel contains no `aether_agent` files, runs
   `twine check`, then uploads via OIDC.

7. **Publish to npm**, once PyPI shows the new version:

   ```bash
   gh workflow run publish-npm.yml -f ref=vX.Y.Z
   ```

   Add `-f dry_run=true` first to see the packed tarball without publishing.

8. **Verify both:**

   ```bash
   pip install --upgrade "aether-context==X.Y.Z" && aether-context --version
   npx --yes aether-context@X.Y.Z --version
   ```

---

## What the distribution contains — and deliberately does not

The wheel ships **`aether_context` only**, and declares one console script, `aether-context`.

`aether_agent/` also lives in this repo and is exercised by the test suite, but it is **not
packaged**. PyPI's separate [`aether-agent`](https://pypi.org/project/aether-agent/)
distribution already owns that import path and the `aether` command, and pip does not detect
file conflicts *across* distributions — shipping a second copy would silently overwrite the
other package's files on any machine that installed both. `publish.yml` re-checks the built
wheel for `aether_agent` files and fails the release if any appear.

---

## Notes & troubleshooting

- **`400 Non-user identities cannot create new projects`.** The pending publisher's *project
  name* does not match the distribution name. Fix that field to `aether-context`; nothing is
  uploaded on a failed attempt, so the version is still free.
- **`invalid-publisher` at the OIDC step.** An owner / repository / workflow / environment
  mismatch. Re-check all four against the table above — the environment is `pypi-production`,
  not `pypi`.
- **Re-publishing.** PyPI files are immutable: a version cannot be re-uploaded, even after it is
  deleted. If a release is broken, bump to a new patch version.
- **Version mismatch.** The published version comes from `aether_context.__version__`, not the
  tag string. Keep them in lockstep (tag `vX.Y.Z` ⇔ `__version__ = "X.Y.Z"`).
- **npm 2FA.** A *publish* token fails in CI when 2FA is enforced; use an **automation** token.
- **Local dev hygiene.** Install the pre-commit hook so commits stay ruff-clean:
  `pip install pre-commit && pre-commit install` (config:
  [`.pre-commit-config.yaml`](.pre-commit-config.yaml)).
