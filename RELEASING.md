# Releasing `aether-context`

This project publishes to [PyPI](https://pypi.org/project/aether-context/) automatically
when a version tag is pushed. The release is built and uploaded by
[`.github/workflows/publish.yml`](.github/workflows/publish.yml) using **PyPI OIDC Trusted
Publishing** — there is **no API token** stored in the repo or in GitHub secrets.

Source of truth for the repo: <https://github.com/DBarr3/Unlimited-Context-LLM>

---

## One-time PyPI setup (do this BEFORE the first tag)

The publish workflow authenticates to PyPI via OIDC. PyPI will reject the upload unless a
matching **Trusted Publisher** has been configured for the project. Set this up once, before
pushing the very first tag, or the publish job fails.

1. Sign in at <https://pypi.org>.
2. If the project does not exist yet, create the Trusted Publisher under
   **Your projects → Publishing → Add a pending publisher** (a "pending" publisher creates the
   `aether-context` project on first successful upload). For an existing project use
   **Manage → Publishing → Add a new publisher**.
3. Enter exactly:
   - **PyPI project name:** `aether-context`
   - **Owner:** the GitHub org/user that owns the repo **at publish time** (currently `DBarr3`)
   - **Repository name:** `Unlimited-Context-LLM` (the repo's exact current name — OIDC matches
     this literally and does **not** follow GitHub's rename redirect, so it must be exact)
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`

> **Repo transfer note.** This repository may be transferred to an **aether-ai** org before the
> first release. The Trusted Publisher **owner** must match the repo's **final** owner at the
> moment a tag is pushed. If you transfer the repo after configuring the publisher, update (or
> re-add) the Trusted Publisher so its owner matches the new org — otherwise the OIDC claim will
> not match and the upload is rejected.

4. (GitHub) Confirm a repo **Environment** named `pypi` exists
   (**Settings → Environments**). The workflow's `environment: pypi` references it; you can attach
   required reviewers there if you want a manual approval gate before each publish.

---

## Cutting a release

1. **Bump the version in BOTH places it lives.** They are separate strings and nothing keeps
   them in sync:

   ```toml
   # pyproject.toml
   [project]
   version = "X.Y.Z"
   ```

   ```python
   # aether_context/__init__.py
   __version__ = "X.Y.Z"
   ```

   `cli.py` imports `__version__`, so the second one is what `aether-context --version`
   prints. Bumping only `pyproject.toml` ships a CLI that reports the previous version.

2. **Update [`CHANGELOG.md`](CHANGELOG.md):** move items out of `## [Unreleased]` into a new
   `## [X.Y.Z] — YYYY-MM-DD` section. The format follows
   [Keep a Changelog](https://keepachangelog.com/) and
   [SemVer](https://semver.org/).

3. **Commit** the bump on `main` (or via PR):

   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "release: vX.Y.Z"
   ```

4. **Tag, push, and cut the GitHub Release:**

   ```bash
   git tag -a vX.Y.Z -m "aether-context vX.Y.Z"
   git push origin vX.Y.Z
   gh release create vX.Y.Z --title "vX.Y.Z — <one concrete capability>" --notes-file notes.md
   ```

   The tag **no longer triggers a publish**. `publish.yml` is dormant
   (`workflow_dispatch` only) because there is no `aether-context` project on PyPI and no
   trusted publisher configured — every tag used to produce a red X while nothing shipped.
   Tagging and publishing are separate decisions now.

   Until PyPI is set up, the supported install is straight from GitHub:

   ```bash
   pip install git+https://github.com/AetherAI3/Unlimited-Context-LLM.git@vX.Y.Z
   ```

   Say that in the release notes rather than `pip install aether-context`, which does not
   work yet.

5. **Verify.** Confirm the release renders correctly, then check the tag installs cleanly from
   a throwaway environment:

   ```bash
   pip install git+https://github.com/AetherAI3/Unlimited-Context-LLM.git@vX.Y.Z
   aether-context --version   # must print X.Y.Z, not the previous version
   ```

   Once PyPI is configured, additionally dispatch **publish** from the Actions tab against the
   tag, then confirm at <https://pypi.org/project/aether-context/>.

---

## Notes & troubleshooting

- **The tag no longer publishes anything.** `publish.yml` is `workflow_dispatch` only. It used
  to fire on `v*` tags, but with no trusted publisher configured every tag just produced a red
  X. Restore the `push: tags: v*` trigger once a manual dispatch has actually succeeded.
- **Re-tagging.** PyPI files are immutable — you cannot re-upload the same version. If a release
  is broken, bump to a new patch version and tag again.
- **Version mismatch.** The published version comes from `pyproject.toml`, not the tag string.
  Keep them in lockstep (tag `vX.Y.Z` ⇔ `version = "X.Y.Z"`).
- **OIDC failure ("not a trusted publisher").** Almost always an owner/repo/workflow/environment
  mismatch — re-check the four values above against the repo's current owner (see the transfer
  note).
- **Local dev hygiene.** Install the pre-commit hook so commits stay ruff-clean:
  `pip install pre-commit && pre-commit install` (config:
  [`.pre-commit-config.yaml`](.pre-commit-config.yaml)).
