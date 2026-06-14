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

1. **Bump the version** in [`pyproject.toml`](pyproject.toml):

   ```toml
   [project]
   version = "X.Y.Z"
   ```

2. **Update [`CHANGELOG.md`](CHANGELOG.md):** move items out of `## [Unreleased]` into a new
   `## [X.Y.Z] — YYYY-MM-DD` section. The format follows
   [Keep a Changelog](https://keepachangelog.com/) and
   [SemVer](https://semver.org/).

3. **Commit** the bump on `main` (or via PR):

   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "release: vX.Y.Z"
   ```

4. **Tag and push.** The tag (prefixed with `v`) is what triggers the publish workflow:

   ```bash
   git tag -a vX.Y.Z -m "aether-context vX.Y.Z"
   git push origin vX.Y.Z
   ```

   Pushing the tag fires [`.github/workflows/publish.yml`](.github/workflows/publish.yml), which:
   `actions/checkout@v4` → `actions/setup-python@v5` (3.12) → `pip install build` →
   `python -m build` (sdist + wheel) → `pypa/gh-action-pypi-publish@release/v1` (OIDC, no token)
   → upload to PyPI.

5. **Verify.** Watch the **publish** run in the Actions tab, then confirm the new version at
   <https://pypi.org/project/aether-context/> and:

   ```bash
   pip install --upgrade aether-context==X.Y.Z
   ```

---

## Notes & troubleshooting

- **Tag format matters.** The workflow triggers on `v*` tags only. A bare `X.Y.Z` tag (no `v`)
  will not publish.
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
