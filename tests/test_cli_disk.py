# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""The pool selector reserves local disk, so it must reject a size that won't fit.

Hermetic: no machine has billions of GB free, so an absurd ``--pool`` deterministically
trips the disk guard. The guard is a no-op only when free space can't be probed at all.
"""
from __future__ import annotations

from aether_context.cli import main

# Larger than any real disk, so the free-space check always rejects it.
_TOO_BIG_GB = "999999999"


def test_init_rejects_a_pool_larger_than_free_disk(tmp_pool_dir, capsys):
    code = main(["init", "--pool", _TOO_BIG_GB, "--dir", str(tmp_pool_dir)])
    assert code == 1
    err = capsys.readouterr().err
    assert "not enough disk" in err           # the flag + rejection message
    assert "--pool" in err                    # the fix hint suggests a smaller size


def test_resize_rejects_a_pool_larger_than_free_disk(tmp_pool_dir, capsys):
    assert main(["init", "--pool", "5", "--dir", str(tmp_pool_dir)]) == 0
    code = main(["--pool", _TOO_BIG_GB, "--dir", str(tmp_pool_dir)])  # top-level resize
    assert code == 1
    assert "not enough disk" in capsys.readouterr().err


def test_init_5gb_fits_and_reports_free_disk(tmp_pool_dir, capsys):
    code = main(["init", "--pool", "5", "--dir", str(tmp_pool_dir)])
    assert code == 0
    out = capsys.readouterr().out
    assert "GB free at" in out                # the friendlier disk read-out
