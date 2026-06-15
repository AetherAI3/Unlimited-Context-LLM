# tests/test_agent_runner_lock.py
from aether_agent import agent_runner


class _RecordingLock:
    def __init__(self):
        self.acquired = 0
    def __enter__(self):
        self.acquired += 1
        return self
    def __exit__(self, *a):
        return False


class _FakeInner:
    test_cmd = ""
    def execute(self, name, args):
        return f"[ran {name}]"


def test_destructive_acquires_lock_read_does_not():
    lock = _RecordingLock()
    pt = agent_runner._PolicyTools(
        _FakeInner(), allowed={"write_file", "read_file"}, permission="skip",
        confirm=lambda n, a: True, write_lock=lock,
    )
    pt.execute("read_file", {"path": "x"})
    assert lock.acquired == 0          # reads are not locked
    pt.execute("write_file", {"path": "x", "content": "y"})
    assert lock.acquired == 1          # destructive acquired the lock


def test_no_lock_means_no_locking():
    pt = agent_runner._PolicyTools(
        _FakeInner(), allowed={"write_file"}, permission="skip", confirm=lambda n, a: True,
    )
    assert "[ran write_file]" in pt.execute("write_file", {"path": "x", "content": "y"})
