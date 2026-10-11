import json

from scripts.export_hosted_schema import NAMES, SCHEMA_PATH, render_schema


def test_hosted_schema_export_is_byte_reproducible_and_complete():
    checked_in = SCHEMA_PATH.read_text(encoding="utf-8")
    assert checked_in == render_schema()
    assert list(json.loads(checked_in)) == NAMES
