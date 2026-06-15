from aether_agent import agents_view


def _rows(*names):
    return [{"name": n, "model": "qwen2.5-coder:7b", "pool_gb": 5, "n_commands": 2} for n in names]


def test_empty():
    assert "no agents" in agents_view.render_agents_columns([]).lower()


def test_single_card_has_fields_and_active_marker():
    out = agents_view.render_agents_columns(_rows("jane"), active="jane")
    assert "jane" in out and "qwen2.5-coder:7b" in out
    assert "5 GB" in out and "2 cmds" in out
    assert "*" in out  # active marker


def test_multiple_cards_wrap_to_width():
    out = agents_view.render_agents_columns(_rows("a", "b", "c", "d"), width=50)
    lines = out.splitlines()
    assert any("a" in line and "b" in line for line in lines)  # a,b side by side
    assert any("c" in line and "d" in line for line in lines)  # c,d on the next block


def test_cp1252_safe():
    agents_view.render_agents_columns(_rows("jane", "neo"), active="neo").encode("cp1252")
