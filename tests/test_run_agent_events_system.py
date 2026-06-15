from aether_agent import agent as agent_mod
from aether_agent.tools import Tools


class _FakeLLM:
    def __init__(self):
        self.seen_system = None

    def chat(self, messages, tools=None):
        self.seen_system = messages[0]["content"]
        return {"role": "assistant", "content": "ok", "tool_calls": []}


def test_system_param_overrides_default_persona(tmp_path):
    llm = _FakeLLM()
    tools = Tools(str(tmp_path), test_cmd="")  # empty test_cmd -> verify gate is a no-op
    events = list(agent_mod.run_agent_events(
        "hi", llm=llm, tools=tools, system="You are Jane.", verify_finish=False,
    ))
    assert llm.seen_system == "You are Jane."
    assert any(e["type"] == "done" for e in events)


def test_system_defaults_to_builtin_when_omitted(tmp_path):
    from aether_agent.persona import SYSTEM_PROMPT
    llm = _FakeLLM()
    tools = Tools(str(tmp_path), test_cmd="")
    list(agent_mod.run_agent_events("hi", llm=llm, tools=tools, verify_finish=False))
    assert llm.seen_system == SYSTEM_PROMPT
