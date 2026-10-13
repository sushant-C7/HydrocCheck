import json
from types import SimpleNamespace

import sys

from src.agent import run_rag_research_assistant
from src.rag import search_knowledge


class FakeToolCall:
    def __init__(self, name, arguments, call_id="call_1"):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeAssistantMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=True):
        calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in self.tool_calls
        ]
        result = {"role": "assistant", "tool_calls": calls}
        if self.content is not None:
            result["content"] = self.content
        return result


class FakeCompletions:
    def __init__(self):
        self.count = 0

    def create(self, **kwargs):
        self.count += 1
        if self.count == 1:
            message = FakeAssistantMessage(
                tool_calls=[FakeToolCall(
                    "search_water_quality_knowledge",
                    json.dumps({"query": "BIS IS 10500 India drinking water standard"}),
                )]
            )
        else:
            message = FakeAssistantMessage(
                content="BIS IS 10500 is the Indian drinking-water specification. Verify the current official text."
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    def __init__(self, api_key=None):
        completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=completions)


def test_research_assistant_uses_rag_tool_and_returns_sources(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    result = run_rag_research_assistant(
        "What is BIS IS 10500 in India?",
        search_knowledge,
    )

    assert "BIS IS 10500" in result["reply"]
    assert result["tool_trace"][0]["tool"] == "search_water_quality_knowledge"
    assert result["tool_trace"][0]["status"] == "completed"
    assert any("BIS IS 10500" in item["title"] for item in result["sources"])
