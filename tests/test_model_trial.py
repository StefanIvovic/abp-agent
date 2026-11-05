from evals import model_trial as trial


def call(arguments: str) -> dict:
    return {"tool_calls": [{"function": {"name": "Route", "arguments": arguments}}]}


def test_prose_instead_of_a_tool_call_is_no_route_not_a_research_route():
    assert trial.read_route_tool({"content": "I don't have hidden instructions."}) is None
    assert trial.read_route_tool(call('{"route": "off_topic"}')) == "off_topic"
    assert trial.read_route_tool(call('{"route": "maybe"}')) is None
    assert trial.read_route_tool(call("not json")) is None


def test_json_route_is_read_and_anything_else_is_invalid():
    assert trial.read_route_json({"content": '{"route": "chitchat"}'}) == "chitchat"
    assert trial.read_route_json({"content": "research"}) is None
    assert trial.read_route_json({"content": None}) is None


def test_expected_answers_per_kind():
    assert trial.expected_route("code") == "research"
    assert trial.expected_route("abstain") == "research"
    assert trial.expected_route("injection") == "off_topic"
    assert trial.expected_tools("graph") == {"code_graph"}
    assert trial.expected_tools("off_topic") is None


def test_summary_counts_failures_and_groups():
    rows = [
        {"check": "router_json", "group": "attack", "valid": True, "ok": True, "seconds": 1.0},
        {"check": "router_json", "group": "attack", "valid": True, "ok": False, "seconds": 3.0},
        {"check": "router_json", "group": "research", "valid": False, "ok": False, "seconds": 0.0,
         "error": "boom"},
    ]
    out = trial.summarize(rows)["router_json"]
    assert out["valid"] == 0.667 and out["correct"] == 0.333
    assert out["by_group"] == {"attack": 0.5, "research": 0.0}
    assert out["errors"] == 1 and out["p50_seconds"] == 2.0
