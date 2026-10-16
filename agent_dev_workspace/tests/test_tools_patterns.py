import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_mine_patterns_tool_fpgrowth_known_answer_17():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Known Answer #17 from TEST_FIXTURE.md:
    # filtering_method="variance" and "term_frequency" keep 0 terms for both categories
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["patterns_found_count"] == 0
    assert res_var["patterns"] == []

    # filtering_method="tfidf", algorithm="fpgrowth":
    # catA -> keeps alpha only -> pattern {alpha} support 3
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA["patterns_found_count"] == 1
    assert res_catA["patterns"][0]["pattern"] == "alpha"
    assert res_catA["patterns"][0]["support"] == 3

    # catB -> keeps gamma only -> pattern {gamma} support 4
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catB["patterns_found_count"] == 1
    assert res_catB["patterns"][0]["pattern"] == "gamma"
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_tool_topk_maxfpgrowth_known_answer_20():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Known Answer #20 from TEST_FIXTURE.md:
    # algorithm="topk" (k=1): catA -> {alpha} support 3, catB -> {gamma} support 4
    res_topk_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_a["patterns_found_count"] == 1
    assert res_topk_a["patterns"][0]["pattern"] == "alpha"
    assert res_topk_a["patterns"][0]["support"] == 3

    res_topk_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_b["patterns_found_count"] == 1
    assert res_topk_b["patterns"][0]["pattern"] == "gamma"
    assert res_topk_b["patterns"][0]["support"] == 4

    # algorithm="maxfpgrowth": returns 0 patterns for both categories (expected PAMI behavior)
    res_max_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_a["patterns_found_count"] == 0
    assert res_max_a["patterns"] == []

    res_max_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_b["patterns_found_count"] == 0
    assert res_max_b["patterns"] == []
