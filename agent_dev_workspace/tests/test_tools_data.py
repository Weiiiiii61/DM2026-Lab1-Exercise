import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]
    assert "result_id" in res


def test_tokenize_tool_known_answer_10():
    session = SessionState()
    
    # Load premade load_dataset_tool to load sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Initialize tools_data tools on the same session
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})

    # Check session state updates
    assert "unigrams" in session.dataframe.columns
    assert len(session.dataframe["unigrams"]) == 8

    unigrams = session.dataframe["unigrams"].tolist()

    # Known Answer #10 from TEST_FIXTURE.md
    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    assert unigrams == expected_unigrams
    assert res["n_documents"] == 8
    assert res["total_tokens"] == sum(len(u) for u in expected_unigrams)


def test_check_missing_tool_known_answer_2():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Initialize tools_data tools on the same session
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})

    # Known Answer #2 from TEST_FIXTURE.md:
    # Exactly 1 row has missing/empty text (the last row, row 7).
    assert res["missing_count"] == 1
    assert res["total_rows"] == 8
    assert res["missing_indices"] == [7]
    assert "result_id" in res


def test_check_duplicates_tool_known_answer_3():
    # Test checking without dropping
    session1 = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools1 = make_tools(session1)
    check_duplicates_tool1 = next(t for t in tools1 if t.name == "check_duplicates_tool")

    res_check = check_duplicates_tool1.invoke({"drop": False})

    # Known Answer #3: Exactly 1 duplicate copy (row 6 is copy of row 5)
    assert res_check["duplicate_count"] == 1
    assert res_check["duplicate_indices"] == [6]
    assert res_check["remaining_rows"] == 8

    # Test checking with drop=True
    session2 = SessionState()
    premade_tools2, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(t for t in premade_tools2 if t.name == "load_dataset_tool")

    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools2 = make_tools(session2)
    check_duplicates_tool2 = next(t for t in tools2 if t.name == "check_duplicates_tool")

    res_drop = check_duplicates_tool2.invoke({"drop": True})

    # Known Answer #3: Dropping with keep=False leaves exactly 6 rows (rows 5 and 6 are both removed)
    assert res_drop["duplicate_count"] == 1
    assert res_drop["dropped"] is True
    assert res_drop["remaining_rows"] == 6
    assert len(session2.dataframe) == 6


def test_sample_data_tool_known_answer_13():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # Known Answer #13 from TEST_FIXTURE.md:
    # Call with n=4, random_state=42: returns exactly rows [1, 5, 0, 7]
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["n_samples"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert "result_id" in res


def test_describe_data_tool_known_answer_18():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})

    # Known Answer #18 from TEST_FIXTURE.md:
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)
    assert overall["min"] == 0
    assert overall["max"] == 30

    catA = res["per_category_stats"]["catA"]
    assert catA["count"] == 4
    assert np.isclose(catA["mean"], 17.25, atol=1e-3)

    catB = res["per_category_stats"]["catB"]
    assert catB["count"] == 4
    assert np.isclose(catB["mean"], 22.5, atol=1e-3)

    assert session.pending_figure is not None


def test_inspect_data_tool_custom_design():
    # Since inspect_data_tool has no predefined spec or known answer in TEST_FIXTURE.md,
    # we test our designed Option C behavior (head, tail, and random) against sample_fixture.csv:
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # Expected Answer 1: Peeking head 3 rows
    res_head = inspect_data_tool.invoke({"n": 3, "position": "head"})
    assert res_head["position"] == "head"
    assert res_head["n_rows"] == 3
    assert res_head["shape"] == [8, 3]  # 8 rows, 3 columns: text, category, category_name
    assert len(res_head["rows"]) == 3
    assert res_head["rows"][0]["text"] == "always alpha alpha alpha beta"

    # Expected Answer 2: Peeking tail 2 rows
    res_tail = inspect_data_tool.invoke({"n": 2, "position": "tail"})
    assert res_tail["position"] == "tail"
    assert res_tail["n_rows"] == 2
    assert len(res_tail["rows"]) == 2
    # Last row in sample_fixture.csv is empty text with label catA
    assert res_tail["rows"][1]["text"] == ""

    # Expected Answer 3: Peeking random 4 rows
    res_random = inspect_data_tool.invoke({"n": 4, "position": "random"})
    assert res_random["position"] == "random"
    assert res_random["n_rows"] == 4
    assert len(res_random["rows"]) == 4
    assert "result_id" in res_random
