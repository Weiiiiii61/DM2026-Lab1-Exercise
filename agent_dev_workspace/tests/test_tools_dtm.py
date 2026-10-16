import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    summary = build_dtm_tool.invoke({})

    assert summary["n_documents"] == 8
    assert summary["n_terms"] == 5
    assert summary["non_zero"] == 21
    assert summary["total_elements"] == 40
    assert summary["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool_known_answer_14():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})
    res = term_frequency_tool.invoke({})

    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }

    assert res["term_frequencies"] == expected_frequencies
    assert res["n_terms"] == 5
    assert res["total_term_count"] == 27
    assert "result_id" in res


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # Known Answer #21 from TEST_FIXTURE.md:
    # A call with n_terms=2, n_documents=3 should return the top-left 3x2 corner:
    # [[3, 1], [2, 1], [1, 1]]
    res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert res["n_terms"] == 2
    assert res["n_documents"] == 3
    assert res["terms"] == ["alpha", "always"]
    assert res["matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert session.pending_figure is not None
    assert "result_id" in res
