import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_known_answer_16():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known Answer 16 tests:
    # 0 vs 1 (both catA, near-identical text): 0.9847
    res_0_1 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert np.isclose(res_0_1["similarity"], 0.9847, atol=1e-3)

    # 0 vs 3 (catA vs catB, share only "always"): 0.0909
    res_0_3 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert np.isclose(res_0_3["similarity"], 0.0909, atol=1e-3)

    # 3 vs 4 (both catB, near-identical text): 0.9847
    res_3_4 = cosine_sim_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert np.isclose(res_3_4["similarity"], 0.9847, atol=1e-3)

    # 0 vs 7 (catA vs empty-text row): 0.0
    res_0_7 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert np.isclose(res_0_7["similarity"], 0.0, atol=1e-3)


def test_feature_correlation_matrix_known_answer_19():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({})

    # Known Answer 19: Top 5 terms by variance descending: alpha, gamma, delta, beta, always
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["top_terms"] == expected_terms

    expected_corr = np.array([
        [1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718, 1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882, 0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ])

    actual_corr = np.array(res["correlation_matrix"])
    assert np.allclose(actual_corr, expected_corr, atol=1e-3)
    assert session.pending_figure is not None
