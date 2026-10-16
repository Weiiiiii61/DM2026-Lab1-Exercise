import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
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

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["kept_count"] == 4
    assert res["filtered_count"] == 1
    assert "always" in res["filtered_features"]
    assert set(res["kept_features"]) == {"alpha", "gamma", "delta", "beta"}

    result_id = res["result_id"]
    stored_data = session.results_store[result_id]
    full_report = stored_data["full_report"]

    assert full_report.columns[0] == "feature"
    assert full_report.columns[1] == "variance"
    
    variance_dict = dict(zip(full_report["feature"], full_report["variance"]))
    assert np.isclose(variance_dict["alpha"], 1.1875, atol=1e-3)
    assert np.isclose(variance_dict["always"], 0.109375, atol=1e-3)
    assert np.isclose(variance_dict["beta"], 0.234375, atol=1e-3)
    assert np.isclose(variance_dict["delta"], 0.25, atol=1e-3)
    assert np.isclose(variance_dict["gamma"], 1.109375, atol=1e-3)


def test_pearson_filter_tool_known_answer_6():
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

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    result_id = res["result_id"]
    stored_data = session.results_store[result_id]
    full_report = stored_data["full_report"]

    assert full_report.columns[0] == "feature"
    assert full_report.columns[1] == "pearson_r"

    pearson_dict = dict(zip(full_report["feature"], full_report["pearson_r"]))

    assert np.isclose(pearson_dict["alpha"], -0.6882, atol=1e-3)
    assert np.isclose(pearson_dict["always"], 0.3780, atol=1e-3)
    assert np.isclose(pearson_dict["beta"], -0.7746, atol=1e-3)
    assert np.isclose(pearson_dict["delta"], 1.0000, atol=1e-3)
    assert np.isclose(pearson_dict["gamma"], 0.8307, atol=1e-3)

    assert set(res["kept_features"]) == {"delta", "gamma", "always"}
    assert set(res["filtered_features"]) == {"alpha", "beta"}


def test_spearman_filter_tool_known_answer_6():
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

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Known Answer #6 from TEST_FIXTURE.md (target_class="catB"):
    # Spearman r: alpha=-0.7500, always=0.3780, beta=-0.7746, delta=1.0000, gamma=0.9363
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    result_id = res["result_id"]
    stored_data = session.results_store[result_id]
    full_report = stored_data["full_report"]

    assert full_report.columns[0] == "feature"
    assert full_report.columns[1] == "spearman_r"

    spearman_dict = dict(zip(full_report["feature"], full_report["spearman_r"]))

    assert np.isclose(spearman_dict["alpha"], -0.7500, atol=1e-3)
    assert np.isclose(spearman_dict["always"], 0.3780, atol=1e-3)
    assert np.isclose(spearman_dict["beta"], -0.7746, atol=1e-3)
    assert np.isclose(spearman_dict["delta"], 1.0000, atol=1e-3)
    assert np.isclose(spearman_dict["gamma"], 0.9363, atol=1e-3)

    assert set(res["kept_features"]) == {"delta", "gamma", "always"}
    assert set(res["filtered_features"]) == {"alpha", "beta"}
