import pytest

from config.scenarios.scenario_generator import ScenarioManager
from models.PoA.PoA_optimization import PoAOptimization
from models.PoA.PoA_tightening.compute_dual_big_m import DualBigMComputer


CASE = "base_test_case"


@pytest.fixture(scope="module")
def dual_big_m_stage():
    scenario_manager = ScenarioManager(CASE)
    scenarios = scenario_manager.create_scenario_set_from_ambiguity_set(
        ambiguity_set=CASE,
        n_scenarios=1,
        seed=7,
    )
    ambiguity_set_config = PoAOptimization.load_ambiguity_set(
        config_path="config/ambiguity_set_config.yaml",
        config_name=CASE,
    )
    stage = DualBigMComputer(
        scenarios_df=scenarios["scenarios_df"],
        costs_df=scenarios["costs_df"],
        ramps_df=scenarios["ramps_df"],
        p_init=None,
        num_time_steps=2,
        ambiguity_set_config=ambiguity_set_config,
        nn_model_dir=None,
        nn_normalization_stats_path=None,
        nn_policy_generators=[],
        reference_case=CASE,
    )
    alpha_report = stage.poa.build_default_alpha_bounds_report()
    stage.poa.alpha_bounds = stage._parse_alpha_bounds(alpha_report)
    return stage


def test_dual_big_m_auxiliary_models_do_not_add_strong_duality(dual_big_m_stage):
    eq_model = dual_big_m_stage._build_side_kkt_model_for_dual_big_m(
        side="eq",
        alpha_bounds=dual_big_m_stage.poa.alpha_bounds,
        fixed_binaries={},
    )
    assert not hasattr(eq_model, "strong_duality_eq")

    opt_model = dual_big_m_stage._build_side_kkt_model_for_dual_big_m(
        side="opt",
        alpha_bounds=dual_big_m_stage.poa.alpha_bounds,
        fixed_binaries={},
    )
    assert not hasattr(opt_model, "strong_duality_opt")


def test_dual_big_m_all_fixed_binaries_skip_programs(dual_big_m_stage):
    fixed_binaries = {}
    for side in ("eq", "opt"):
        for i, b in dual_big_m_stage.generator_block_pairs:
            for t in range(dual_big_m_stage.num_time_steps):
                for constraint_type in ("upper", "lower"):
                    fixed_binaries.setdefault(
                        dual_big_m_stage._binary_name(side, constraint_type),
                        {},
                    )[dual_big_m_stage._json_key((i, b, t))] = {"fixed_value": 0}
        for i in range(dual_big_m_stage.num_physical_generators):
            for t in range(dual_big_m_stage.num_time_steps):
                for constraint_type in ("ramp_up", "ramp_down"):
                    fixed_binaries.setdefault(
                        dual_big_m_stage._binary_name(side, constraint_type),
                        {},
                    )[dual_big_m_stage._json_key((i, t))] = {"fixed_value": 0}

    report = dual_big_m_stage.run_dual_big_m_tightening(
        alpha_bounds=dual_big_m_stage.poa.alpha_bounds,
        fixed_binaries=fixed_binaries,
        parallel_workers=1,
    )

    assert "aggregate_dual_bounds" not in report
    assert "lambda_bounds" not in report
    entries = [
        entry
        for component_entries in report["tight_big_m"].values()
        for entry in component_entries.values()
    ]
    assert entries
    assert all(entry["tight_big_m"] == 0.0 for entry in entries)
    assert all(entry["fixed_by_slack"] is True for entry in entries)
    assert all(entry["certified"] is True for entry in entries)
    assert all(entry["cap_limited"] is False for entry in entries)


def test_dual_big_m_cap_hits_are_reported_uncertified(dual_big_m_stage):
    model = dual_big_m_stage._build_side_kkt_model_for_dual_big_m(
        side="eq",
        alpha_bounds=dual_big_m_stage.poa.alpha_bounds,
        fixed_binaries={},
    )
    model.lambda_eq[0].set_value(dual_big_m_stage.default_lambda_upper)

    diagnostics = dual_big_m_stage._dual_big_m_diagnostics(
        m=model,
        side="eq",
        constraint_type="upper",
        dual_value=dual_big_m_stage.default_dual_big_m,
        solved=True,
    )

    assert diagnostics["certified"] is False
    assert diagnostics["cap_limited"] is True
    assert "default dual Big-M cap" in diagnostics["cap_limit_reason"]


def test_reports_with_aggregate_dual_bounds_are_rejected(dual_big_m_stage):
    with pytest.raises(ValueError, match="deprecated aggregate_dual_bounds"):
        dual_big_m_stage.poa.load_tightening_report_data(
            {"aggregate_dual_bounds": {}},
            prepare_bounds=False,
        )
