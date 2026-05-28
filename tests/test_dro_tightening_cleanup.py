import json
from types import SimpleNamespace

from config.scenarios.scenario_generator import ScenarioManager
from models.DRO_PoA.DRO_PoA_optimization import DRO_PoAOptimization
from models.DRO_PoA.DRO_PoA_tightening import compute_dual_big_m
from models.DRO_PoA.DRO_PoA_tightening.compute_dual_big_m import DRODualBigMComputer
from models.DRO_PoA.DRO_PoA_tightening.tightening_main import DROPoATighteningMain


REMOVED_REPORT_KEYS = {
    "scenario_lambda_bounds",
    "regime_lambda_bounds",
    "lambda_bounds",
    "aggregate_dual_bounds",
}
AGGREGATE_CONSTRAINT_NAMES = {
    "aggregate_mu_max_bound",
    "aggregate_mu_min_bound",
    "aggregate_mu_ramp_up_bound",
    "aggregate_mu_ramp_down_bound",
}
DUAL_NAMES = (
    "mu_upper_eq",
    "mu_lower_eq",
    "mu_ramp_up_eq",
    "mu_ramp_down_eq",
    "mu_upper_opt",
    "mu_lower_opt",
    "mu_ramp_up_opt",
    "mu_ramp_down_opt",
)


def _fake_dual_computer() -> DRODualBigMComputer:
    computer = DRODualBigMComputer.__new__(DRODualBigMComputer)
    poa = SimpleNamespace(
        alpha_bounds={"0,0,0,0": {"lower": 0.0, "upper": 1.0}},
        fixed_binaries={},
        primal_big_m={"dummy": {}},
        tight_big_m={},
        reference_case="base_test_case",
        regime_set="PoA_analysis",
        regime_name="normal",
        eta=0.0,
        epsilon=0.0,
        num_time_steps=1,
        num_empirical_scenarios=1,
        num_physical_generators=1,
        generator_block_pairs=[(0, 0)],
        physical_generator_names=["G2"],
        block_names=["G2_B1"],
        nn_policy_generator_names=[],
        nn_model_dir=None,
        nn_normalization_stats_path=None,
        selected_regime_parameters={},
    )
    object.__setattr__(computer, "poa", poa)
    object.__setattr__(
        computer,
        "tightening_data",
        {
            "primal_big_m": poa.primal_big_m,
            "alpha_bounds": poa.alpha_bounds,
            "fixed_binaries": poa.fixed_binaries,
        },
    )
    object.__setattr__(computer, "stage_reports", {})
    return computer


def test_dro_dual_big_m_tightening_returns_only_componentwise_fields(monkeypatch) -> None:
    computer = _fake_dual_computer()

    def fake_solve(task):
        (
            side,
            constraint_type,
            scenario_index,
            regime_index,
            _alpha_bounds,
            _fixed_binaries,
            _solver_name,
            _time_limit,
            _tee,
            _solver_options,
        ) = task
        return {
            "side": side,
            "constraint_type": constraint_type,
            "dual_name": computer._dual_name(side, constraint_type),
            "scenario_index": tuple(scenario_index),
            "regime_index": tuple(regime_index),
            "scenario_key": computer._json_key(scenario_index),
            "regime_key": computer._json_key(regime_index),
            "tight_big_m": 1.5,
            "fixed_by_slack": False,
            "termination_condition": "optimal",
        }

    monkeypatch.setattr(compute_dual_big_m, "_solve_parallel_dual_big_m", fake_solve)

    report = computer.run_dual_big_m_tightening(parallel_workers=1)

    assert REMOVED_REPORT_KEYS.isdisjoint(report)
    assert set(report) == {"scenario_tight_big_m", "regime_tight_big_m", "tight_big_m"}
    assert set(report["scenario_tight_big_m"]) == set(DUAL_NAMES)


def test_dro_dual_big_m_stage_report_excludes_lambda_and_aggregate(monkeypatch) -> None:
    computer = _fake_dual_computer()
    dual_payload = {
        "scenario_tight_big_m": {"mu_upper_eq": {"0,0,0,0": {"tight_big_m": 1.0}}},
        "regime_tight_big_m": {"mu_upper_eq": {"0,0,0": {"tight_big_m": 1.0}}},
        "tight_big_m": {"mu_upper_eq": {"0,0,0,0": {"tight_big_m": 1.0}}},
    }
    captured = {}

    object.__setattr__(
        computer,
        "run_dual_big_m_tightening",
        lambda **_kwargs: dual_payload,
    )
    object.__setattr__(computer, "_metadata", lambda: {"model_type": "DRO_PoA"})

    def fake_save(stage_name, report, output_path):
        captured["stage_name"] = stage_name
        captured["report"] = report
        captured["output_path"] = output_path
        return report

    object.__setattr__(computer, "_save_stage_report", fake_save)

    report = computer.run_dual_big_m(output_path=None)

    assert captured["stage_name"] == "dual_big_m"
    assert REMOVED_REPORT_KEYS.isdisjoint(report)
    assert REMOVED_REPORT_KEYS.isdisjoint(computer.tightening_data)
    assert not hasattr(computer.poa, "lambda_bounds")
    assert not hasattr(computer.poa, "aggregate_dual_bounds")


def test_dro_final_tightening_report_excludes_lambda_and_aggregate(tmp_path) -> None:
    stage = DROPoATighteningMain.__new__(DROPoATighteningMain)
    object.__setattr__(
        stage,
        "tightening_data",
        {
            "scenario_lambda_bounds": {"legacy": {}},
            "regime_lambda_bounds": {"legacy": {}},
            "lambda_bounds": {"legacy": {}},
            "aggregate_dual_bounds": {"legacy": {}},
            "scenario_tight_big_m": {"mu_upper_eq": {}},
            "regime_tight_big_m": {"mu_upper_eq": {}},
            "tight_big_m": {"mu_upper_eq": {}},
        },
    )
    object.__setattr__(stage, "stage_reports", {})
    object.__setattr__(stage, "_metadata", lambda: {"model_type": "DRO_PoA"})

    output_path = stage.save_final_report(tmp_path / "final_tightening_report.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert REMOVED_REPORT_KEYS.isdisjoint(payload)
    assert {"scenario_tight_big_m", "regime_tight_big_m", "tight_big_m"}.issubset(payload)


def _small_dro_optimizer() -> DRO_PoAOptimization:
    manager = ScenarioManager(base_case_reference="base_test_case")
    manager.base_case["time_steps"] = 2
    scenario_set = manager.create_scenario_set_from_regimes(
        regime_config_path="config/regime_definitions.yaml",
        regime_set="PoA_analysis",
        seed=1,
    )
    return DRO_PoAOptimization(
        scenarios_df=scenario_set["scenarios_df"],
        costs_df=scenario_set["costs_df"],
        ramps_df=scenario_set["ramps_df"],
        num_time_steps=2,
        regime_config_path="config/regime_definitions.yaml",
        regime_set="PoA_analysis",
        regime_name="normal",
        reference_case="base_test_case",
    )


def _key(index) -> str:
    return ",".join(str(int(part)) for part in tuple(index))


def _componentwise_tightening_report(optimizer: DRO_PoAOptimization) -> dict:
    primal_big_m = {
        "block_capacity": {
            _key((i, b, t)): 100.0
            for i, b in optimizer.generator_block_pairs
            for t in range(optimizer.num_time_steps)
        },
        "physical_capacity": {
            _key((i,)): 100.0 for i in range(optimizer.num_physical_generators)
        },
        "ramp_up": {
            _key((i, t)): 100.0
            for i in range(optimizer.num_physical_generators)
            for t in range(optimizer.num_time_steps)
        },
        "ramp_down": {
            _key((i, t)): 100.0
            for i in range(optimizer.num_physical_generators)
            for t in range(optimizer.num_time_steps)
        },
        "ramp_up_initial": {
            _key((i,)): 100.0 for i in range(optimizer.num_physical_generators)
        },
        "ramp_down_initial": {
            _key((i,)): 100.0 for i in range(optimizer.num_physical_generators)
        },
    }
    scenario_tight_big_m = {dual_name: {} for dual_name in DUAL_NAMES}
    for k in range(optimizer.num_empirical_scenarios):
        for i, b in optimizer.generator_block_pairs:
            for t in range(optimizer.num_time_steps):
                for dual_name in (
                    "mu_upper_eq",
                    "mu_lower_eq",
                    "mu_upper_opt",
                    "mu_lower_opt",
                ):
                    scenario_tight_big_m[dual_name][_key((k, i, b, t))] = {
                        "tight_big_m": 7.0
                    }
        for i in range(optimizer.num_physical_generators):
            for t in range(optimizer.num_time_steps):
                for dual_name in (
                    "mu_ramp_up_eq",
                    "mu_ramp_down_eq",
                    "mu_ramp_up_opt",
                    "mu_ramp_down_opt",
                ):
                    scenario_tight_big_m[dual_name][_key((k, i, t))] = {
                        "tight_big_m": 5.0
                    }

    return {
        "metadata": {
            "model_type": "DRO_PoA",
            "tightening_type": "regime_wide",
            "tightening_scope": "scenario_wise",
            "regime_name": optimizer.regime_name,
            "num_time_steps": optimizer.num_time_steps,
        },
        "primal_big_m": primal_big_m,
        "scenario_tight_big_m": scenario_tight_big_m,
        "tight_big_m": scenario_tight_big_m,
    }


def test_dro_model_has_no_aggregate_constraints_and_applies_componentwise_dual_bounds() -> None:
    optimizer = _small_dro_optimizer()
    optimizer.build_model()

    assert all(
        not hasattr(optimizer.model, component_name)
        for component_name in AGGREGATE_CONSTRAINT_NAMES
    )

    report = _componentwise_tightening_report(optimizer)
    stats = optimizer.apply_regime_wide_tightening_to_model(
        report=report,
        apply_alpha_bounds=False,
        apply_fixed_binaries=False,
        apply_dual_bounds=True,
        apply_relu_bounds=False,
    )

    assert "lambda_bounds" not in stats
    assert "aggregate_dual_bounds" not in stats
    assert stats["dual_upper_bounds"] > 0
    assert all(
        not hasattr(optimizer.model, component_name)
        for component_name in AGGREGATE_CONSTRAINT_NAMES
    )

    for dual_name in ("mu_upper_eq", "mu_lower_eq", "mu_upper_opt", "mu_lower_opt"):
        dual_var = getattr(optimizer.model, dual_name)
        for k in optimizer.model.scenarios:
            for i, b in optimizer.model.generator_blocks:
                for t in optimizer.model.time_steps:
                    assert dual_var[int(k), int(i), int(b), int(t)].ub == 7.0

    for dual_name in (
        "mu_ramp_up_eq",
        "mu_ramp_down_eq",
        "mu_ramp_up_opt",
        "mu_ramp_down_opt",
    ):
        dual_var = getattr(optimizer.model, dual_name)
        for k in optimizer.model.scenarios:
            for i in optimizer.model.physical_generators:
                for t in optimizer.model.time_steps:
                    assert dual_var[int(k), int(i), int(t)].ub == 5.0
