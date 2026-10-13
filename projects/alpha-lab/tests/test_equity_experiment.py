import json
from pathlib import Path

import pytest
pytest.importorskip("numpy")
pytest.importorskip("pandas")
from alpha_lab.equity_experiment import export_experiments, run_experiment
from alpha_lab.experiment_db import ExperimentDB


def test_export_uses_fixed_scripts_and_safe_paths(tmp_path):
    panel=tmp_path/"panel";panel.mkdir()
    (panel/"panel.json").write_text('{}')
    run=tmp_path/"run";run.mkdir()
    spec={"name":"../model title; not a path", "factors":{"roa":1}}
    (run/"candidates.json").write_text(json.dumps([spec]))
    workspace=tmp_path/"jobs"
    items=export_experiments(panel,[run],workspace)
    directory=Path(items[0]["directory"])
    assert directory.parent==workspace/"experiments"
    assert directory.name.startswith("candidate_")
    assert "../model" not in (directory/"run_experiment.py").read_text()
    assert json.loads((directory/"experiment.json").read_text())["holdout_access"] is False
    assert ExperimentDB(str(workspace/"experiments.db")).get(items[0]["id"]).status=="checked"
    with pytest.raises(ValueError,match="new"):
        export_experiments(panel,[run],workspace)


def test_worker_cannot_unlock_holdout(tmp_path):
    config=tmp_path/"experiment.json"
    config.write_text(json.dumps({"holdout_access":True}))
    with pytest.raises(ValueError,match="no holdout"):
        run_experiment(config)
