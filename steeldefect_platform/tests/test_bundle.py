"""The bundle's dispatched tasks name real tasks, with parameters their entrypoints accept."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import yaml

from steeldefect_platform import run_task

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "databricks.yml"


def _bundle() -> dict:
    return yaml.safe_load(BUNDLE.read_text(encoding="utf-8"))


def _dispatched():
    """(job name, task) for every task that runs steeldefect_platform/run_task.py."""
    for job_name, job in _bundle()["resources"]["jobs"].items():
        for task in job["tasks"]:
            spt = task.get("spark_python_task")
            if spt and spt["python_file"] == "steeldefect_platform/run_task.py":
                yield job_name, task


def _filled(argument: str) -> str:
    """A task argument with its job parameters and bundle variables filled in: Databricks
    splices their text into the argument."""
    argument = re.sub(r"\{\{job\.parameters\.\w+\}\}", "7", argument)
    return re.sub(r"\$\{var\.\w+\}", "x", argument)


def test_every_dispatched_task_names_its_task_and_passes_parameters_it_accepts():
    names = set()
    for job_name, task in _dispatched():
        argv = [_filled(argument) for argument in task["spark_python_task"]["parameters"]]
        name, kwargs = run_task.parse(argv)              # as the dispatcher reads them
        assert name == task["task_key"], job_name
        inspect.signature(run_task.TASKS[name]).bind(**kwargs)
        names.add(name)
    assert "ab_decision" in names


def test_the_ab_decision_job_runs_the_gate_one_run_at_a_time():
    # The bundle's A/B task used to run nb_ab.py, which registers two versions and decides
    # nothing; that notebook is now a registry demo, under a name that says so.
    bundle = _bundle()
    jobs = bundle["resources"]["jobs"]
    job = jobs["steeldefect_ab_decision"]
    assert job["max_concurrent_runs"] == 1
    assert {p["name"]: p["default"] for p in job["parameters"]} == {
        "eval_dataset_id": "", "model_a_version": "", "model_b_version": "", "promote": "true"}
    # Each value is one argument of its own: spliced into a JSON template, a typed id
    # could close its string and add keys.
    [task] = job["tasks"]
    expected = ["ab_decision"]
    for name in [p["name"] for p in job["parameters"]] + ["model_name", "registry_uri"]:
        value = {"model_name": "${var.segmenter_model}",
                 "registry_uri": "databricks-uc"}.get(name, f"{{{{job.parameters.{name}}}}}")
        expected += ["--param", f"{name}={value}"]
    assert task["spark_python_task"]["parameters"] == expected
    # The decision promotes a Unity Catalog model, named in full; the registry demo writes
    # @champion of a model of its own.
    model = bundle["variables"]["segmenter_model"]["default"]
    assert model == "main.steeldefect.steeldefect_segmenter"
    demo = jobs["steeldefect_registry_demo"]["tasks"]
    assert [(t["task_key"], t["notebook_task"]["notebook_path"]) for t in demo] == [
        ("register_versions", "./nb_ab.py")]
    [demo_model] = re.findall(r'^MODEL = "([^"]+)"$',
                              (ROOT / "nb_ab.py").read_text(encoding="utf-8"), flags=re.M)
    assert demo_model == "main.steeldefect.steeldefect_segmenter_registry_demo" != model
