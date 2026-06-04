"""Databricks job scripts for Subject 3 (datamining + cutting plan).

The two-task postprocessing Job requested by the plant: ``01_severity_score.py`` reads
local defects from the OBT, predicts severity (+ CQR interval) and MERGEs it
back; ``02_cutting_plan.py`` solves the per-coil guillotine plan and persists
``steeldefect.cutting_plans`` / ``steeldefect.cutting_plan_pieces``.  Numeric-prefixed names
match the flatness pilot; loaded by path via :mod:`steeldefect_platform.jobs`.
"""
