"""Databricks job scripts for Subject 1 (Active Learning + A/B).

These modules have numeric-prefixed filenames (``01_al_pool_build.py``,
``02_ab_decision.py``) matching the flatness-clustering deployment-pattern
pilot.  Numeric-prefixed names are not importable as ``module.name``; in
Databricks they are referenced directly as Job task notebook/script paths, and
the platform orchestrator (:mod:`steeldefect_platform.jobs`) loads them by file path.
"""
