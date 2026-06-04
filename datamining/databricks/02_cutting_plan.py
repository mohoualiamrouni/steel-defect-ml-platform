"""Databricks job (Subject 3) — per-coil cutting plan (postprocessing of the inspection output).

Second task of the datamining postprocessing Job: for each coil, read its local
defects (with the severity written back by ``01_severity_score``), solve the 2D
guillotine cutting plan, and persist ``steeldefect.cutting_plans`` (one row/coil,
carrying the area-weighted grade fractions = the H3 evidence) and
``steeldefect.cutting_plan_pieces`` (one row/piece).

Spark injected by Databricks.   run:  main(spark, since="2026-05-01")
"""

from __future__ import annotations

import datetime as _dt
import os
import sys

OBT_TABLE = "steeldefect.obt_segmentations"
PLANS_TABLE = "steeldefect.cutting_plans"
PIECES_TABLE = "steeldefect.cutting_plan_pieces"


def _bootstrap_path() -> None:
    root = os.environ.get("STEELDEFECT_WORKSPACE_ROOT", "/Workspace/Repos/steel-defect-ml-platform")
    if root not in sys.path:
        sys.path.insert(0, root)


def _obt_to_cutting_df(pdf, severity_col: str = "severity"):
    """Rename OBT position columns to what the optimiser expects (x, y)."""
    out = pdf.rename(columns={"pos_x_m": "x", "pos_y_m": "y"}).copy()
    if severity_col != "severity" and severity_col in out.columns:
        out["severity"] = out[severity_col]
    return out


def _plan_to_rows(plan, coil_id, plan_id, model_uri):
    """Convert a CuttingPlan into (plan_row, [piece_rows]) for Delta."""
    fr = plan.grade_area_fractions()
    total_area = sum(plan.area_by_grade.values()) or 1.0
    plan_row = {
        "plan_id": plan_id, "coil_id": coil_id, "n_pieces": plan.n_pieces,
        "total_value": float(plan.total_value),
        "grade_A_frac": float(fr.get("A", 0.0)),
        "grade_B_frac": float(fr.get("B", 0.0)),
        "grade_C_frac": float(fr.get("C", 0.0)),
        "scrap_frac": float(fr.get("scrap", 0.0)),
        "yield_frac": float(plan.yield_pct / 100.0),
        "n_uncertain_pieces": int(sum(getattr(p, "grade_uncertain", False)
                                      for p in plan.pieces)),
        "cqr_coverage": float("nan"),
        "solver": "guillotine_dp_v1", "model_uri": model_uri,
        "created_ts": _dt.datetime.utcnow().isoformat(),
    }
    piece_rows = [{
        "plan_id": plan_id, "coil_id": coil_id, "piece_idx": i,
        "x0_m": float(p.x_start), "x1_m": float(p.x_end),
        "y0_m": float(p.y_start), "y1_m": float(p.y_end),
        "area_m2": float(p.area), "grade": p.grade_name,
        "piece_value": float(p.value), "mean_severity": float(p.max_severity),
        "sev_lo": float(getattr(p, "severity_lo", 0.0)),
        "sev_hi": float(getattr(p, "severity_hi", 0.0)),
        "grade_uncertain": bool(getattr(p, "grade_uncertain", False)),
    } for i, p in enumerate(plan.pieces)]
    return plan_row, piece_rows


def main(spark, since: str = "2026-01-01", solver: str = "guillotine_dp_v1",
         model_uri: str = "models:/steeldefect_severity_xgb/Production") -> int:
    _bootstrap_path()
    from datamining.config import DataminingConfig
    from datamining.cutting.constraints import CuttingConstraints
    from datamining.cutting.guillotine_optimizer import compute_cutting_plan
    from datamining.quality.quality_profile import build_quality_heatmap

    cfg = DataminingConfig()
    constraints = CuttingConstraints(
        min_piece_length=cfg.min_piece_length, min_piece_width=cfg.min_piece_width,
        quality_thresholds=cfg.quality_thresholds, grade_values=cfg.grade_values,
        grade_names=cfg.grade_names,
    )
    coils = [r["coil_id"] for r in
             spark.table(OBT_TABLE).where(f"defect_kind='local' AND date_key >= '{since}'")
             .select("coil_id").distinct().collect()]

    plan_rows, piece_rows = [], []
    for coil_id in coils:
        pdf = (spark.table(OBT_TABLE)
               .where(f"defect_kind='local' AND coil_id='{coil_id}'")
               .select("pos_x_m", "pos_y_m", "severity", "coil_length_m", "coil_width_m")
               .toPandas())
        if pdf.empty:
            continue
        cdf = _obt_to_cutting_df(pdf)
        L = float(pdf["coil_length_m"].iloc[0]); W = float(pdf["coil_width_m"].iloc[0])
        plan = compute_cutting_plan(cdf, constraints, coil_length=L, coil_width=W,
                                    n_length_bins=cfg.quality_grid_shape[0],
                                    n_width_bins=cfg.quality_grid_shape[1])
        pr, prs = _plan_to_rows(plan, coil_id, f"{coil_id}_{since}", model_uri)
        plan_rows.append(pr); piece_rows.extend(prs)

    if plan_rows:
        spark.createDataFrame(plan_rows).write.format("delta").mode("append").saveAsTable(PLANS_TABLE)
        spark.createDataFrame(piece_rows).write.format("delta").mode("append").saveAsTable(PIECES_TABLE)
    print(f"[cutting_plan] wrote {len(plan_rows)} plans / {len(piece_rows)} pieces")
    return len(plan_rows)
