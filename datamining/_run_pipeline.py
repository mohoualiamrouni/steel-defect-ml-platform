"""One-shot pipeline runner with progress output."""
import sys
sys.stdout.reconfigure(line_buffering=True)

from datamining.pipeline.run_pipeline import run_pipeline
results = run_pipeline(n_coils=50)
print("SUMMARY:", results)
