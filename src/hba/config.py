from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEYSTROKE_CSV = ROOT / "data" / "raw" / "DSL-StrongPasswordData.csv"
RBA_CSV = ROOT / "data" / "raw" / "rba_dataset_cleaned.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SEED = 42
