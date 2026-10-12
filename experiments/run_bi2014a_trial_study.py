"""Run the BI2014a trial plan through the common programme owner."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research.trial_candidate_programme import main  # noqa: E402

if __name__ == "__main__":
    main()
