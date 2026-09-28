"""Pre-download and index the current 3-quarter TCS document set."""
from pathlib import Path
import sys

# Make `python scripts/ingest.py` work from the repository root on Windows/macOS/Linux.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.financial_data_extractor import extract_quarter
from app.tools.qualitative_analysis import _build_index


QUARTERS = ["Q1_FY27", "Q4_FY26", "Q3_FY26"]


def main() -> None:
    for q in QUARTERS:
        print(extract_quarter(q).model_dump_json(indent=2))
    _build_index(QUARTERS)
    print("\nRAG index ready.")


if __name__ == "__main__":
    main()
