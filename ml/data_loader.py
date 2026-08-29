"""
Data loading and validation for the house price pipeline.

Responsible for reading the raw Kaggle Ames Housing CSV, sanity-checking
it, and handing back a clean DataFrame plus a small data-quality report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ml.config import ID_COLUMN, RAW_TRAIN_PATH, TARGET_COLUMN

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the raw data fails basic sanity checks."""


@dataclass
class DataQualityReport:
    n_rows: int
    n_columns: int
    missing_pct_by_column: dict = field(default_factory=dict)
    dtypes: dict = field(default_factory=dict)
    target_present: bool = True

    def top_missing(self, n: int = 10) -> dict:
        """Return the n columns with the highest missing percentage."""
        return dict(
            sorted(self.missing_pct_by_column.items(), key=lambda kv: kv[1], reverse=True)[:n]
        )

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "target_present": self.target_present,
            "top_missing_columns": self.top_missing(),
        }


def load_raw_data(path: str | Path | None = None, require_target: bool = True) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Load the raw CSV, run basic validation, and return (dataframe, report).

    Parameters
    ----------
    path: Optional override path. Defaults to config.RAW_TRAIN_PATH.
    require_target: If True, raises when SalePrice is missing (i.e. this
        is meant to be training data, not inference input).
    """
    csv_path = Path(path) if path else RAW_TRAIN_PATH

    if not csv_path.exists():
        raise DataValidationError(
            f"Could not find data file at {csv_path}. "
            f"Download the Kaggle Ames Housing dataset and place train.csv there."
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise DataValidationError(f"{csv_path} loaded but contains zero rows.")

    if ID_COLUMN not in df.columns:
        logger.warning("Expected an '%s' column but didn't find one.", ID_COLUMN)

    target_present = TARGET_COLUMN in df.columns
    if require_target and not target_present:
        raise DataValidationError(
            f"'{TARGET_COLUMN}' column not found in {csv_path}. "
            f"Is this the training set?"
        )

    missing_pct = (df.isna().mean() * 100).round(2).to_dict()
    dtypes = df.dtypes.astype(str).to_dict()

    report = DataQualityReport(
        n_rows=len(df),
        n_columns=len(df.columns),
        missing_pct_by_column=missing_pct,
        dtypes=dtypes,
        target_present=target_present,
    )

    logger.info(
        "Loaded %d rows, %d columns from %s (target_present=%s)",
        report.n_rows, report.n_columns, csv_path, target_present,
    )

    return df, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    frame, rpt = load_raw_data()
    print(f"Shape: {frame.shape}")
    print(f"Top missing columns: {rpt.top_missing()}")
