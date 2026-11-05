"""IO for the competition's index file, submission file and video metadata."""

from highlight.io.index import (
    IndexEntry,
    SubmissionRow,
    load_index,
    read_submission,
    write_submission,
)

__all__ = [
    "IndexEntry",
    "SubmissionRow",
    "load_index",
    "read_submission",
    "write_submission",
]
