"""
This module contains a small helper functions to parallelize the execution of
universes in a multiverse analysis.
"""

import contextlib

import joblib
from rich.progress import Progress, TaskID


@contextlib.contextmanager
def rich_joblib(progress: Progress, task_id: TaskID):
    """Context manager to patch joblib to report into rich progress bar.

    Args:
        progress: The rich Progress instance to report to
        task_id: The ID of the task to update

    Yields:
        None
    """

    class RichBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            progress.update(task_id, advance=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = RichBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
