import os
import uuid

from jupyter_client.manager import AsyncKernelManager
from jupyter_core.paths import jupyter_runtime_dir


class IPCKernelManager(AsyncKernelManager):
    def __init__(self, *args, **kwargs):
        kernel_id = str(uuid.uuid4())
        os.makedirs(jupyter_runtime_dir(), exist_ok=True)
        connection_file = os.path.join(
            jupyter_runtime_dir(), f"kernel-{kernel_id}.json"
        )
        super().__init__(
            *args,
            transport="ipc",
            kernel_id=kernel_id,
            connection_file=connection_file,
            **kwargs,
        )
