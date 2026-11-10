"""ComfyUI-ReCaVSR: ReCaVSR one-step streaming diffusion video super-resolution (kopperx/ReCaVSR) as ComfyUI nodes."""
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import ReCaVSRLoader, ReCaVSRUpscale


class ReCaVSRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ReCaVSRLoader, ReCaVSRUpscale]


async def comfy_entrypoint() -> ReCaVSRExtension:
    return ReCaVSRExtension()
