import logging
import os
import time

import torch
from tqdm import tqdm

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .recavsr import loader
from .recavsr.session import (DecoderStream, VideoStream, adain_color_fix, block_schedule, padded_size, prepare_block,
                              scaled_size, upscale_frames, wavelet_color_fix)

log = logging.getLogger("ReCaVSR")
FOLDER = "recavsr"
ACTIVATION_BYTES_PER_TOKEN = int(2.6 * 1024 * 1024)  # peak minus weights, measured at 416 and 1612 tokens
KV_BYTES_PER_TOKEN = 66 * 2 * 3072 * 2  # the router keeps 66 planes of bf16 keys and values over the 30 blocks
folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))


def _files():
    return [f for f in folder_paths.get_filename_list(FOLDER) if f.endswith(".safetensors")]


class ReCaVSRLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = _files()
        return io.Schema(
            node_id="ReCaVSRLoader",
            display_name="ReCaVSR Model Loader",
            category="ReCaVSR",
            description="Loads ReCaVSR (the Wan2.2-based one-step streaming video super-resolution transformer) with its "
                        "prompt embedding and FlashDecoder from models/recavsr.",
            inputs=[
                io.Combo.Input("transformer", options=files, default="transformer.safetensors" if "transformer.safetensors" in files else None,
                               tooltip="kopper/ReCaVSR transformer.safetensors (bf16), or the int8 ConvRot file made by "
                                       "tools/convert_int8.py (half the size, 1.5x faster, 45.5 dB against bf16; needs an NVIDIA "
                                       "GPU of compute capability 7.5 or newer). prompt.safetensors and flashdecoder.safetensors "
                                       "are read from the same folder."),
            ],
            outputs=[io.Custom("RECAVSR").Output(display_name="model")],
        )

    @classmethod
    def execute(cls, transformer) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise(FOLDER, transformer)
        folder = os.path.dirname(path)
        for name in ("prompt.safetensors", "flashdecoder.safetensors"):
            if not os.path.isfile(os.path.join(folder, name)):
                raise FileNotFoundError(f"{name} is missing next to {path}; download kopper/ReCaVSR into models/{FOLDER}")
        return io.NodeOutput(loader.load(path, os.path.join(folder, "flashdecoder.safetensors"), os.path.join(folder, "prompt.safetensors")))


class ReCaVSRUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ReCaVSRUpscale",
            display_name="ReCaVSR Upscale Video",
            category="ReCaVSR",
            description="One-step streaming video super-resolution: 21 frames, then 8 at a time, each block decoded as it "
                        "is generated.",
            inputs=[
                io.Custom("RECAVSR").Input("model"),
                io.Image.Input("images", tooltip="The video's frames, in order."),
                io.Float.Input("scale", default=4.0, min=0.5, max=8.0, step=0.25,
                               tooltip="Output size = input size x scale, each dimension rounded half up. The model was "
                                       "trained for 4x; 2x and 1.5x are the reference's other settings."),
                io.Combo.Input("color_fix", options=["none", "adain", "wavelet"], default="none",
                               tooltip="Per-frame colour match to the input: AdaIN matches mean and variance, wavelet keeps "
                                       "the input's low frequencies."),
                io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, tooltip="Seed of the one-step noise."),
                io.Combo.Input("decoder", options=["flashdecoder", "vae"], default="flashdecoder",
                               tooltip="flashdecoder: the LQ-conditioned FlashDecoder, streaming and fast. vae: the stock "
                                       "Wan 2.2 VAE connected below, decoded at the end."),
                io.Vae.Input("vae", optional=True, tooltip="The Wan 2.2 VAE, for decoder = vae."),
                io.Combo.Input("kernels", options=["fast", "exact"], default="fast",
                               tooltip="fast: comfy-kitchen's fused norm, rope and residual kernels and retuned attention "
                                       "tiles; 47 dB against exact, no visible difference. The decoder's window kernel needs "
                                       "compute capability 8.0 (RTX 30 series or newer). exact: the reference's op sequence, "
                                       "bit for bit its output."),
                io.Boolean.Input("kv_offload", default=False,
                                 tooltip="Keep the attention history (the past latent frames every block attends) in "
                                         "RAM between steps instead of VRAM: about 0.8 MB per output token, 5 GB at 4x "
                                         "of 480p. For cards whose VRAM the output size does not fit; costs the "
                                         "transfer time each step. ComfyUI streams the weights by itself when they do "
                                         "not fit."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, model, images, scale, color_fix, seed, decoder, vae=None, kernels="fast", kv_offload=False) -> io.NodeOutput:
        if decoder == "vae" and vae is None:
            raise ValueError("decoder = vae needs the Wan 2.2 VAE connected")
        frames = images.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        total, in_h, in_w = frames.shape[:3]
        size = scaled_size(in_h, in_w, scale)
        height, width = padded_size(*size)
        blocks = list(block_schedule(total))
        log.info("ReCaVSR: %d frames %dx%d -> %dx%d (x%g; model size %dx%d), %d blocks, %s transformer, %s kernels, "
                 "decoder %s, colour fix %s%s", total, in_w, in_h, size[1], size[0], scale, width, height, len(blocks),
                 model.precision, kernels, decoder, color_fix, ", attention history in RAM" if kv_offload else "")
        patchers = model.patchers() if decoder == "flashdecoder" else [model.transformer]
        started = time.perf_counter()
        comfy.model_management.load_models_gpu(patchers, memory_required=activation_bytes(height, width, kv_offload))
        device = model.transformer.load_device
        output = torch.empty((total, size[0], size[1], 3), dtype=torch.float32)
        latents = []
        progress = comfy.utils.ProgressBar(len(blocks))
        with comfy.model_management.cuda_device_context(device):
            stream = VideoStream(model.transformer.model, model.prompt, height, width, seed, device, kernels,
                                 torch.device("cpu") if kv_offload else None)
            decode = DecoderStream(model.decoder.model, model.latent_format, device, torch.bfloat16, kernels) if decoder == "flashdecoder" else None
            sync = torch.cuda.synchronize if device.type == "cuda" else lambda device: None
            sync(device)
            log.info("ReCaVSR: models on %s and conditioning ready in %.1f s", device, time.perf_counter() - started)
            transformer_time = decoder_time = 0.0
            bar = tqdm(total=total, desc="ReCaVSR", unit="frame", dynamic_ncols=True)
            for number, (start, valid, block) in enumerate(blocks, 1):
                lq = prepare_block(frames, start, valid, block, size, device)
                t0 = time.perf_counter()
                latent = stream.step(lq)
                sync(device)
                t1 = time.perf_counter()
                if decode is None:
                    latents.append(latent.cpu())
                else:
                    offset, remaining = start, valid
                    for rgb in decode.step(latent, lq):
                        take = min(remaining, rgb.shape[2])
                        if take:
                            cls.write(output, rgb[0, :, :take, :size[0], :size[1]].transpose(0, 1), frames, offset, color_fix)
                            offset += take
                            remaining -= take
                    sync(device)
                t2 = time.perf_counter()
                transformer_time += t1 - t0
                decoder_time += t2 - t1
                bar.update(valid)
                bar.set_postfix_str(f"block {number}/{len(blocks)}: transformer {(t1 - t0) * 1000:.0f} ms"
                                    + (f", decode and copy {(t2 - t1) * 1000:.0f} ms" if decode is not None else ""))
                progress.update(1)
            bar.close()
        if decode is None:
            t0 = time.perf_counter()
            log.info("ReCaVSR: decoding %d latent frames with the VAE", sum(l.shape[2] for l in latents))
            rgb = vae.decode(model.latent_format.process_out(torch.cat(latents, 2)))
            # one block at a time on the device: the whole 4x video in float32 can be larger than the card
            rgb = rgb.reshape(-1, *rgb.shape[-3:])[:total, :size[0], :size[1]].permute(0, 3, 1, 2)
            for start, valid, _ in blocks:
                cls.write(output, rgb[start:start + valid].to(device) * 2 - 1, frames, start, color_fix)
            decoder_time = time.perf_counter() - t0
        elapsed = time.perf_counter() - started
        log.info("ReCaVSR: %d frames in %.1f s (%.2f fps): transformer %.1f s, decode and frame copies %.1f s",
                 total, elapsed, total / elapsed, transformer_time, decoder_time)
        return io.NodeOutput(output)

    @staticmethod
    def write(output, rgb, frames, offset, color_fix):
        """Store decoded frames [T, 3, H, W] in [-1, 1] as [T, H, W, 3] in [0, 1], colour fixed against the input."""
        take = rgb.shape[0]
        if color_fix == "wavelet":
            fixed = wavelet_color_fix(rgb, frames[offset:offset + take])
        else:
            if color_fix == "adain":
                rgb = adain_color_fix(rgb, upscale_frames(frames[offset:offset + take], rgb.shape[-2:]).to(rgb.device))
            fixed = (rgb.float().clamp(-1, 1) + 1) * 0.5
        output[offset:offset + take] = fixed.permute(0, 2, 3, 1).cpu()



def activation_bytes(height, width, kv_offload=False):
    return (height // 32) * (width // 32) * (ACTIVATION_BYTES_PER_TOKEN - (KV_BYTES_PER_TOKEN if kv_offload else 0))
