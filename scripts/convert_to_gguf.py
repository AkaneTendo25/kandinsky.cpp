#!/usr/bin/env python3
"""
Convert Kandinsky 5 safetensors models to GGUF format.

Usage:
  # Convert all components from a K5 model directory:
  python convert_to_gguf.py --model-dir H:/models/kandinsky-5 --output-dir ./models

  # Convert just the DiT from a specific file:
  python convert_to_gguf.py --dit H:/models/kandinsky-5/kandinsky5pro_t2v_sft_5s.safetensors --output-dir ./models

  # Convert with Q8_0 quantization:
  python convert_to_gguf.py --dit H:/models/k5/dit.safetensors --output-dir ./models --type q8_0

  # Convert shared modules:
  python convert_to_gguf.py --model-dir H:/models/kandinsky-5/k5_modules --output-dir ./models
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Use torch for BF16 support
try:
    import torch
    from safetensors.torch import load_file as torch_load_file
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors package required. Install with: pip install safetensors")
    sys.exit(1)

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGML types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_I32  = 26
GGML_TYPE_BF16 = 30

GGML_TYPE_NAMES = {
    GGML_TYPE_F32: "f32", GGML_TYPE_F16: "f16",
    GGML_TYPE_Q4_0: "q4_0", GGML_TYPE_Q4_1: "q4_1",
    GGML_TYPE_Q5_0: "q5_0", GGML_TYPE_Q5_1: "q5_1",
    GGML_TYPE_Q8_0: "q8_0", GGML_TYPE_BF16: "bf16",
}

TYPE_STR_TO_GGML = {
    "f32": GGML_TYPE_F32, "f16": GGML_TYPE_F16,
    "q4_0": GGML_TYPE_Q4_0, "q4_1": GGML_TYPE_Q4_1,
    "q5_0": GGML_TYPE_Q5_0, "q5_1": GGML_TYPE_Q5_1,
    "q8_0": GGML_TYPE_Q8_0,
}

GGML_BLOCK_SIZES = {
    GGML_TYPE_Q4_0: 32, GGML_TYPE_Q4_1: 32,
    GGML_TYPE_Q5_0: 32, GGML_TYPE_Q5_1: 32,
    GGML_TYPE_Q8_0: 32,
}

# GGUF metadata types
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING  = 8

TOKENIZER_VOCAB_JSON_KEY = "tokenizer.vocab_json"
TOKENIZER_MERGES_TXT_KEY = "tokenizer.merges_txt"


# ── Quantization ──────────────────────────────────────────────────────

def quantize_q8_0(data: np.ndarray) -> bytes:
    """Quantize float32 array to Q8_0 (34 bytes per 32-element block)."""
    data = data.astype(np.float32).flatten()
    n = data.size
    assert n % 32 == 0, f"Q8_0 requires elements divisible by 32, got {n}"

    n_blocks = n // 32
    blocks = data.reshape(n_blocks, 32)
    amax = np.max(np.abs(blocks), axis=1)
    scales = amax / 127.0
    scales[scales == 0] = 1.0
    quantized = np.clip(np.round(blocks / scales[:, np.newaxis]), -128, 127).astype(np.int8)
    scales_f16 = scales.astype(np.float16)

    # Vectorized packing: interleave scale + 32 int8 values
    result = bytearray(n_blocks * 34)
    for i in range(n_blocks):
        off = i * 34
        result[off:off+2] = scales_f16[i].tobytes()
        result[off+2:off+34] = quantized[i].tobytes()
    return bytes(result)


def quantize_q4_0(data: np.ndarray) -> bytes:
    """Quantize float32 array to Q4_0 (18 bytes per 32-element block)."""
    data = data.astype(np.float32).flatten()
    n = data.size
    assert n % 32 == 0, f"Q4_0 requires elements divisible by 32, got {n}"

    n_blocks = n // 32
    blocks = data.reshape(n_blocks, 32)
    # Match ggml quantize_row_q4_0_ref layout and scale sign:
    # - scale d = max_signed / -8
    # - low nibble stores element j, high nibble stores element j + 16
    amax = np.max(np.abs(blocks), axis=1)
    max_idx = np.argmax(np.abs(blocks), axis=1)
    max_signed = blocks[np.arange(n_blocks), max_idx]
    scales = max_signed / -8.0
    inv_scales = np.zeros_like(scales, dtype=np.float32)
    nz = amax > 0
    inv_scales[nz] = 1.0 / scales[nz]
    scales_f16 = scales.astype(np.float16)

    # Vectorized nibble packing: low nibble = j, high nibble = j + 16
    x0 = blocks[:, :16] * inv_scales[:, np.newaxis]
    x1 = blocks[:, 16:] * inv_scales[:, np.newaxis]
    q0 = np.clip((x0 + 8.5).astype(np.int32), 0, 15).astype(np.uint8)
    q1 = np.clip((x1 + 8.5).astype(np.int32), 0, 15).astype(np.uint8)
    packed = q0 | (q1 << 4)

    out = np.empty((n_blocks, 18), dtype=np.uint8)
    out[:, :2] = scales_f16.view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:] = packed
    return out.tobytes()


# ── GGUF Writer ───────────────────────────────────────────────────────

def _quantize_tensor(data: np.ndarray, target_type: int) -> Tuple[bytes, int]:
    """Quantize a tensor and return (bytes, actual_type)."""
    shape = list(data.shape)
    n_elements = int(np.prod(shape))
    # ggml requires ne[0] (inner dim = shape[-1] in safetensors) to be a multiple of block size
    inner_dim = shape[-1] if len(shape) >= 2 else n_elements
    can_quantize = (
        target_type in GGML_BLOCK_SIZES and
        inner_dim % GGML_BLOCK_SIZES[target_type] == 0 and
        data.ndim >= 2
    )
    if target_type == GGML_TYPE_F32:
        return data.astype(np.float32).tobytes(), GGML_TYPE_F32
    elif target_type == GGML_TYPE_F16:
        return data.astype(np.float16).tobytes(), GGML_TYPE_F16
    elif target_type == GGML_TYPE_Q8_0 and can_quantize:
        return quantize_q8_0(data), GGML_TYPE_Q8_0
    elif target_type == GGML_TYPE_Q4_0 and can_quantize:
        return quantize_q4_0(data), GGML_TYPE_Q4_0
    else:
        return data.astype(np.float16).tobytes(), GGML_TYPE_F16


class GGUFWriter:
    """Two-pass GGUF writer: collects tensor info in pass 1, streams data in pass 2."""

    def __init__(self, path: str):
        self.path = path
        self.metadata: List[Tuple] = []
        # tensor_info: list of (name, shape, actual_type, data_size)
        self.tensor_info: List[Tuple] = []
        # Temporary file for tensor data (streamed to disk immediately)
        self.data_path = path + ".data.tmp"
        self.data_file = open(self.data_path, "wb")
        self.data_offset = 0

    def add_string(self, key: str, value: str):
        self.metadata.append((key, GGUF_TYPE_STRING, value))

    def add_uint32(self, key: str, value: int):
        self.metadata.append((key, GGUF_TYPE_UINT32, value))

    def add_int32(self, key: str, value: int):
        self.metadata.append((key, GGUF_TYPE_INT32, value))

    def add_float32(self, key: str, value: float):
        self.metadata.append((key, GGUF_TYPE_FLOAT32, value))

    def add_tensor(self, name: str, data: np.ndarray, target_type: int = GGML_TYPE_F16):
        """Quantize tensor and stream data to temp file immediately."""
        shape = list(data.shape)
        tensor_bytes, actual_type = _quantize_tensor(data, target_type)

        # Align offset
        aligned = (self.data_offset + 31) & ~31
        if aligned > self.data_offset:
            self.data_file.write(b"\x00" * (aligned - self.data_offset))
            self.data_offset = aligned

        self.tensor_info.append((name, shape, actual_type, len(tensor_bytes), aligned))
        self.data_file.write(tensor_bytes)
        self.data_offset += len(tensor_bytes)

    def write(self):
        """Finalize: write header + copy temp data."""
        self.data_file.close()

        with open(self.path, "wb") as f:
            # GGUF header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensor_info)))
            f.write(struct.pack("<Q", len(self.metadata)))

            # Metadata
            for key, vtype, value in self.metadata:
                self._write_string(f, key)
                f.write(struct.pack("<I", vtype))
                if vtype == GGUF_TYPE_STRING:
                    self._write_string(f, value)
                elif vtype == GGUF_TYPE_UINT32:
                    f.write(struct.pack("<I", value))
                elif vtype == GGUF_TYPE_INT32:
                    f.write(struct.pack("<i", value))
                elif vtype == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack("<f", value))

            # Tensor info entries
            for name, shape, dtype, data_size, offset in self.tensor_info:
                self._write_string(f, name)
                f.write(struct.pack("<I", len(shape)))
                for dim in reversed(shape):  # GGUF: ggml order
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", dtype))
                f.write(struct.pack("<Q", offset))

            # Pad to alignment before tensor data
            current_pos = f.tell()
            aligned_pos = (current_pos + 31) & ~31
            f.write(b"\x00" * (aligned_pos - current_pos))

            # Stream tensor data from temp file
            with open(self.data_path, "rb") as df:
                while True:
                    chunk = df.read(64 * 1024 * 1024)  # 64 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)

        # Cleanup temp file
        os.remove(self.data_path)

        total_size = os.path.getsize(self.path)
        print(f"  Written: {self.path} ({total_size / 1024 / 1024:.1f} MB, "
              f"{len(self.tensor_info)} tensors)")

    @staticmethod
    def _write_string(f, s: str):
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)


def embed_tokenizer_assets(writer: GGUFWriter, model_path: str, label: str) -> bool:
    """Embed vocab.json + merges.txt into GGUF metadata when present."""
    roots = []
    path = Path(model_path)
    if path.is_dir():
        roots.append(path)
    else:
        roots.append(path.parent)

    for root in roots:
        vocab = root / "vocab.json"
        merges = root / "merges.txt"
        if vocab.is_file() and merges.is_file():
            writer.add_string(TOKENIZER_VOCAB_JSON_KEY, vocab.read_text(encoding="utf-8"))
            writer.add_string(TOKENIZER_MERGES_TXT_KEY, merges.read_text(encoding="utf-8"))
            print(f"  Embedded {label} tokenizer assets from {root}")
            return True

    print(f"  WARNING: {label} tokenizer assets not found near {model_path}; GGUF will still require external vocab/merges files")
    return False


# ── Lazy tensor loading (handles BF16 via torch, streams one at a time) ──

class LazyTensors:
    """Lazy tensor reader that loads one tensor at a time to avoid OOM."""

    def __init__(self):
        self.files: List[str] = []

    def add_file(self, path: str):
        self.files.append(path)

    def add_dir(self, directory: str):
        for f in sorted(Path(directory).glob("*.safetensors")):
            self.files.append(str(f))

    def keys(self) -> List[str]:
        """Get all tensor names without loading data."""
        result = []
        for path in self.files:
            with safe_open(path, framework="pt" if HAS_TORCH else "numpy") as f:
                result.extend(f.keys())
        return result

    def items(self):
        """Yield (name, numpy_f32_array) one at a time."""
        for path in self.files:
            print(f"  Streaming from {Path(path).name}...")
            framework = "pt" if HAS_TORCH else "numpy"
            with safe_open(path, framework=framework) as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if HAS_TORCH:
                        arr = tensor.float().numpy()
                    else:
                        arr = tensor.astype(np.float32)
                    yield key, arr
                    del tensor, arr  # free immediately

    def get_tensor(self, name: str) -> np.ndarray:
        """Load a single tensor by name."""
        for path in self.files:
            framework = "pt" if HAS_TORCH else "numpy"
            with safe_open(path, framework=framework) as f:
                if name in f.keys():
                    tensor = f.get_tensor(name)
                    if HAS_TORCH:
                        return tensor.float().numpy()
                    return tensor.astype(np.float32)
        raise KeyError(f"Tensor {name} not found")

    def get_shape(self, name: str) -> list:
        """Get tensor shape without loading data (reads header only)."""
        for path in self.files:
            header = _read_safetensors_header(path)
            if name in header:
                return header[name]["shape"]
        raise KeyError(f"Tensor {name} not found in headers")

    def get_all_shapes(self) -> Dict[str, list]:
        """Get all tensor shapes from headers (no data loading)."""
        result = {}
        for path in self.files:
            header = _read_safetensors_header(path)
            for k, v in header.items():
                result[k] = v["shape"]
        return result


def _read_safetensors_header(path: str) -> dict:
    """Read safetensors JSON header without loading tensor data."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


# ── Auto-detect DiT architecture ─────────────────────────────────────

def detect_dit_config(shapes: Dict[str, list]) -> dict:
    """Detect DiT architecture from tensor shapes (no data loading)."""
    text_blocks = set()
    visual_blocks = set()
    for k in shapes:
        if k.startswith("text_transformer_blocks."):
            text_blocks.add(int(k.split(".")[1]))
        elif k.startswith("visual_transformer_blocks."):
            visual_blocks.add(int(k.split(".")[1]))

    model_dim = shapes["text_embeddings.in_layer.weight"][0]
    ff_dim = shapes["visual_transformer_blocks.0.feed_forward.in_layer.weight"][0]
    time_dim = shapes["time_embeddings.in_layer.weight"][0]
    in_visual_dim = shapes["visual_embeddings.in_layer.weight"][1]  # 132
    out_visual_dim = shapes["out_layer.out_layer.weight"][0]  # 64

    config = {
        "model_dim": model_dim,
        "ff_dim": ff_dim,
        "time_dim": time_dim,
        "in_visual_dim": in_visual_dim,
        "out_visual_dim": out_visual_dim,
        "num_text_blocks": len(text_blocks),
        "num_visual_blocks": len(visual_blocks),
    }

    variant = "unknown"
    if model_dim == 1792 and len(visual_blocks) == 32:
        variant = "lite"
    elif model_dim == 4096 and len(visual_blocks) == 60:
        variant = "pro"
    elif model_dim == 2560 and len(visual_blocks) == 50:
        variant = "lite-image"

    config["variant"] = variant
    return config


# ── Converters ────────────────────────────────────────────────────────

def convert_dit(path: str, output_dir: str, target_type: int, output_name: str):
    """Convert a DiT safetensors file to GGUF (streaming, low memory)."""
    print(f"\n=== Converting DiT: {Path(path).name} ===")

    lazy = LazyTensors()
    if os.path.isdir(path):
        lazy.add_dir(path)
    else:
        lazy.add_file(path)

    # Detect config from headers (no data loaded)
    shapes = lazy.get_all_shapes()
    if not shapes:
        print("  No tensors found, skipping")
        return

    config = detect_dit_config(shapes)
    print(f"  Detected: variant={config['variant']}, model_dim={config['model_dim']}, "
          f"blocks={config['num_text_blocks']}t+{config['num_visual_blocks']}v, "
          f"ff_dim={config['ff_dim']}, time_dim={config['time_dim']}")

    writer = GGUFWriter(os.path.join(output_dir, output_name))
    writer.add_string("general.architecture", "kandinsky5-dit")
    writer.add_string("general.name", f"Kandinsky 5 DiT ({config['variant']})")
    writer.add_uint32("dit.model_dim", config["model_dim"])
    writer.add_uint32("dit.ff_dim", config["ff_dim"])
    writer.add_uint32("dit.time_dim", config["time_dim"])
    writer.add_uint32("dit.in_visual_dim", config["in_visual_dim"])
    writer.add_uint32("dit.out_visual_dim", config["out_visual_dim"])
    writer.add_uint32("dit.num_text_blocks", config["num_text_blocks"])
    writer.add_uint32("dit.num_visual_blocks", config["num_visual_blocks"])

    count_q = 0
    count_f = 0
    for name, data in lazy.items():
        if "rope_embeddings.args" in name:
            continue
        if data.ndim <= 1 or data.size < 256:
            writer.add_tensor(name, data, GGML_TYPE_F32)
            count_f += 1
        else:
            writer.add_tensor(name, data, target_type)
            if target_type in GGML_BLOCK_SIZES:
                count_q += 1
            else:
                count_f += 1

    print(f"  Tensors: {count_q} quantized ({GGML_TYPE_NAMES.get(target_type, '?')}), "
          f"{count_f} kept as f32/f16")
    writer.write()


def convert_qwen(path: str, output_dir: str, target_type: int):
    """Convert Qwen2.5 text encoder (streaming)."""
    print(f"\n=== Converting Qwen2.5: {path} ===")

    lazy = LazyTensors()
    if os.path.isdir(path):
        lazy.add_dir(path)
    else:
        lazy.add_file(path)

    all_keys = lazy.keys()
    if not all_keys:
        print("  No tensors found, skipping")
        return

    skip_keys = {k for k in all_keys if "visual" in k or "vision" in k or "lm_head" in k}
    print(f"  Found {len(all_keys)} tensors, skipping {len(skip_keys)} (visual/lm_head)")

    writer = GGUFWriter(os.path.join(output_dir, "qwen.gguf"))
    writer.add_string("general.architecture", "qwen2")
    writer.add_string("general.name", "Qwen2.5-VL-7B Text Encoder")
    writer.add_uint32("qwen2.block_count", 28)
    writer.add_uint32("qwen2.embedding_length", 3584)
    writer.add_uint32("qwen2.attention.head_count", 28)
    writer.add_uint32("qwen2.attention.head_count_kv", 4)
    writer.add_float32("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_float32("qwen2.rope.freq_base", 1000000.0)
    embed_tokenizer_assets(writer, path, "Qwen")

    count = 0
    for name, data in lazy.items():
        if name in skip_keys:
            continue
        if data.ndim <= 1 or data.size < 256:
            writer.add_tensor(name, data, GGML_TYPE_F32)
        else:
            writer.add_tensor(name, data, target_type)
        count += 1

    print(f"  Converted {count} tensors")
    writer.write()


def convert_clip(path: str, output_dir: str, target_type: int):
    """Convert CLIP text encoder (streaming)."""
    print(f"\n=== Converting CLIP: {path} ===")

    lazy = LazyTensors()
    if os.path.isdir(path):
        lazy.add_dir(path)
    else:
        lazy.add_file(path)

    writer = GGUFWriter(os.path.join(output_dir, "clip.gguf"))
    writer.add_string("general.architecture", "clip")
    writer.add_string("general.name", "CLIP-ViT-L/14 Text Encoder")
    writer.add_uint32("clip.embedding_length", 768)
    writer.add_uint32("clip.block_count", 12)
    writer.add_uint32("clip.attention.head_count", 12)
    embed_tokenizer_assets(writer, path, "CLIP")

    count = 0
    skipped = 0
    for name, data in lazy.items():
        # Skip vision model — we only need text model for K5
        if name.startswith("vision_model.") or name.startswith("visual_projection."):
            skipped += 1
            continue
        if data.ndim <= 1 or data.size < 256:
            writer.add_tensor(name, data, GGML_TYPE_F32)
        else:
            writer.add_tensor(name, data, target_type)
        count += 1

    print(f"  Converted {count} tensors, skipped {skipped} (vision)")
    writer.write()


def convert_vae_name(name: str) -> str:
    """Convert HuggingFace diffusers VAE name to internal C++ name."""
    import re
    n = name
    # conv_in.conv → conv_in, conv_out.conv → conv_out
    n = re.sub(r'(conv_in|conv_out)\.conv\.', r'\1.', n)
    # conv_norm_out → norm_out
    n = n.replace('conv_norm_out.', 'norm_out.')
    # down_blocks.X.resnets.Y → down.X.block.Y
    n = re.sub(r'down_blocks\.(\d+)\.resnets\.(\d+)\.', r'down.\1.block.\2.', n)
    # down_blocks.X.downsamplers.0.conv.conv → down.X.downsample.conv
    n = re.sub(r'down_blocks\.(\d+)\.downsamplers\.0\.conv\.conv\.', r'down.\1.downsample.conv.', n)
    # up_blocks.X.resnets.Y → up.X.block.Y
    n = re.sub(r'up_blocks\.(\d+)\.resnets\.(\d+)\.', r'up.\1.block.\2.', n)
    # up_blocks.X.upsamplers.0.conv.conv → up.X.upsample.conv
    n = re.sub(r'up_blocks\.(\d+)\.upsamplers\.0\.conv\.conv\.', r'up.\1.upsample.conv.', n)
    # mid_block.resnets.0 → mid.block_1, mid_block.resnets.1 → mid.block_2
    n = n.replace('mid_block.resnets.0.', 'mid.block_1.')
    n = n.replace('mid_block.resnets.1.', 'mid.block_2.')
    # mid_block.attentions.0.group_norm → mid.attn_1.norm
    n = n.replace('mid_block.attentions.0.group_norm.', 'mid.attn_1.norm.')
    # mid_block.attentions.0.to_q → mid.attn_1.q (etc for k, v)
    n = n.replace('mid_block.attentions.0.to_q.', 'mid.attn_1.q.')
    n = n.replace('mid_block.attentions.0.to_k.', 'mid.attn_1.k.')
    n = n.replace('mid_block.attentions.0.to_v.', 'mid.attn_1.v.')
    n = n.replace('mid_block.attentions.0.to_out.0.', 'mid.attn_1.proj_out.')
    # resnet conv wrappers: conv1.conv → conv1, conv2.conv → conv2
    n = re.sub(r'(conv[12])\.conv\.', r'\1.', n)
    # conv_shortcut.conv → nin_shortcut
    n = n.replace('conv_shortcut.conv.', 'nin_shortcut.')
    n = n.replace('conv_shortcut.', 'nin_shortcut.')
    # Remove quant_conv / post_quant_conv (not used in FLUX VAE without quantization)
    return n


def convert_vae(path: str, output_dir: str, target_type: int):
    """Convert VAE (streaming, always F16 for quality)."""
    print(f"\n=== Converting VAE: {path} ===")

    lazy = LazyTensors()
    if os.path.isdir(path):
        lazy.add_dir(path)
    else:
        lazy.add_file(path)

    writer = GGUFWriter(os.path.join(output_dir, "vae.gguf"))
    writer.add_string("general.architecture", "vae")
    writer.add_string("general.name", "Kandinsky 5 VAE")
    writer.add_uint32("vae.latent_channels", 16)
    writer.add_float32("vae.scaling_factor", 0.476986)

    count = 0
    skipped = 0
    for name, data in lazy.items():
        internal_name = convert_vae_name(name)
        # Skip quant_conv / post_quant_conv
        if name.startswith('quant_conv') or name.startswith('post_quant_conv'):
            skipped += 1
            continue
        if data.ndim <= 1 or data.size < 256:
            writer.add_tensor(internal_name, data, GGML_TYPE_F32)
        else:
            writer.add_tensor(internal_name, data, GGML_TYPE_F16)
        count += 1

    print(f"  Converted {count} tensors, skipped {skipped}")
    writer.write()


# ── Path discovery ────────────────────────────────────────────────────

def find_dit_path(model_dir: str) -> str:
    """Find DiT safetensors in model directory."""
    # Prefer official checkpoints over alternate distilled / fp8 exports when both exist.
    preferred = [
        "kandinsky5pro_t2v_sft_5s.safetensors",
        "kandinsky5pro_t2v_sft_10s.safetensors",
        "kandinsky5pro_t2v_pretrain_5s.safetensors",
        "kandinsky5pro_t2v_pretrain_10s.safetensors",
        "kandinsky5pro_i2v_sft_5s.safetensors",
        "kandinsky5lite_t2v_pretrain_5s.safetensors",
        "kandinsky5lite_i2v_5s.safetensors",
    ]
    for name in preferred:
        path = Path(model_dir) / name
        if path.is_file():
            return str(path)

    # Direct K5 checkpoint files
    for pat in ["kandinsky5*t2v*.safetensors", "kandinsky5*t2i*.safetensors",
                "kandinsky5*i2v*.safetensors", "kandinsky5*i2i*.safetensors"]:
        matches = list(Path(model_dir).glob(pat))
        if matches:
            return str(matches[0])
    # Subdirectories
    for subdir in ["model", "transformer", "dit"]:
        d = os.path.join(model_dir, subdir)
        if os.path.isdir(d):
            for f in Path(d).glob("*.safetensors"):
                return str(f)
    # Any safetensors in root with DiT-like names
    for f in Path(model_dir).glob("*.safetensors"):
        if "dit" in f.name.lower() or "diffusion" in f.name.lower():
            return str(f)
    return ""


def find_qwen_path(model_dir: str) -> str:
    for d in [os.path.join(model_dir, "text_encoder"),
              os.path.join(model_dir, "k5_modules", "text_encoder")]:
        if os.path.isdir(d) and list(Path(d).glob("*.safetensors")):
            return d
    return ""


def find_clip_path(model_dir: str) -> str:
    for d in [os.path.join(model_dir, "text_encoder_2"),
              os.path.join(model_dir, "text_encoder2"),
              os.path.join(model_dir, "k5_modules", "text_encoder2")]:
        if os.path.isdir(d) and list(Path(d).glob("*.safetensors")):
            return d
    return ""


def convert_vae3d_name(name: str) -> str:
    """Convert HunyuanVideo CausalVideoVAE name to internal C++ name."""
    import re
    n = name

    # Keep "decoder." prefix — C++ HunyuanVideoVAE inits decoder with prefix "decoder",
    # so GGUF tensor names must include "decoder." to match C++ collect_params()

    # conv_in.conv → conv_in, conv_out.conv → conv_out
    n = re.sub(r'(conv_in|conv_out)\.conv\.', r'\1.', n)

    # conv_norm_out → norm_out
    n = n.replace('conv_norm_out.', 'norm_out.')

    # mid_block.resnets.0 → mid.block_1, mid_block.resnets.1 → mid.block_2
    n = n.replace('mid_block.resnets.0.', 'mid.block_1.')
    n = n.replace('mid_block.resnets.1.', 'mid.block_2.')

    # mid_block.attentions.0.group_norm → mid.attn_1.norm
    n = n.replace('mid_block.attentions.0.group_norm.', 'mid.attn_1.norm.')
    n = n.replace('mid_block.attentions.0.to_q.', 'mid.attn_1.q.')
    n = n.replace('mid_block.attentions.0.to_k.', 'mid.attn_1.k.')
    n = n.replace('mid_block.attentions.0.to_v.', 'mid.attn_1.v.')
    n = n.replace('mid_block.attentions.0.to_out.0.', 'mid.attn_1.proj_out.')

    # up_blocks.X.resnets.Y → up.X.block.Y
    n = re.sub(r'up_blocks\.(\d+)\.resnets\.(\d+)\.', r'up.\1.block.\2.', n)

    # up_blocks.X.upsamplers.0.conv.conv → up.X.upsample.conv
    n = re.sub(r'up_blocks\.(\d+)\.upsamplers\.0\.conv\.conv\.', r'up.\1.upsample.conv.', n)

    # resnet conv wrappers: conv1.conv → conv1, conv2.conv → conv2
    n = re.sub(r'(conv[12])\.conv\.', r'\1.', n)

    # conv_shortcut.conv → nin_shortcut
    n = n.replace('conv_shortcut.conv.', 'nin_shortcut.')
    n = n.replace('conv_shortcut.', 'nin_shortcut.')

    return n


def convert_vae3d(path: str, output_dir: str, target_type: int):
    """Convert HunyuanVideo 3D VAE (CausalVideoVAE) to GGUF.

    5D conv3d weights [OC, IC, kD, kH, kW] are reshaped to 4D [OC*kD, IC, kH, kW]
    with kD encoded as a suffix in the tensor name (e.g. ".weight.kd3").
    1x1x1 convs (post_quant_conv) are also 5D [OC, IC, 1, 1, 1] and get suffix .kd1.
    """
    print(f"\n=== Converting 3D VAE (HunyuanVideo): {path} ===")

    lazy = LazyTensors()
    if os.path.isdir(path):
        lazy.add_dir(path)
    else:
        lazy.add_file(path)

    writer = GGUFWriter(os.path.join(output_dir, "vae3d.gguf"))
    writer.add_string("general.architecture", "vae3d")
    writer.add_string("general.name", "HunyuanVideo CausalVideoVAE")
    writer.add_uint32("vae3d.latent_channels", 16)
    writer.add_float32("vae3d.scaling_factor", 0.476986)
    writer.add_uint32("vae3d.temporal_compression", 4)
    writer.add_uint32("vae3d.spatial_compression", 8)

    # Get all shapes to know which tensors are 5D
    shapes = lazy.get_all_shapes()
    print(f"  Total tensors in source: {len(shapes)}")

    # Only keep decoder + post_quant_conv tensors (we don't need encoder)
    count = 0
    skipped = 0
    for name, data in lazy.items():
        # Skip encoder tensors — we only need decoder for generation
        if name.startswith('encoder.'):
            skipped += 1
            continue
        # Skip quant_conv (encoder-side quantization)
        if name.startswith('quant_conv'):
            skipped += 1
            continue

        internal_name = convert_vae3d_name(name)

        if data.ndim == 5:
            # 5D conv3d weight: [OC, IC, kD, kH, kW]
            oc, ic, kd, kh, kw = data.shape
            # Reorder to [OC, kD, IC, kH, kW], then collapse [OC, kD] -> [OC*kD].
            # Direct reshape from [OC, IC, kD, kH, kW] is incorrect because IC sits
            # between OC and kD in memory layout.
            data_4d = np.transpose(data, (0, 2, 1, 3, 4)).reshape(oc * kd, ic, kh, kw)
            # Encode kD in tensor name
            if internal_name.endswith('.weight'):
                internal_name = internal_name[:-len('.weight')] + f'.weight.kd{kd}'
            print(f"    5D->4D: {name} [{oc},{ic},{kd},{kh},{kw}] -> {internal_name} [{oc*kd},{ic},{kh},{kw}]")
            writer.add_tensor(internal_name, data_4d, GGML_TYPE_F16)
        elif data.ndim == 2:
            # 2D weight (e.g. post_quant_conv 1x1x1 stored as [OC, IC])
            writer.add_tensor(internal_name, data, GGML_TYPE_F16)
        elif data.ndim <= 1 or data.size < 256:
            writer.add_tensor(internal_name, data, GGML_TYPE_F32)
        else:
            writer.add_tensor(internal_name, data, GGML_TYPE_F16)
        count += 1

    print(f"  Converted {count} tensors, skipped {skipped} (encoder)")
    writer.write()


def find_vae3d_path(model_dir: str) -> str:
    """Find HunyuanVideo 3D VAE safetensors."""
    for d in [os.path.join(model_dir, "hunyuan", "vae"),
              os.path.join(model_dir, "k5_modules", "hunyuan", "vae"),
              os.path.join(model_dir, "vae3d")]:
        if os.path.isdir(d) and list(Path(d).glob("*.safetensors")):
            return d
    return ""


def find_vae_path(model_dir: str) -> str:
    for d in [os.path.join(model_dir, "vae"),
              os.path.join(model_dir, "k5_modules", "hunyuan", "vae"),
              os.path.join(model_dir, "flux", "vae")]:
        if os.path.isdir(d) and list(Path(d).glob("*.safetensors")):
            return d
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kandinsky 5 safetensors to GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Pro T2V DiT to Q8_0:
  %(prog)s --dit H:/models/kandinsky-5/kandinsky5pro_t2v_sft_5s.safetensors --output-dir ./models --type q8_0

  # Convert all from model directory:
  %(prog)s --model-dir H:/models/kandinsky-5 --output-dir ./models

  # Convert just the shared modules:
  %(prog)s --model-dir H:/models/kandinsky-5/k5_modules --output-dir ./models --skip-dit
""")
    parser.add_argument("--model-dir", help="Model directory (auto-discovers components)")
    parser.add_argument("--output-dir", required=True, help="Output directory for GGUF files")
    parser.add_argument("--type", default="f16", choices=list(TYPE_STR_TO_GGML.keys()),
                        help="Target quantization type (default: f16)")
    parser.add_argument("--dit", help="Path to DiT safetensors file (overrides auto-discovery)")
    parser.add_argument("--qwen", help="Path to Qwen directory or file")
    parser.add_argument("--clip", help="Path to CLIP directory or file")
    parser.add_argument("--vae", help="Path to VAE directory or file")
    parser.add_argument("--vae3d", help="Path to HunyuanVideo 3D VAE directory or file")
    parser.add_argument("--dit-name", default="", help="Output filename for DiT GGUF (default: <checkpoint>.<type>.gguf)")
    parser.add_argument("--skip-dit", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-vae3d", action="store_true")
    args = parser.parse_args()

    if not args.model_dir and not any([args.dit, args.qwen, args.clip, args.vae, args.vae3d]):
        parser.error("Must specify --model-dir or at least one of --dit/--qwen/--clip/--vae/--vae3d")

    os.makedirs(args.output_dir, exist_ok=True)
    target_type = TYPE_STR_TO_GGML[args.type]

    print(f"Output: {args.output_dir}")
    print(f"Type: {args.type}")
    if not HAS_TORCH:
        print("WARNING: torch not found — BF16 tensors will fail. Install: pip install torch")

    # Resolve paths
    dit_path = args.dit or (find_dit_path(args.model_dir) if args.model_dir else "")
    qwen_path = args.qwen or (find_qwen_path(args.model_dir) if args.model_dir else "")
    clip_path = args.clip or (find_clip_path(args.model_dir) if args.model_dir else "")
    vae_path = args.vae or (find_vae_path(args.model_dir) if args.model_dir else "")
    vae3d_path = args.vae3d or (find_vae3d_path(args.model_dir) if args.model_dir else "")

    if dit_path and not args.skip_dit:
        dit_name = args.dit_name
        if not dit_name:
            base = Path(dit_path).name
            if base.endswith(".safetensors"):
                base = base[:-len(".safetensors")]
            dit_name = f"{base}.{args.type}.gguf"
        convert_dit(dit_path, args.output_dir, target_type, dit_name)
    elif not args.skip_dit and not dit_path:
        print("\nDiT: not found (use --dit to specify)")

    if qwen_path and not args.skip_qwen:
        convert_qwen(qwen_path, args.output_dir, target_type)
    elif not args.skip_qwen and not qwen_path:
        print("\nQwen: not found (use --qwen to specify)")

    if clip_path and not args.skip_clip:
        convert_clip(clip_path, args.output_dir, target_type)
    elif not args.skip_clip and not clip_path:
        print("\nCLIP: not found (use --clip to specify)")

    if vae_path and not args.skip_vae:
        convert_vae(vae_path, args.output_dir, target_type)
    elif not args.skip_vae and not vae_path:
        print("\nVAE: not found (use --vae to specify)")

    if vae3d_path and not args.skip_vae3d:
        convert_vae3d(vae3d_path, args.output_dir, target_type)
    elif not args.skip_vae3d and not vae3d_path:
        print("\nVAE3D: not found (use --vae3d to specify)")

    print("\nDone!")


if __name__ == "__main__":
    main()
