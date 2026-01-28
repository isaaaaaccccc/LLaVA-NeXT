import os
import argparse
from transformers import SiglipModel
from llava.model.builder import load_pretrained_model

ROOT = os.path.dirname(os.path.abspath(__file__))

QWEN = os.path.join(ROOT, "model_weights", "llava-onevision-qwen2-0.5b-ov")
SIGLIP = os.path.join(ROOT, "model_weights", "siglip-so400m-patch14-384")


def remove_single_weight(folder):
    f = os.path.join(folder, "model.safetensors")
    if os.path.exists(f):
        os.remove(f)


def cleanup_shards(folder):
    for f in os.listdir(folder):
        if f.startswith("model-") and f.endswith(".safetensors"):
            os.remove(os.path.join(folder, f))

    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        os.remove(index)


def shard_qwen(max_shard_size):
    print("Sharding LLaVA-Qwen")

    tok, model, _, _ = load_pretrained_model(
        QWEN,
        None,
        "llava_qwen",
        device_map="cpu",
        attn_implementation="sdpa",
    )

    model.get_model().save_pretrained(
        QWEN,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    remove_single_weight(QWEN)
    print("Qwen sharded OK")


def shard_siglip(max_shard_size):
    print("Sharding SigLIP")

    model = SiglipModel.from_pretrained(SIGLIP)

    model.save_pretrained(
        SIGLIP,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    remove_single_weight(SIGLIP)
    print("SigLIP sharded OK")


def merge_qwen():
    print("Merging Qwen")

    tok, model, _, _ = load_pretrained_model(
        QWEN,
        None,
        "llava_qwen",
        device_map="cpu",
        attn_implementation="sdpa",
    )

    model.get_model().save_pretrained(QWEN, safe_serialization=True)
    cleanup_shards(QWEN)
    print("Qwen merged OK")


def merge_siglip():
    print("Merging SigLIP")

    model = SiglipModel.from_pretrained(SIGLIP)
    model.save_pretrained(SIGLIP, safe_serialization=True)

    cleanup_shards(SIGLIP)
    print("SigLIP merged OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shard", "merge"], required=True)
    ap.add_argument("--max_shard_size", default="100MB")
    args = ap.parse_args()

    print("=== HF SAFE PIPELINE ===")

    if args.mode == "shard":
        shard_qwen(args.max_shard_size)
        shard_siglip(args.max_shard_size)

    if args.mode == "merge":
        merge_qwen()
        merge_siglip()

    print("=== DONE ===")
