import os
import argparse
import torch

from transformers import SiglipModel
from llava.model.builder import load_pretrained_model


def shard_llava_qwen(src_dir, dst_dir, max_shard_size="100MB"):
    print(f"\n=== Sharding LLaVA-Qwen model: {src_dir} ===")

    os.makedirs(dst_dir, exist_ok=True)

    # Load via LLaVA with FlashAttention DISABLED
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        src_dir,
        None,
        "llava_qwen",
        device_map="cpu",
        attn_implementation="sdpa",   # <<< FIX HERE
    )

    llm = model.get_model()

    print("Saving sharded LLaVA-Qwen weights...")
    llm.save_pretrained(
        dst_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    print(f"✅ Done LLaVA-Qwen → {dst_dir}")


def shard_siglip(src_dir, dst_dir, max_shard_size="100MB"):
    print(f"\n=== Sharding SigLIP model: {src_dir} ===")

    os.makedirs(dst_dir, exist_ok=True)

    model = SiglipModel.from_pretrained(src_dir)

    print("Saving sharded SigLIP weights...")
    model.save_pretrained(
        dst_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    print(f"✅ Done SigLIP → {dst_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="model_weights")
    parser.add_argument("--out", default="model_weights_sharded")
    parser.add_argument("--max_shard_size", default="100MB")
    args = parser.parse_args()

    qwen_src = os.path.join(args.root, "llava-onevision-qwen2-0.5b-ov")
    siglip_src = os.path.join(args.root, "siglip-so400m-patch14-384")

    qwen_dst = os.path.join(args.out, "llava-onevision-qwen2-0.5b-ov")
    siglip_dst = os.path.join(args.out, "siglip-so400m-patch14-384")

    print("\n=== START SHARDING PROCESS ===")

    shard_llava_qwen(qwen_src, qwen_dst, args.max_shard_size)
    shard_siglip(siglip_src, siglip_dst, args.max_shard_size)

    print("\n🎉 ALL MODELS SHARDED SUCCESSFULLY")


if __name__ == "__main__":
    main()
