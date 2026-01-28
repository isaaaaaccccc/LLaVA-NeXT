import os
import sys
import copy
import warnings

import torch
import av
from PIL import Image


# =====================================================
# Resolve project root dynamically
# =====================================================
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(REPO_ROOT, "model_weights", "llava-onevision-qwen2-0.5b-ov")
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")

IMAGE_PATH = os.path.join(EXAMPLES_DIR, "example.png")
VIDEO_PATH = os.path.join(EXAMPLES_DIR, "rgb.mp4")


# =====================================================
# Offline + silence noise
# =====================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

warnings.filterwarnings("ignore")


# =====================================================
# Make "llava" importable WITHOUT pip install
# =====================================================
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


# =====================================================
# Model selection
# =====================================================
PRETRAINED = MODEL_DIR
MODEL_NAME = "llava_qwen"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability(0)
    DTYPE = torch.bfloat16 if major >= 8 else torch.float16
else:
    DTYPE = torch.float32


# =====================================================
# Load LLaVA
# =====================================================
llava_model_args = {
    "multimodal": True,
    "attn_implementation": "sdpa",
    "torch_dtype": DTYPE,
}

tokenizer, model, image_processor, max_length = load_pretrained_model(
    PRETRAINED,
    None,
    MODEL_NAME,
    device_map=DEVICE,
    **llava_model_args,
)

mm_projector = model.get_model().mm_projector
mm_projector.to(dtype=DTYPE, device=DEVICE)

model.eval()
model.to(DEVICE)


# =====================================================
# Debug print model info
# =====================================================
print("LLM class:", model.get_model().__class__)
print("LLM config name_or_path:", model.config._name_or_path)

vision_tower = model.get_vision_tower()
print("Vision tower class:", vision_tower.__class__)
print("Vision tower name:", getattr(vision_tower, "vision_tower_name", None))
print("MM projector class:", model.get_model().mm_projector.__class__)


# =====================================================
# PyAV VIDEO LOADER
# =====================================================
def load_video_frames(video_path, max_frames=8, stride=10):
    container = av.open(video_path)
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % stride == 0:
            frames.append(frame.to_image())
        if len(frames) >= max_frames:
            break
    return frames


# =====================================================
# INPUT: IMAGE or VIDEO
# =====================================================
USE_VIDEO = False

if USE_VIDEO:
    images = load_video_frames(VIDEO_PATH, max_frames=8)
    modalities = ["video"]
    image_sizes = [img.size for img in images]
else:
    images = [Image.open(IMAGE_PATH).convert("RGB")]
    modalities = ["image"]
    image_sizes = [images[0].size]


# =====================================================
# Image preprocessing
# =====================================================
image_tensor = process_images(images, image_processor, model.config)
image_tensor = [_img.to(dtype=DTYPE, device=DEVICE) for _img in image_tensor]


# =====================================================
# Prompt
# =====================================================
conv_template = "qwen_1_5"

question = (
    DEFAULT_IMAGE_TOKEN
    + " You are a robotics perception system.\n"
      "Describe the scene in detail:\n"
      "1. List all objects visible (type, color, approximate shape).\n"
      "2. Describe interactions between objects.\n"
      "3. Describe the robot's current action stage.\n"
      "4. Describe spatial relationships.\n"
      "Be precise. Do not guess unseen details."
)

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()


# =====================================================
# Tokenize
# =====================================================
input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    IMAGE_TOKEN_INDEX,
    return_tensors="pt",
).unsqueeze(0).to(DEVICE)


# =====================================================
# Generate
# =====================================================
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        modalities=modalities,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=256,
    )

answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print("\n=== OUTPUT ===\n")
print(answer)
