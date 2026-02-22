# server.py
import os
import io
import traceback
import threading
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from PIL import Image

import torch
import numpy as np
import cv2

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from ultralytics import YOLO

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

app = FastAPI(title="Curiosity Camera API (YOLO-grounded)")

# ---------------- Models ----------------
VLM_ID = "Qwen/Qwen2-VL-2B-Instruct"
YOLO_WEIGHTS = "yolo11n.pt"  # change path if needed

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

processor = AutoProcessor.from_pretrained(VLM_ID)
vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_ID,
    torch_dtype="auto",
    device_map="auto",
)

# Run YOLO on CPU to avoid VRAM issues
detector = YOLO(YOLO_WEIGHTS)

gen_lock = threading.Lock()

SYSTEM_RULES = (
    "You are a helpful, kid-safe tutor for children.\n"
    "Only describe what you can actually see.\n"
    "If the image is too dark/blurry, say so and ask for a brighter/closer photo.\n"
    "DO NOT invent objects.\n"
    "If you use the 'Detected objects' list, treat it as a helpful hint, not absolute truth.\n"
    "Only give safety warnings if you clearly see something dangerous.\n"
)

# ---------------- Image helpers ----------------
def shrink_image(pil: Image.Image, max_side: int = 640) -> Image.Image:
    w, h = pil.size
    m = max(w, h)
    if m <= max_side:
        return pil
    scale = max_side / float(m)
    return pil.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

def rotate_pil(pil: Image.Image, rotate: int) -> Image.Image:
    rotate = rotate % 360
    if rotate == 90:
        return pil.transpose(Image.ROTATE_90)
    if rotate == 180:
        return pil.transpose(Image.ROTATE_180)
    if rotate == 270:
        return pil.transpose(Image.ROTATE_270)
    return pil

def mean_luma(pil: Image.Image) -> float:
    img = np.array(pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())

def enhance_low_light(pil: Image.Image, luma: float) -> Image.Image:
    """
    Low-light enhancement:
      - denoise
      - gamma brighten (stronger if darker)
      - CLAHE on luminance
    """
    rgb = np.array(pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Denoise first (helps reduce hallucination after brightening)
    bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)

    # Gamma based on darkness
    if luma < 15:
        gamma = 2.8
    elif luma < 25:
        gamma = 2.2
    else:
        gamma = 1.7

    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    bgr = cv2.LUT(bgr, table)

    # CLAHE on LAB luminance
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb2)

# ---------------- YOLO grounding ----------------
def yolo_detect_labels(pil: Image.Image, imgsz: int = 320, conf_thres: float = 0.35) -> List[Tuple[str, float]]:
    """
    Returns list of (label, conf) sorted desc, unique labels kept.
    """
    rgb = np.array(pil.convert("RGB"))
    results = detector(rgb, imgsz=imgsz, verbose=False, device="cpu")
    r0 = results[0]

    found = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        for b in r0.boxes:
            conf = float(b.conf[0])
            if conf < conf_thres:
                continue
            cls = int(b.cls[0])
            name = detector.names.get(cls, str(cls))
            found.append((name, conf))

    # keep best confidence per label
    best = {}
    for name, conf in found:
        best[name] = max(best.get(name, 0.0), conf)

    out = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return out[:6]  # top few

# ---------------- VLM helpers ----------------
def move_inputs_to_model_device(inputs: dict) -> dict:
    device = vlm.get_input_embeddings().weight.device
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    return inputs

def run_vlm(pil_image: Image.Image, prompt: str, max_new_tokens: int = 90) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    if not image_inputs:
        image_inputs = [pil_image]
        video_inputs = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model_device(inputs)

    with gen_lock, torch.inference_mode():
        out = vlm.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    ans = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return ans.strip()

# ---------------- API ----------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "vlm": VLM_ID,
        "vlm_device": str(vlm.get_input_embeddings().weight.device),
        "yolo": YOLO_WEIGHTS,
    }

@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    print(traceback.format_exc())
    if "CUDA out of memory" in str(exc) and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    age: int = Form(8),
    rotate: int = Form(0),          # 0/90/180/270
    force_enhance: int = Form(1),   # 1 = always enhance if low light
):
    pil = Image.open(io.BytesIO(await image.read())).convert("RGB")
    pil = rotate_pil(pil, rotate)
    pil = shrink_image(pil, max_side=640)

    # Save raw input for debugging
    pil.save("last_input.jpg", quality=95)

    luma_before = mean_luma(pil)
    low_light = luma_before < 45.0

    # Enhance if low light
    pil2 = pil
    if low_light and force_enhance:
        pil2 = enhance_low_light(pil, luma_before)

    luma_after = mean_luma(pil2)
    pil2.save("last_enhanced.jpg", quality=95)

    # YOLO grounding on the (possibly enhanced) image
    dets = yolo_detect_labels(pil2, imgsz=320, conf_thres=0.35)
    det_text = ", ".join([f"{n}({c:.2f})" for n, c in dets]) if dets else "none"

    prompt = (
        f"{SYSTEM_RULES}\n\n"
        f"Detected objects (hint): {det_text}\n"
        f"Photo brightness (0-255): before={luma_before:.1f}, after={luma_after:.1f}\n\n"
        f"For a child aged {age}:\n"
        "1) What is this? (use the hint if it makes sense)\n"
        "2) What is it used for?\n"
        "3) One fun fact.\n"
        "4) Two safe follow-up questions.\n"
        "If you are not confident because the photo is dark/blurry, say what is unclear and ask for a brighter/closer photo.\n"
    )

    answer = run_vlm(pil2, prompt, max_new_tokens=110)

    return {
        "answer": answer,
        "low_light": low_light,
        "mean_luma_before": luma_before,
        "mean_luma_after": luma_after,
        "yolo_hints": dets,
        "debug_files": ["last_input.jpg", "last_enhanced.jpg"],
    }
