# app.py
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch, io, cv2, numpy as np
from torchvision import transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Corrosion-Detection API",
    description="Upload an image → get corrosion yes/no, confidence %, and blur/lighting warnings.",
)

# ───── 1.  Load model once at startup ────────────────────────────────────
model = models.resnet18(weights=None)
model.fc = torch.nn.Sequential(torch.nn.Linear(512, 1), torch.nn.Sigmoid())
model.load_state_dict(torch.load("resnet18_corr.pth", map_location=DEVICE))
model.to(DEVICE).eval()

prep = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# ───── 2.  Helper to flag blurry or dark photos ──────────────────────────
def blur_dark_flags(img_bytes: bytes, blur_thr=100.0, bright_thr=50.0):
    arr = np.frombuffer(img_bytes, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    blur   = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright = gray.mean()
    return bool(blur < blur_thr), bool(bright < bright_thr)   # cast to plain bool

# ───── 3.  /predict endpoint ─────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload one image → get
      corrosion        – True if corrosion detected
      confidence_pct   – How sure the model is about its label (0-100 %)
      warning_blur     – True if image seems blurry
      warning_dark     – True if image seems dark
    """
    # ── 1.  Get raw bytes from the upload
    img_bytes = await file.read()

    # ── 2.  Fast blur / brightness check (heuristic warnings)
    is_blur, is_dark = blur_dark_flags(img_bytes)
    is_blur = bool(is_blur)   # cast NumPy bool  →  plain bool
    is_dark = bool(is_dark)

    # ── 3.  Run the model
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    with torch.no_grad():
        p_no = model(prep(img).unsqueeze(0).to(DEVICE)).item() * 100   # P(no-corrosion) %
    p_corr = 100.0 - p_no                                              # P(corrosion)   %

    # ── 4.  Decide label and confidence
    is_rust     = p_corr >= 50.0
    confidence  = p_corr if is_rust else p_no      # larger prob = confidence

    # ── 5.  Return JSON
    return {
        "corrosion":       is_rust,                   # True = corrosion detected
        "confidence_pct":  round(confidence, 2),      # 0-100 %
        "warning_blur":    is_blur,
        "warning_dark":    is_dark,
    }

# ───── 4.  Optional health check ─────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "alive"}
