from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image

import torch
from torchvision import transforms

from app.bigram_model import BigramModel
from helper_lib.model import get_model

app = FastAPI(title="Simple Text Generator + Image Classifier")

# ----- text generator (kept)
class TextInput(BaseModel):
    text: str
    start_word: str
    num_words: int = 20

@app.post("/generate")
def generate_text(input_data: TextInput):
    model = BigramModel(input_data.text)
    output = model.generate(input_data.start_word, input_data.num_words)
    return {"generated_text": output}

# ----- image classifier (A2CNN on CIFAR-10)
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

_preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

model_a2: Optional[torch.nn.Module] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_classifier():
    global model_a2
    model_a2 = get_model("A2CNN", num_classes=10).to(device)
    try:
        state = torch.load("checkpoints/a2cnn_cifar10.pt", map_location=device)
        model_a2.load_state_dict(state)
        model_a2.eval()
        print("Loaded checkpoints/a2cnn_cifar10.pt")
    except Exception as e:
        print("Warning: failed to load checkpoint:", e)
        model_a2.eval()

def _read_image(file: UploadFile) -> Image.Image:
    img_bytes = BytesIO(file.file.read())
    img = Image.open(img_bytes).convert("RGB")
    return img

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    img = _read_image(file)
    x = _preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model_a2(x)
        pred = logits.argmax(dim=1).item()
        prob = torch.softmax(logits, dim=1)[0, pred].item()
    return {"class_id": int(pred), "class_name": CIFAR10_CLASSES[pred], "prob": float(prob)}
