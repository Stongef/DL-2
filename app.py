"""Evcil Hayvan Irkı Sınıflandırıcı – Gradio uygulaması.

Gereken dosyalar (notebook'tan indirilir): pet_resnet50.pth, classes.json
Çalıştırma:  pip install torch torchvision gradio pillow  &&  python app.py
"""
import json

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50

with open("classes.json") as f:
    class_names = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("pet_resnet50.pth", map_location=device))
model.to(device).eval()

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(img):
    if img is None:
        return None
    x = eval_tf(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Bir kedi/köpek fotoğrafı yükle"),
    outputs=gr.Label(num_top_classes=5, label="Tahmin"),
    title="Evcil Hayvan Irkı Sınıflandırıcı (ResNet-50)",
    description="Oxford-IIIT Pet veri setiyle eğitilmiş 37 sınıflı model.",
)

if __name__ == "__main__":
    demo.launch()
