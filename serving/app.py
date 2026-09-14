"""
Sesi 3 - Single Object Detection: Model Serving (FastAPI, lokal)
Bootcamp: rubythalib.ai -- mentor Daniel Syahputra

ARSITEKTUR: v3 (split head: objectness_head + bbox_head). Kalau best.pt yang
di-load berasal dari training v1/v2 (head gabungan 'classifier'), load_state_dict
akan gagal (Missing/Unexpected keys). Pastikan versi training & serving sinkron.

Jalankan: uvicorn app:app --host 0.0.0.0 --port 8000
Endpoint: POST /predict (multipart file upload) -> JSON {is_object, bbox}
"""
import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import transforms
from torchvision.models import mobilenet_v2
from torch import nn
from PIL import Image
import uvicorn


class ObjectDetectionModel(nn.Module):
    def __init__(self, freeze_until: int = 14):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = mobilenet_v2(weights="DEFAULT").features

        for i, block in enumerate(self.backbone):
            if i < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False

        # HARUS identik dengan arsitektur di notebook training (v3, split head) --
        # kalau training diganti lagi, update juga di sini, atau load_state_dict akan gagal.
        self.objectness_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        features = self.backbone(x)
        is_object = torch.sigmoid(self.objectness_head(features))
        bbox = torch.sigmoid(self.bbox_head(features))
        return is_object, bbox


app = FastAPI(title="Single Object Detection API")
model = ObjectDetectionModel()
model.load_state_dict(torch.load("best.pt", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            is_object, bbox = model(image_tensor)

        is_object_value = is_object.item()
        bbox = bbox.squeeze(0).tolist()  # (x, y, w, h) ternormalisasi [0,1]

        if is_object_value > 0.5:
            response = {"is_object": True, "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}}
        else:
            response = {"is_object": False, "bbox": None}
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def root():
    return {"message": "Single Object Detection API is running!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
