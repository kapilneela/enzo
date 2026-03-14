# models/api/enzo_model_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import yaml
from models.core.model import EnzoModel

app = FastAPI(title="ENZO-1 API")

# Load config and model globally for the API
with open("models/config/default.yaml", "r") as f:
    config = yaml.safe_load(f)

model = EnzoModel(
    config['model']['input_dim'],
    config['model']['hidden_dim'],
    config['model']['output_dim']
)
model.load_state_dict(torch.load("models/checkpoints/enzo_v1.pt", weights_only=True))
model.eval()

class InferenceRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: InferenceRequest):
    # Convert input list to tensor
    input_tensor = torch.tensor([request.features], dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        
    return {"prediction": prediction}