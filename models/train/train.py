# models/train/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from models.core.model import EnzoModel
from models.data.dataloader import get_dataloader

def train():
    # Load Config
    with open("models/config/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'])
    
    # Initialize Model, Data, Loss, and Optimizer
    model = EnzoModel(
        config['model']['input_dim'],
        config['model']['hidden_dim'],
        config['model']['output_dim']
    ).to(device)
    
    dataloader = get_dataloader(config['training']['batch_size'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])

    # Training Loop
    model.train()
    for epoch in range(config['training']['epochs']):
        total_loss = 0
        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_data)
            loss = criterion(predictions, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Loss: {total_loss/len(dataloader):.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), "models/checkpoints/enzo_v1.pt")
    print("Training complete. Model saved to models/checkpoints/enzo_v1.pt")

if __name__ == "__main__":
    train()