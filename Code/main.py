import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from models import ResnetModel
from train import Trainer
from dataset import SkinCancerDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_to_train_data = ''
    path_to_val_data = ''
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   
    ])
    #Load dataset
    dataset = SkinCancerDataset(hdf5_image_path='D:\Tài liệu\Project\Skin Cancer Detection\\train-image.hdf5',
                                metadata_path='D:\Tài liệu\Project\Skin Cancer Detection\\train-metadata.csv',
                                transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize ResNet101 model
    model = ResnetModel()

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        lr=1e-3,
        weight_decay=1e-5,
        device=device
    )

    # Train the model
    trainer.train(epochs=10)

    # Save the trained model
    trainer.save_checkpoint()
    
if __name__ == "__main__":
    main()