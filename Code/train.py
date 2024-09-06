import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

class Trainer:
    def __init__(self, model, dataloader, lr=1e-3, weight_decay=1e-5, device='cuda'):
        # Use GPU if available; otherwise, use CPU
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)  # Move the model to the specified device
        
        self.images = []
        self.metadata = []
        self.targets = []
        self.dataloader = dataloader
        (self.image_train,self.image_val,self.target_train,self.target_val) = self.train_val_split(dataloader)
        
        
        # Optimizer for updating model parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # Using MSELoss for regression instead of BCELoss
        self.criterion = nn.MSELoss()
        
        # History dictionary to store training and validation losses
        self.history = {'train_loss': [], 'val_loss': [], 'val_pauc': []}
        
    def train_val_split(self,dataset):
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for image,metadata,target in dataset:
            self.images.append(image)
            self.metadata.append(metadata)
            self.targets.append(target)
        for train_index, val_index in split.split(self.images, self.targets):
            image_train, image_val = self.images[train_index], self.images[val_index]
            target_train, target_val = self.target[train_index], self.targets[val_index]
        return (image_train,image_val,target_train,target_val)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_pauc = self._validate() if self.val_loader else (None, None)
            
            self.history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_pauc'].append(val_pauc)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}" +
                  (f" | Val Loss: {val_loss:.4f} | Val pAUC: {val_pauc:.4f}" if val_loss is not None else ""))
    
    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for (images, targets) in (self.image_train,self.target_train):
            images, targets = images.to(self.device), targets.to(self.device)
            
            #Forwass pass
            outputs = self.model(images)
            loss = self.criterion(outputs,targets.float())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() + images.size(0)
        
        return running_loss/len(self.train_loader.dataset)
    
    def validate(self):
        self.model.eval()
        running_loss = 00
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for images, targets in (self.image_val,self.target_val):
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                running_loss+=loss.item()*images.size(0)
                
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        pauc = roc_auc_score(all_targets,all_outputs,max_fpr=0.1)
        return running_loss/len(self.val_loader.dataset), pauc
    
    def test(self):
        if not self.test_loader:
            print("No test dataset provided")
            return
        self.model.eval()
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                                
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        pauc = roc_auc_score(all_targets,all_outputs,max_fpr=0.1)
        print(f"Test pAUC: {pauc:.4f}")
        return
    
    def save_checkpoint(self, path = 'Code/model.pth'):
        torch.save(self.model.state_dict(), path)
    def load_checkpoint(self, path='Code/model.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)        