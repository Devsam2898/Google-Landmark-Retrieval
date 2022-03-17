# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import timm

def criterion(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, labels) in bar:         
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / CONFIG['n_accumulate']
            
        scaler.scale(loss).backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            for p in model.parameters():
                p.grad = None

            if scheduler is not None:
                scheduler.step()
                
               @torch.no_grad()
              
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    TARGETS = []
    PREDS = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, labels) in bar:        
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        
        batch_size = images.size(0)
        
        outputs = model(images)
        _, preds = torch.max(outputs,1)
        loss = criterion(outputs, labels)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss/dataset_size
        
        PREDS.append(preds.view(-1).cpu().detach().numpy())
        TARGETS.append(labels.view(-1).cpu().detach().numpy())
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])   
    
    TARGETS = np.concatenate(TARGETS)
    PREDS = np.concatenate(PREDS)
    val_acc = accuracy_score(TARGETS, PREDS)
    gc.collect()
    
    return epoch_loss, val_acc
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss/dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss
  def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_acc = 0
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        
        val_epoch_loss, val_epoch_acc = valid_one_epoch(model, valid_loader, 
                                                        device=CONFIG['device'], 
                                                        epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Valid Acc'].append(val_epoch_acc)
        
        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})
        wandb.log({"Valid Acc": val_epoch_acc})
        
        print(f'Valid Acc: {val_epoch_acc}')
        
        # deep copy the model
        if val_epoch_acc >= best_epoch_acc:
            print(f"{c_}Validation Acc Improved ({best_epoch_acc} ---> {val_epoch_acc})")
            best_epoch_acc = val_epoch_acc
            run.summary["Best Accuracy"] = best_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "ACC{:.4f}_epoch{:.0f}.bin".format(best_epoch_acc, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            wandb.save(PATH)
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best ACC: {:.4f}".format(best_epoch_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history
  def prepare_loaders():    
    train_dataset = LandmarkDataset(TRAIN_DIR, df_train, transforms=data_transforms['train'])
    valid_dataset = LandmarkDataset(TRAIN_DIR, df_valid, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=4, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=4, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader
 def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG['T_0'], 
                                                             T_mult=1, eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler
  train_loader, valid_loader = prepare_loaders()
  optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
  scheduler = fetch_scheduler(optimizer)
  
  model, history = run_training(model, optimizer, scheduler,
                              device=CONFIG['device'],
                              num_epochs=CONFIG['epochs'])
