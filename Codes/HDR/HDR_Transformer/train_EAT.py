import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import  transforms
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
import random
import json
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler



from imageloader import SIG17_Training_Dataset,SIG17_Validation_Dataset
from EAT import MODEL, DIS
from loss_EAT import CharbonnierLoss_loss, d_fake_fun, d_real_fun, gradient_penalty,ssim_loss,compute_metrics
from utils import tensor_torch


#Seed
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

#### Hyperparemetr Details ######

'''print(torch.cuda.device_count())  # Number of available GPUs
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")'''
# Set specific GPUs (modify this based on available GPUs)
device_ids = [0]  # Use GPU 0 and 1 (modify as needed)
device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
# Load JSON from a file
with open("params.json", "r") as file:
    params = json.load(file)



### model uploading #######
net = MODEL(in_channel=18, output_channel=3)
discri = DIS(in_channel=3, output_channel=3)

# Wrap model with DataParallel for multiple GPUs
#net = torch.nn.DataParallel(net)
net.to(device)
discri.to(device)
net.load_state_dict(torch.load('./Results/EAT.pkl', map_location='cuda'))
discri.load_state_dict(torch.load('./Results/EAT_disc.pkl', map_location='cuda'))


trainset = SIG17_Training_Dataset(root_dir=params['data_dir'], sub_set=params['sub_set'], is_training=True)
# Define training sampler (without batch_size)
train_sampler = RandomSampler(trainset)

# Create DataLoader for training
trainloader = DataLoader(trainset, batch_size=params['train_batch_size'], sampler=train_sampler,drop_last=True, num_workers=params['num_workers'])
print(len(trainset))

test_set = SIG17_Validation_Dataset(root_dir=params['data_dir'], is_training=False, crop=True, crop_h=1000, crop_w=1000)
test_sampler = SequentialSampler(test_set)
test_dl = DataLoader(test_set, batch_size=params['eval_batch_size'],  sampler=test_sampler,  num_workers=params['num_workers'])

print(len(test_set))
### loss functions
optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
optimizerG = optim.Adam(net.parameters(), lr=params['learning_rate'], betas=(0.9, 0.999), eps=1e-8)
optimizerD = optim.Adam(discri.parameters(), lr=params['learning_rate'], betas=(0.9, 0.999), eps=1e-8)
milestones = [100, 150]
scheduler = MultiStepLR(optimizer, milestones, 0.5)

transform = transforms.Compose([
    transforms.ToTensor(),
]) 
train_loss_dict = {}
disc_loss_dict ={}
val_loss_dict = {}
psnr_dict ={}
psnr_mu_dict = {}
epochs = params['num_epochs']
best_psnr = 20
for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch+1}/{epochs}")
    
    running_loss = 0.0
    running_disc_loss =0.0
    ranking_count_full = 0
    
    net.train()  # Set model to training mode
    
    # Training loop
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs = tensor_torch(data)  # Convert inputs to tensors
        #print(inputs['label'].shape)
        optimizer.zero_grad()  # Clear previous gradients
        out = net(inputs)
        #print(out.shape)
        d_fake=discri(out)

        GT = inputs['label']
        d_real =discri(GT)


        alpha = torch.rand(GT.size(0), 1, 1, 1).cuda()
        img_hat = (alpha * GT.data + (1 - alpha) * out.data).requires_grad_(True)
        d_hat_decision =discri(img_hat)


        loss_CharbonnierLoss =  CharbonnierLoss_loss(out,GT)

        loss_ssimLoss=  ssim_loss(out,GT)

        loss_d_fake = d_fake_fun(d_fake)

        loss = loss_CharbonnierLoss + loss_ssimLoss - 1 *loss_d_fake

        running_loss += loss.item()  # Accumulate loss

        d_fake = discri(out.detach())
        loss_d_fake = d_fake_fun(d_fake)
        loss_d_real = - d_real_fun(d_real)
        loss_d_gp = gradient_penalty(d_hat_decision, img_hat)

        lossD = loss_d_real+ loss_d_fake+ 10* loss_d_gp

        running_disc_loss += lossD.item()  # Accumulate loss
        
        optimizerG.zero_grad()
        loss.backward()
        optimizerG.step()


        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        
        ranking_count_full += 1

    train_loss_dict[epoch] = running_loss / ranking_count_full
    disc_loss_dict[epoch] = running_disc_loss/ ranking_count_full
    print(f"Training Loss at Epoch {epoch+1}: {train_loss_dict[epoch]:.6f}")

    #scheduler.step()  # Adjust learning rate
    
    # ------------------- Evaluation Phase -------------------
    net.eval()  # Set model to evaluation mode
    eval_loss = 0.0
    psnr_eval_metric =0.0
    psnr_mu_eval_metric = 0.0
    eval_count = 0

    with torch.no_grad():  # Disable gradient computation
        for i, data in tqdm(enumerate(test_dl), total=len(test_dl)):
            inputs = tensor_torch(data)  # Convert inputs to tensors
            outputs = net(inputs)  # Forward pass
            
            #loss = compute_losses(inputs, outputs, params)  # Compute loss
            metrics = compute_metrics(inputs, outputs, params)
            eval_loss += loss.item()
            psnr_eval_metric +=metrics['psnr'].item()
            #psnr_mu_eval_metric +=metrics['psnr_mu'].item()
            eval_count += 1

    #val_loss_dict[epoch] = eval_loss / eval_count
    psnr_dict[epoch] = psnr_eval_metric / eval_count
    #psnr_mu_dict[epoch] = psnr_mu_eval_metric / eval_count
    #print(f"Validation Loss at Epoch {epoch+1}: {val_loss_dict[epoch]:.6f}")
    print(f"Validation PSNR at Epoch {epoch+1}: {psnr_dict[epoch]:.6f}")
    #print(f"Validation PSNR-mu at Epoch {epoch+1}: {psnr_mu_dict[epoch]:.6f}")

    

    #Saving logs

    dataStorePath = './Results/'

    train_loss_path = os.path.join(dataStorePath, 'EAT_train_loss.pkl')
    with open(train_loss_path, 'wb') as handle:
        pickle.dump(train_loss_dict, handle)
        print("Saving train loss log to ", train_loss_path)
    
    

    psnr_eval_path = os.path.join(dataStorePath, 'EAT_psnr_eval.pkl')
    with open(psnr_eval_path, 'wb') as handle:
        pickle.dump(psnr_dict, handle)
        print("Saving train loss log to ", psnr_eval_path)

    psnr_mu_eval_path = os.path.join(dataStorePath, 'EAT_psnr_mu_eval.pkl')
    with open(psnr_mu_eval_path, 'wb') as handle:
        pickle.dump(psnr_mu_dict, handle)
        print("Saving train loss log to ", psnr_mu_eval_path)

    model_save_path = os.path.join(dataStorePath, f'EAT.pkl')
    disc_model_save_path = os.path.join(dataStorePath, f'EAT_disc.pkl')
    if psnr_dict[epoch] > best_psnr:
        best_psnr = psnr_dict[epoch]
        torch.save(net.state_dict(), model_save_path)
        torch.save(discri.state_dict(), disc_model_save_path)
        print('------------------model saved-----------------------')#
    print('------------------------------------------------------------------------------------------------')
