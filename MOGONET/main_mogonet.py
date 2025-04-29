""" Example for MOGONET classification
"""
from train_test import train_test
import torch

# from utils import cosine_distance_torch, rbf_distance_torch

# def normalize_data(data):
#     mean = torch.mean(data, dim=0, keepdim=True)
#     std = torch.std(data, dim=0, keepdim=True)
#     return (data - mean) / (std + 1e-8)

# def hybrid_similarity_torch(x1, x2=None, alpha=0.5):
#     x2 = x1 if x2 is None else x2
#     cosine_sim = cosine_distance_torch(x1, x2)
#     rbf_sim = rbf_distance_torch(x1, x2, gamma=0.01)
#     return alpha * cosine_sim + (1 - alpha) * rbf_sim

# def adaptive_rbf_distance_torch(x1, x2=None):
#     x2 = x1 if x2 is None else x2
#     # Compute median pairwise distance
#     x1_norm = (x1 ** 2).sum(dim=1).view(-1, 1)
#     x2_norm = x1_norm if x2 is x1 else (x2 ** 2).sum(dim=1).view(1, -1)
#     dist_squared = x1_norm + x2_norm - 2.0 * torch.mm(x1, x2.T)
#     median_dist = torch.median(dist_squared[dist_squared > 0])
#     gamma = 1.0 / (2.0 * median_dist)  # Adaptive gamma
#     sim = torch.exp(-gamma * dist_squared)
#     print("Sim: ",sim)
#     return 1 - sim

if __name__ == "__main__":    
    data_folder = 'ROSMAP'
    view_list = [1,2,3]
    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 1e-3  
    lr_e = 5e-4
    lr_c = 1e-3
    
    if data_folder == 'ROSMAP':
        num_class = 2
        # adj_parameter = 5  # Try different values
        # dim_he_list = [200,200,100]
    if data_folder == 'BRCA':
        num_class = 5
    
    adj_metric = "cosine"
    
    train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch)             