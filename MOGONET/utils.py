import os
import numpy as np
import torch
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False

# def normalize_data(data):
#     mean = torch.mean(data, dim=0, keepdim=True)
#     std = torch.std(data, dim=0, keepdim=True)
#     return (data - mean) / (std + 1e-8)
def normalize_data(data):
    mean = torch.mean(data, dim=0, keepdim=True)
    std = torch.std(data, dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-8)
    return (data - mean) / std

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels==i)[0]] = count[i]/np.sum(count)
    
    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    
    return y_onehot



def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def rbf_kernel(x1, x2=None, gamma=None):
    """
    Radial Basis Function (Gaussian) Kernel as similarity.
    """
    print("From rbf_kernel")
    x2 = x1 if x2 is None else x2
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # [N, M, D]
    dist_sq = torch.sum(diff ** 2, dim=2)     # [N, M]
    if gamma is None:
        gamma = 1.0 / x1.shape[1]             # Default: 1 / num_features
    sim = torch.exp(-gamma * dist_sq)
    return sim


def adaptive_rbf_distance_torch(x1, x2=None):
    print("From adaptive_rbf_distance_torch")
    x2 = x1 if x2 is None else x2
    # Compute median pairwise distance
    x1_norm = (x1 ** 2).sum(dim=1).view(-1, 1)
    x2_norm = x1_norm if x2 is x1 else (x2 ** 2).sum(dim=1).view(1, -1)
    dist_squared = x1_norm + x2_norm - 2.0 * torch.mm(x1, x2.T)
    median_dist = torch.median(dist_squared[dist_squared > 0])
    gamma = 1.0 / (2.0 * median_dist)  # Adaptive gamma
    sim = torch.exp(-gamma * dist_squared)
    print("Sim: ",sim)
    return 1 - sim

# def cosine_distance_torch(x1, x2=None, eps=1e-8):
#     print("From cosine_distance_torch")
#     x2 = x1 if x2 is None else x2
#     w1 = x1.norm(p=2, dim=1, keepdim=True)
#     w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
#     return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    res = 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    return res

def rbf_distance_torch(x1, x2=None, gamma=None):
    """
    Computes RBF-based pairwise distances between samples.
    The smaller the Euclidean distance, the higher the similarity (max=1).
    """
    print("From rbf_distance_torch")
    x2 = x1 if x2 is None else x2
    x1_norm = (x1 ** 2).sum(dim=1).view(-1, 1)
    x2_norm = x1_norm if x2 is x1 else (x2 ** 2).sum(dim=1).view(1, -1)
    
    dist_squared = x1_norm + x2_norm - 2.0 * torch.mm(x1, x2.T)
    dist_squared = torch.clamp(dist_squared, min=0.0)  # numerical safety
    
    if gamma is None:
        median_dist = torch.median(dist_squared[dist_squared > 0])
        gamma = 1.0 / (2.0 * median_dist)

    sim = torch.exp(-gamma * dist_squared)  # RBF similarity
    res = 1 - sim  # Return distance (small similarity -> larger distance)
    return res


def hybrid_similarity_torch(x1, x2=None, alpha=0.5, gamma=0.01):
    print("Hybrid")
    x2 = x1 if x2 is None else x2
    
    # Calculate cosine similarity
    cosine_sim = cosine_distance_torch(x1, x2)
    
    # Normalize cosine similarity to be in range [0, 1]
    cosine_sim = torch.clamp(1 - cosine_sim, 0, 1)  # Cosine similarity is between 0 and 1
    print("COS: ")
    print(cosine_sim[:5, :5])
    # Calculate RBF similarity
    rbf_sim = rbf_distance_torch(x1, x2, gamma=gamma)
    
    # Normalize RBF similarity to be in range [0, 1] (if not already)
    rbf_sim = torch.clamp(rbf_sim, 0, 1)
    print("RBF: ")
    print(rbf_sim[:5, :5])
    alpha = cosine_sim.var() / (cosine_sim.var() + rbf_sim.var() + 1e-8)
    print("Alpha: ", alpha)
    # Combine similarities
    hybrid_sim = alpha * cosine_sim + (1 - alpha) * rbf_sim
    
    # Ensure the hybrid similarity is within the range [0, 1]
    hybrid_sim = torch.clamp(hybrid_sim, 0, 1)
    hybrid_sim = F.normalize(hybrid_sim, p=1, dim=1)

    print("\nHybrid Similarity Matrix Shape:", hybrid_sim.shape)
    print("Hybrid Similarity Values (first 5x5):")
    print(hybrid_sim[:5, :5])
    
    return hybrid_sim

# def hybrid_similarity_torch(x1, x2=None, alpha=0.5, gamma=0.01):
#     print("Hybrid")
#     x2 = x1 if x2 is None else x2
    
#     # Calculate cosine similarity
#     cosine_sim = cosine_distance_torch(x1, x2)
    
#     # Calculate RBF similarity
#     rbf_sim = rbf_distance_torch(x1, x2, gamma=gamma)
    
#     # Print similarity values
#     print("\nCosine Similarity Matrix Shape:", cosine_sim.shape)
#     print("Cosine Similarity Values (first 5x5):")
#     print(cosine_sim[:5, :5])
    
#     print("\nRBF Similarity Matrix Shape:", rbf_sim.shape)
#     print("RBF Similarity Values (first 5x5):")
#     print(rbf_sim[:5, :5])
    
#     # Combine similarities
#     hybrid_sim = alpha * cosine_sim + (1 - alpha) * rbf_sim
    
#     print("\nHybrid Similarity Matrix Shape:", hybrid_sim.shape)
#     print("Hybrid Similarity Values (first 5x5):")
#     print(hybrid_sim[:5, :5])
    
#     return hybrid_sim




# def cal_adj_mat_parameter(edge_per_node, data, metric="rbf"):
# def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
# def cal_adj_mat_parameter(edge_per_node, data, metric="adaptive_rbf"):
def cal_adj_mat_parameter(edge_per_node, data, metric="hybrid"):
    assert metric in ["rbf", "cosine", "adaptive_rbf", "hybrid"], "Only rbf, cosine, adaptive_rbf, and hybrid supported"

    if metric == "rbf":
        dist = rbf_distance_torch(data, data, gamma=0.01)
    elif metric == "cosine":
        dist = cosine_distance_torch(data, data)
    elif metric == "adaptive_rbf":
        dist = adaptive_rbf_distance_torch(data, data)
    elif metric == "hybrid":
        dist = hybrid_similarity_torch(data, data, alpha=0.5, gamma=0.01)

    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node*data.shape[0]]
    return np.isscalar(parameter.data.cpu().numpy())
    #return parameter.item()  # Return the scalar value


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
    return g

def graph_from_topk_tensor(similarity, k):
    topk_values, topk_indices = torch.topk(similarity, k=k, dim=-1)
    mask = torch.zeros_like(similarity)
    mask.scatter_(1, topk_indices, 1)
    return mask


# def gen_adj_mat_tensor(data, parameter, metric="rbf"):
# def gen_adj_mat_tensor(data, parameter, metric="cosine"):
# def gen_adj_mat_tensor(data, parameter, metric="adaptive_rbf"):
def gen_adj_mat_tensor(data, parameter, metric="hybrid"):
    assert metric in ["rbf", "cosine", "adaptive_rbf", "hybrid"], "Only rbf, cosine, adaptive_rbf, and hybrid supported"

    # Normalize data before computing similarity
    data = normalize_data(data)

    if metric == "rbf":
        dist = rbf_distance_torch(data, data, gamma=0.01)
    elif metric == "cosine":
        dist = cosine_distance_torch(data, data)
    elif metric == "adaptive_rbf":
        dist = adaptive_rbf_distance_torch(data, data)
    elif metric == "hybrid":
        dist = hybrid_similarity_torch(data, data, alpha=0.5, gamma=0.01)

    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric in ["rbf", "cosine", "adaptive_rbf", "hybrid"]:
        adj = 1-dist
    else:
        adj = 1-dist
    adj = adj*g 
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    
    # Convert to dense tensor before returning
    adj = adj.to_dense()
    
    return adj

# def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="rbf"):
# def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
# def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="adaptive_rbf"):
def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="hybrid"):
    assert metric in ["rbf", "cosine", "adaptive_rbf", "hybrid"], "Only rbf, cosine, adaptive_rbf, and hybrid supported"

    # Normalize data before computing similarity
    data = normalize_data(data)

    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])
    
    if metric == "rbf":
        dist_tr2te = rbf_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]], gamma=0.01)
    elif metric == "cosine":
        dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    elif metric == "adaptive_rbf":
        dist_tr2te = adaptive_rbf_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    elif metric == "hybrid":
        dist_tr2te = hybrid_similarity_torch(data[trte_idx["tr"]], data[trte_idx["te"]], alpha=0.5)

    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    adj[:num_tr,num_tr:] = 1-dist_tr2te
    adj[:num_tr,num_tr:] = adj[:num_tr,num_tr:]*g_tr2te
    
    if metric == "rbf":
        dist_te2tr = rbf_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]], gamma=0.01)
    elif metric == "cosine":
        dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    elif metric == "adaptive_rbf":
        dist_te2tr = adaptive_rbf_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    elif metric == "hybrid":
        dist_te2tr = hybrid_similarity_torch(data[trte_idx["te"]], data[trte_idx["tr"]], alpha=0.5)

    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    adj[num_tr:,:num_tr] = 1-dist_te2tr
    adj[num_tr:,:num_tr] = adj[num_tr:,:num_tr]*g_te2tr
    
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    
    # Convert to dense tensor before returning
    adj = adj.to_dense()
    
    return adj


def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module+".pth"))
            
    
def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pth")):
#            print("Module {:} loaded!".format(module))
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pth"), map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()    
    return model_dict