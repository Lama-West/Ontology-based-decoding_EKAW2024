import torch

def log_to_prob_stable(log_probs):
    return torch.exp(log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True))

def kl_divergence(input: torch.Tensor, target: torch.Tensor):
    """
    Input and target must be probability distributions (not in log space)
    """
    
    # Normalize input using softmax
    probs1 = torch.nn.functional.softmax(input, dim=-1)
    
    # Compute KL divergence
    kl_div = torch.nn.functional.kl_div(probs1.log(), target, reduction='sum')
    
    # Check if KL divergence is below threshold
    return kl_div.item()
