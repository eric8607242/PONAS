import time
import torch

def calculate_latency(model, input_depth, input_size):
    input_sample = torch.randn((1, input_depth, input_size, input_size))
    end = time.time()
    model(input_sample)
    inference_time = time.time() - end
    
    return inference_time

def calculate_param_nums(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
