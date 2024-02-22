import torch
import numpy as np
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# def acc(y_true, y_hat):
#     y_hat = torch.argmax(y_hat, dim=-1)
#     tot = y_true.shape[0]
#     hit = torch.sum(y_true == y_hat)
#     return hit.data.float() * 1.0 / tot
# accuracy = evaluate.load("accuracy")

# def compute_metrics(*eval_pred):
#     predictions, labels = eval_pred
#     predictions, labels = predictions.detach().cpu().numpy(), labels.detach().cpu().numpy()
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)
def compute_metrics(*eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(predictions, dim=-1)
    tot = labels.shape[0]
    acc = torch.sum(predictions==labels)
    return acc.data.float() * 1.0/tot