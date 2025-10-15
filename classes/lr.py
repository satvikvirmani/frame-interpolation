
from math import cos, pi

def cosine_lr(epoch, total_epochs, initial_lr, min_lr=1e-6):
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * epoch / total_epochs))