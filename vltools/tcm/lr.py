import math
import numpy as np

class CosAnnealingLR(object):

    def __init__(self, iterations, lr_max, lr_min=0, warmup_iters=0):

        assert lr_max >= 0
        assert warmup_iters >= 0
        assert iterations >= 0 and iterations >= warmup_iters

        self.iterations = iterations
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_iters = warmup_iters

        self.counter = 0

        self.last_lr = 0
    
    def restart(self, lr_max=None):

        if lr_max:
            self.lr_max = lr_max
        self.counter = 0

    def step(self):

        if self.warmup_iters > 0 and self.counter < self.warmup_iters:

            self.last_lr = float(self.counter / self.warmup_iters) * self.lr_max

        else:
            self.last_lr = (1 + math.cos((self.counter-self.warmup_iters) / \
                                    (self.iterations - self.warmup_iters) * math.pi)) / 2 * self.lr_max

        self.counter += 1

        return self.last_lr
                           

if __name__ == "__main__":
    max_epochs = 20
    iters_per_epoch = 10
    warmup_epochs = 2

    lr_scheduler = CosAnnealingLR(max_epochs*iters_per_epoch, lr_max=0.1, lr_min=0.01,
                                  warmup_iters=warmup_epochs*iters_per_epoch)
    lr_record = []
    for i in range(max_epochs*iters_per_epoch):
        lr_record.append(lr_scheduler.step())

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
    axes[0].plot(lr_record, color="blue", linewidth=2)
    axes[0].set_xlabel("Iter", fontsize=12)
    axes[0].set_ylabel("Learning Rate", fontsize=12)
    axes[0].set_title("Warmup + Cosine Annealing")

    lr_scheduler.restart(lr_max=0.05)
    for i in range(max_epochs*iters_per_epoch):
        lr_record.append(lr_scheduler.step())
    lr_scheduler.restart(lr_max=0.025)
    for i in range(max_epochs*iters_per_epoch):
        lr_record.append(lr_scheduler.step())
    axes[1].plot(lr_record, color="blue", linewidth=2)
    axes[1].set_xlabel("Iter", fontsize=12)
    axes[1].set_ylabel("Learning Rate", fontsize=12)
    axes[1].set_title("Warmup + Cosine Annealing + Restart")
    plt.tight_layout()
    plt.show()
    

        


