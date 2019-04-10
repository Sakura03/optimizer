class DayHourMinute(object):
  
  def __init__(self, seconds):
      
      self.days = int(seconds // 86400)
      self.hours = int((seconds- (self.days * 86400)) // 3600)
      self.minutes = int((seconds - self.days * 86400 - self.hours * 3600) // 60)

def get_lr(epoch, base_lr, warmup_epochs=5, warmup_start_lr=0.001):
    lr = 0
    if epoch < warmup_epochs:
        lr = ((base_lr - warmup_start_lr) / warmup_epochs) * epoch
    else:
        lr = base_lr * (0.1 ** ((epoch-warmup_epochs) // 30))
    return lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count