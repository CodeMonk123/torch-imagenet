import torch
import os
import json

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

class SpeedMerter(object):
    def __init__(self, is_master:bool) -> None:
        self._is_master = is_master
        if not is_master:
            return
        pod_name = os.getenv('HOSTNAME')
        self.job_config = os.getenv('JOB_CONFIG')
        ddlp_dir = '/ddlp'
        # ddlp_dir = '/home/czh/'
        pod_dir = os.path.join(ddlp_dir, pod_name)
        if not os.path.exists(pod_dir):
            os.makedirs(pod_dir,exist_ok=True)

        self.speed_file_path = os.path.join(pod_dir, 'speed.json')
        self.config_file_path = os.path.join(pod_dir, 'config.json')
        self.speed = []
    
    def update(self, val):
        if self._is_master:
            self.speed.append(val)
    
    def output(self):
        if self._is_master:
            with open(self.speed_file_path, mode='w') as fp:
                fp.write(str(self.speed))
            if self.job_config is not None:
                with open(self.config_file_path, mode='w') as fp:
                    fp.write(self.job_config)
                    fp.write('\n')
                
    


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res