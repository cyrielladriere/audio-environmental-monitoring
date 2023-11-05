import os
import torch
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        fmtstr = f'{self.name} {self.val}' # ({avg' + self.fmt + '})'
        return fmtstr

def print_model_size(mdl):
    """Prints the size of the given model in Megabytes"""
    torch.save(mdl.state_dict(), "tmp.pt")
    print("Size: ", end="")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

    total_params = sum(param.numel() for param in mdl.parameters())
    print(f"Number of parameters: {total_params}")

    trainable_params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = torch.topk(output, maxk, dim=1, largest=True)
        pred = pred.t()     #shape: [maxk, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, data_loader, classes):#, neval_batches):
    start_time = time.time()
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    with torch.no_grad():
        for image, target in data_loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image)     # shape: [batch_size, num_classes]

            # Convert target to tensor of class numbers before giving to accuracy
            # target = torch.tensor([classes[item] for item in target]).to(device)    # shape: [batch_size]

            acc1, acc5 = accuracy(output, target, topk=(1, 2))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    end_time = time.time()
    inference_time = end_time - start_time
    return top1, top5, inference_time

"""
accuracy with array as input
with torch.no_grad():
        maxk = max(topk)
        batch_size = output.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        target = np.expand_dims(target, axis=0)
        target = np.tile(target, pred.shape)
        print(f"target: {target}")
        print(f"pred: {pred}")
        correct = pred.eq(target)   #.view(1, -1).expand_as(pred))   

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
"""
