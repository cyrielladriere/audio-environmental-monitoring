import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}

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

def print_model_size(mdl):
    """Prints the size of the given model in Megabytes"""
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = torch.topk(output, maxk, dim=1, largest=True)
        print(f"pred: {pred}")
        print(f"target: {target}")
        pred = pred.t()     #shape: [maxk, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, data_loader):#, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            image_batch = sample['image'].permute(0, 3, 1, 2).float().to(device)         # change shape: [batch_size, height, width, channels] -> [batch_size, channels, height, width]
            target = sample['label']
            output = model(image_batch)     # shape: [batch_size, num_classes]

            # Convert target to tensor of class numbers before giving to accuracy
            target = torch.tensor([classes[item] for item in target]).to(device)    # shape: [batch_size]

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image_batch.size(0))
            top5.update(acc5[0], image_batch.size(0))
    return top1, top5

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
