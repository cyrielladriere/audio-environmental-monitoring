import os
from thop import profile
import torch
import time
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = {'Bark': 0, 'Motorcycle': 1, 'Writing': 2, 'Female_speech_and_woman_speaking': 3, 'Tap': 4, 'Child_speech_and_kid_speaking': 5, 'Screaming': 6, 'Meow': 7, 'Scissors': 8, 'Fart': 9, 'Car_passing_by': 10, 'Harmonica': 11, 'Sink_(filling_or_washing)': 12, 'Burping_and_eructation': 13, 'Slam': 14, 'Drawer_open_or_close': 15, 'Cricket': 16, 'Hiss': 17, 'Frying_(food)': 18, 'Sneeze': 19, 'Chink_and_clink': 20, 'Fill_(with_liquid)': 21, 'Crowd': 22, 'Marimba_and_xylophone': 23, 'Sigh': 24, 'Accordion': 25, 'Electric_guitar': 26, 'Cupboard_open_or_close': 27, 'Bicycle_bell': 28, 'Waves_and_surf': 29, 'Stream': 30, 'Bus': 31, 'Toilet_flush': 32, 'Trickle_and_dribble': 33, 'Tick-tock': 34, 'Keys_jangling': 35, 'Acoustic_guitar': 36, 'Finger_snapping': 37, 'Cheering': 38, 'Race_car_and_auto_racing': 39, 'Bass_guitar': 40, 'Yell': 41, 'Water_tap_and_faucet': 42, 'Run': 43, 'Traffic_noise_and_roadway_noise': 44, 'Crackle': 45, 'Skateboard': 46, 'Glockenspiel': 47, 'Computer_keyboard': 48, 'Whispering': 49, 'Zipper_(clothing)': 50, 'Microwave_oven': 51, 'Bathtub_(filling_or_washing)': 52, 'Male_speech_and_man_speaking': 53, 'Gong': 54, 'Shatter': 55, 'Strum': 56, 'Bass_drum': 57, 'Dishes_and_pots_and_pans': 58, 'Accelerating_and_revving_and_vroom': 59, 'Male_singing': 60, 'Gurgling': 61, 'Walk_and_footsteps': 62, 'Printer': 63, 'Cutlery_and_silverware': 64, 'Chirp_and_tweet': 65, 'Clapping': 66, 'Hi-hat': 67, 'Raindrop': 68, 'Gasp': 69, 'Buzz': 70, 'Drip': 71, 'Chewing_and_mastication': 72, 'Squeak': 73, 'Female_singing': 74, 'Church_bell': 75, 'Mechanical_fan': 76, 'Purr': 77, 'Applause': 78, 'Knock': 79}

def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int32)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool_)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float64)))
    return pos_class_indices, precision_at_hits

def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    labels = []
    for label in truth:
        one_hot_vector = np.zeros(len(classes), dtype=int)
        one_hot_vector[label] = 1
        labels.append(one_hot_vector)

    # print(f"labels: {labels[0]}")
    # print(f"scores: {scores[0]}")
    truth = np.array(labels)
    scores = np.array(scores)
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class

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

    # macs, params = profile(mdl, inputs=(torch.randn(1, 1, 256, 128))) #image size: (256,128)
    # print(f"MACs(G): {macs}")

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
