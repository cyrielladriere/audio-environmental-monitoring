from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from compression.evaluation import print_model_size
from compression.models.PANN_pretrained import MobileNetV2
from compression.models.PANN_pruned import MobileNetV2_pruned
from compression.preprocessing import TrainDataset, convert_dataset, load_pkl, get_labels, convert_labels
from compression.pruning import import_pruned_weights, pruned_fine_tuning, save_pruned_layers
from compression.quantization import pann_qat_v1, pann_qat_v2, pann_sq
from compression.training import train_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------- Testing Env
TENSORBOARD = False
# ----
PREPROCESSING = False
MODEL_PANN = False
PANN_QAT = False
PANN_QAT_V2 = False
PANN_SQ = False         
OPNORM_PRUNING = True; P=0.5
L1_PRUNING = False;
# ------------- Variables
training_audio_data = "data/audio/train_curated"
val_audio_data = "data/audio/test"
training_audio_labels = "data/audio/train_curated.csv"
test_audio_labels = "data/audio/sample_submission.csv"
training_data = "data/train_curated"
val_data = "data/test"
model_pann = "resources/MobileNetV2.pth"
model_pann_trained = "resources/model_pann.pt"
model_at = "resources/mn10_as.pt"
# ------------- Hyperparameters
image_size = (256, 128)
threshold = 0.5
batch_size = 64
n_epochs = 100
n_classes = 80
classes = {'Bark': 0, 'Motorcycle': 1, 'Writing': 2, 'Female_speech_and_woman_speaking': 3, 'Tap': 4, 'Child_speech_and_kid_speaking': 5, 'Screaming': 6, 'Meow': 7, 'Scissors': 8, 'Fart': 9, 'Car_passing_by': 10, 'Harmonica': 11, 'Sink_(filling_or_washing)': 12, 'Burping_and_eructation': 13, 'Slam': 14, 'Drawer_open_or_close': 15, 'Cricket': 16, 'Hiss': 17, 'Frying_(food)': 18, 'Sneeze': 19, 'Chink_and_clink': 20, 'Fill_(with_liquid)': 21, 'Crowd': 22, 'Marimba_and_xylophone': 23, 'Sigh': 24, 'Accordion': 25, 'Electric_guitar': 26, 'Cupboard_open_or_close': 27, 'Bicycle_bell': 28, 'Waves_and_surf': 29, 'Stream': 30, 'Bus': 31, 'Toilet_flush': 32, 'Trickle_and_dribble': 33, 'Tick-tock': 34, 'Keys_jangling': 35, 'Acoustic_guitar': 36, 'Finger_snapping': 37, 'Cheering': 38, 'Race_car_and_auto_racing': 39, 'Bass_guitar': 40, 'Yell': 41, 'Water_tap_and_faucet': 42, 'Run': 43, 'Traffic_noise_and_roadway_noise': 44, 'Crackle': 45, 'Skateboard': 46, 'Glockenspiel': 47, 'Computer_keyboard': 48, 'Whispering': 49, 'Zipper_(clothing)': 50, 'Microwave_oven': 51, 'Bathtub_(filling_or_washing)': 52, 'Male_speech_and_man_speaking': 53, 'Gong': 54, 'Shatter': 55, 'Strum': 56, 'Bass_drum': 57, 'Dishes_and_pots_and_pans': 58, 'Accelerating_and_revving_and_vroom': 59, 'Male_singing': 60, 'Gurgling': 61, 'Walk_and_footsteps': 62, 'Printer': 63, 'Cutlery_and_silverware': 64, 'Chirp_and_tweet': 65, 'Clapping': 66, 'Hi-hat': 67, 'Raindrop': 68, 'Gasp': 69, 'Buzz': 70, 'Drip': 71, 'Chewing_and_mastication': 72, 'Squeak': 73, 'Female_singing': 74, 'Church_bell': 75, 'Mechanical_fan': 76, 'Purr': 77, 'Applause': 78, 'Knock': 79}

def main():
    if(PREPROCESSING):
        convert_dataset(pd.read_csv(training_audio_labels), training_audio_data, training_data) 
        # convert_dataset(pd.read_csv(test_audio_labels), val_audio_data, val_data)
        # data = load_pkl(training_data)
        # save_images(data, True)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    data = load_pkl(training_data)

    labels = get_labels(data.keys())

    labels = convert_labels(labels)

    data = train_test_split(list(data.values()), labels, test_size=0.2)
    x_trn, x_val, y_trn, y_val = data

    train_dataset = TrainDataset(x_trn, y_trn, transform)
    val_dataset = TrainDataset(x_val, y_val, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    if(MODEL_PANN):
        # Tensorboard
        today = datetime.now()
        date = today.strftime('%b%d_%y-%H-%M')
        model_dir = f"compression/runs/PANN/{date}"
        if TENSORBOARD: writer = SummaryWriter(model_dir)

        model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 527).to(device)
        pretrained_weights = torch.load(model_pann)["model"] # keys: {iteration: , model: }
        model.load_state_dict(pretrained_weights)

        print_model_size(model)

        # Initialize layers that are not frozen
        model.bn0 = nn.BatchNorm2d(128)
        model.fc1 = nn.Linear(in_features=1280, out_features=256, bias=True)    # out_features tested: 1024(pretty bad), 512(ok), 128(okok)
        model.fc_audioset = nn.Linear(256, n_classes, bias=True)
        
       
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        model.cuda()
        if TENSORBOARD:
            model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, n_epochs, data, threshold, batch_size, True, TENSORBOARD, writer)
            writer.flush()
            writer.close()

            torch.save(model.state_dict(), f"{model_dir}/model_pann.pt")
        else:
            model = train_model(model, dataloaders, optimizer, exp_lr_scheduler, n_epochs, data, threshold, batch_size, True, TENSORBOARD)

    elif(PANN_QAT):          
        model_qat = pann_qat_v1(TENSORBOARD, model_pann, n_classes, dataloaders, n_epochs, data, threshold, batch_size) 
    elif(PANN_QAT_V2):
        model_qat = pann_qat_v2(TENSORBOARD, model_pann_trained, dataloaders, n_epochs, data, threshold, batch_size) 
    elif(PANN_SQ):
        model_sq = pann_sq(model_pann_trained, dataloaders)
    elif(OPNORM_PRUNING):
        today = datetime.now()
        date = today.strftime('%b%d_%y-%H-%M')
        model_dir = f"compression/runs/OPNORM_PRUNING/{date}"
        if TENSORBOARD: writer = SummaryWriter(model_dir)

        model_pruned = MobileNetV2_pruned(P, 44100, 1024, 320, 64, 50, 14000, 80)
        model_original = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, post_training=True).to(device)
        pretrained_weights = torch.load(model_pann_trained)
        model_original.load_state_dict(pretrained_weights)

        print_model_size(model_original)

        save_pruned_layers()
        model_pruned = import_pruned_weights(model_original, model_pruned, P)
        
        if TENSORBOARD:
            torch.save(model_pruned.state_dict(), f"resources/model_pann_opnorm_{P}.pt")
            pruned_fine_tuning(model_pruned, P, model_dir, dataloaders, n_epochs, data, threshold, batch_size, TENSORBOARD, writer)
        else:
            pruned_fine_tuning(model_pruned, P, model_dir, dataloaders, n_epochs, data, threshold, batch_size, TENSORBOARD)
        print_model_size(model_pruned)

    elif(L1_PRUNING):
        today = datetime.now()
        date = today.strftime('%b%d_%y-%H-%M')
        model_dir = f"compression/runs/L1_PRUNING/{date}"
        if TENSORBOARD: writer = SummaryWriter(model_dir)

        model_pruned = MobileNetV2_pruned(P, 44100, 1024, 320, 64, 50, 14000, 80)
        model_original = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, post_training=True).to(device)
        pretrained_weights = torch.load(model_pann_trained)
        model_original.load_state_dict(pretrained_weights)

        print_model_size(model_original)

        save_pruned_layers(opnorm=False)
        model_pruned = import_pruned_weights(model_original, model_pruned, P, opnorm=False)
        
        if TENSORBOARD:
            torch.save(model_pruned.state_dict(), f"resources/model_pann_opnorm_{P}.pt")
            pruned_fine_tuning(model_pruned, P, model_dir, dataloaders, n_epochs, data, threshold, batch_size, TENSORBOARD, writer, opnorm=False)
        else:
            pruned_fine_tuning(model_pruned, P, model_dir, dataloaders, n_epochs, data, threshold, batch_size, TENSORBOARD, opnorm=False)
        print_model_size(model_pruned)

if __name__ == "__main__":
    main()