import argparse
from compression.evaluation import print_model_size
from compression.models.PANN_pretrained import MobileNetV2
from compression.models.PANN_pruned import MobileNetV2_pruned
import torch
import time
device = "cpu"
# ------------- Variables
model_pann = "resources/model_pann.pt"
model_pann_qat = "resources/model_pann_qat.pt"
model_pann_qat_v2 = "resources/model_pann_qat_v2.pt"
model_pann_sq = "resources/model_pann_sq.pt"
model_comb = "resources/comb_0.81.pt"
# ------------- Hyperparameters
image_size = (256, 128)
batches = 1000

def main(args):
    P = args.p
    if(args.base):
        model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, post_training=True).to(device)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann, map_location=torch.device("cpu"))
        model.load_state_dict(pretrained_weights)
        # model.cuda()
        model.eval()
        test_predict(model)
    elif(args.qat):
        model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, post_training=True, quantize=True)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann_qat)

        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.ao.quantization.prepare_qat(model, inplace=True)
        model = torch.quantization.convert(model.eval(), inplace=True)

        model.load_state_dict(pretrained_weights)
        model.eval()
        test_predict(model)
    elif(args.qat2):
        model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, post_training=True, quantize=True)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann_qat_v2)

        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.ao.quantization.prepare_qat(model, inplace=True)
        model = torch.quantization.convert(model.eval(), inplace=True)

        model.load_state_dict(pretrained_weights)
        model.eval()
        test_predict(model)

    elif(args.sq):
        model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, quantize=True, post_training=True)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann_sq)

        model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        # torch.backends.quantized.engine = 'x86'
        model = torch.quantization.prepare(model, inplace=True)
        model = torch.quantization.convert(model, inplace=True)

        model.load_state_dict(pretrained_weights)
        model.eval()
        test_predict(model)
    elif(args.op):
        model_pann_opnorm_pruning = f"resources/model_opnorm_pruning_{P}_FT.pt"
        model = MobileNetV2_pruned(P, 44100, 1024, 320, 64, 50, 14000, 80)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann_opnorm_pruning, map_location=torch.device('cpu'))

        model.load_state_dict(pretrained_weights)
        model.eval()
        test_predict(model)
    elif(args.l1):
        model_pann_l1_pruning = f"resources/model_L1_norm_pruning_{P}_FT.pt"
        model = MobileNetV2_pruned(P, 44100, 1024, 320, 64, 50, 14000, 80)
        pretrained_weights = torch.load(model_pann_l1_pruning, map_location=torch.device('cpu'))

        model.load_state_dict(pretrained_weights)
        model.eval()
        test_predict(model)
    elif(args.comb):
        P = 0.81
        model = MobileNetV2_pruned(P, 44100, 1024, 320, 64, 50, 14000, 80, quantize=True)
        model.to("cpu")
        pretrained_weights = torch.load(model_comb)

        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.ao.quantization.prepare_qat(model, inplace=True)
        model = torch.quantization.convert(model.eval(), inplace=True)

        model.load_state_dict(pretrained_weights)
        model.eval()
        test_predict(model)

    print_model_size(model, macs=True)

def test_predict(model):
    start = time.time()

    avg_time = 0
    with torch.no_grad():
        random_input = torch.rand(batches, 1, 1, image_size[0], image_size[1]) # shape: [amount_of_batches, batch_size, channels, height, width]
        for i, data in enumerate(random_input): 
            start_avg = time.time()

            inputs = data.to(device)
            outputs = model(inputs)["clipwise_output"]
            print(i)
            outputs = torch.sigmoid(outputs)
            end_avg = time.time()
            avg_time += end_avg-start_avg

        end = time.time()
        print(f"Total time: {end-start:.2f}s, average inference time for 1batch: {(avg_time/batches)*1000:.4f}ms")

def parser():
    parser = argparse.ArgumentParser(description="Argument parser for the provided variables")
    parser.add_argument("--base", default=False, action="store_true", help="Enable MODEL_PANN")
    parser.add_argument("--qat", default=False, action="store_true", help="Enable PANN_QAT")
    parser.add_argument("--qat2", default=False, action="store_true", help="Enable PANN_QAT_V2")
    parser.add_argument("--sq", default=False, action="store_true", help="Enable PANN_SQ")
    parser.add_argument("--op", default=False, action="store_true", help="Enable OPNORM_PRUNING")
    parser.add_argument("-p", type=float, default=0.5, help="Value of P if pruning is enabled (default: 0.5)")
    parser.add_argument("--l1", default=False, action="store_true", help="Enable L1_PRUNING")
    parser.add_argument("--comb", default=False, action="store_true", help="Enable COMBINATION")

    return parser.parse_args()
    

if __name__ == "__main__":
    args = parser()
    main(args)
