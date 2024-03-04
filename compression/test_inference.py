from compression.evaluation import print_model_size
from compression.models.PANN_pretrained import MobileNetV2
from compression.models.PANN_pruned import MobileNetV2_pruned
import torch
import time
device = "cpu"
# ------------- Testing Env
MODEL_PANN = False
PANN_QAT = False
PANN_QAT_V2 = False
PANN_SQ = False
OPNORM_PRUNING = False; P=0.5
L1_PRUNING = True
# ------------- Variables
model_pann = "resources/model_pann.pt"
model_pann_qat = "resources/model_pann_qat.pt"
model_pann_qat_v2 = "resources/model_pann_qat_v2.pt"
model_pann_sq = "resources/model_pann_sq.pt"
model_pann_opnorm_pruning = "resources/model_opnorm_pruning_0.5_FT.pt"
model_pann_l1_pruning = "resources/model_L1_norm_pruning_0.5_FT.pt"
# ------------- Hyperparameters
image_size = (256, 128)
batches = 1000

def main():
    if(MODEL_PANN):
        model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, post_training=True).to(device)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann, map_location=torch.device("cpu"))
        model.load_state_dict(pretrained_weights)
        # model.cuda()
        model.eval()
        test_predict(model)
    elif(PANN_QAT):
        model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, post_training=True, quantize=True)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann_qat)

        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.ao.quantization.prepare_qat(model, inplace=True)
        model = torch.quantization.convert(model.eval(), inplace=True)

        model.load_state_dict(pretrained_weights)
        model.eval()
        test_predict(model)
    elif(PANN_QAT_V2):
        model = MobileNetV2(44100, 1024, 320, 64, 50, 14000, 80, post_training=True, quantize=True)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann_qat_v2)

        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        torch.ao.quantization.prepare_qat(model, inplace=True)
        model = torch.quantization.convert(model.eval(), inplace=True)

        model.load_state_dict(pretrained_weights)
        model.eval()
        test_predict(model)

    elif(PANN_SQ):
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
    elif(OPNORM_PRUNING):
        model = MobileNetV2_pruned(P, 44100, 1024, 320, 64, 50, 14000, 80)
        model.to("cpu")
        pretrained_weights = torch.load(model_pann_opnorm_pruning, map_location=torch.device('cpu'))

        model.load_state_dict(pretrained_weights)
        model.eval()
        test_predict(model)
    elif(L1_PRUNING):
        model = MobileNetV2_pruned(P, 44100, 1024, 320, 64, 50, 14000, 80)
        pretrained_weights = torch.load(model_pann_l1_pruning, map_location=torch.device('cpu'))

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
            print(i)
            start_avg = time.time()

            inputs = data.to(device)

            if MODEL_PANN or PANN_QAT or PANN_QAT_V2 or OPNORM_PRUNING or PANN_SQ or L1_PRUNING:
                # Outputs: {"clipwise_output": [batch_size, num_classes], "Embedding": }
                outputs = model(inputs)["clipwise_output"]
            else:
                outputs = model(inputs)

            outputs = torch.sigmoid(outputs)

            end_avg = time.time()
            avg_time += end_avg-start_avg

        end = time.time()
        print(f"Total time: {end-start:.2f}s, average inference time for 1batch: {(avg_time/batches)*1000:.4f}ms")

if __name__ == "__main__":
    main()
