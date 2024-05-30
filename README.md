# Audio-based Environmental Monitoring for Edge Devices

This repo contains code for my master thesis: **Audio-based Environmental Monitoring for Edge Devices**. The thesis focuses on neural network compression, specifically for the audio environmental monitoring domain. A variety of CNN models can be trained using this neural network compression framework. For the thesis, I compressed 2 models: MobileNetV2 and CNN14. Both models have already been pre-trained on AudioSet. The models have been fine-tuned on the DCASE 2019 task 2 curated dataset and are used for audio tagging.

## Environments (WIP)

The codebase is developed with Python 3.11. Install requirements as follows:
```sh
pip install -r requirements.txt
```

## Training on DCASE and Compressing CNN Models (WIP)

Models can be trained on the DCASE dataset using the `compression/main.py` file. Execute the following command to do this:
```sh
python -m compression.main
```

Note: Currently code behavior still has to be determined by changing all the variables in the code script (see "Testing env" and "Variables" sections at the top of the script). TODO: fix by creating CLI flags for this.

Note: Pruning and combination techniques require a baseline model that has already been trained on the DCASE dataset. Quantization does not require this, but performs much better when already trained on the target dataset (DCASE in our case).

## Compressed models

All compressed models can be found in the `resources` directory. CNN14 models are not included here, these files are too large. However, CNN14 training evaluation is available in Tensorboard (see [Evaluating Training Results](#Evaluating-Training-Results) section)

## Inference on Edge Devices

To test inference on edge devices the `compression/monitoring.py` script can be run. You can choose which (compressed) model to test with the included flags:
```sh
python -m compression.monitoring --base
```
Check all possibilities with the `-h` flag:
```sh
python -m compression.monitoring -h
```

## Evaluating Training Results 

Training results can be reviewed using TensorBoard with the following command:
```sh
tensorboard --logdir=compression/runs
```

## Plot Thesis Figures

To reproduce all figures from the thesis, just run the notebooks in the `scripts` folder.

## External links

Both baseline models, MobileNetV2 and CNN14, were taken from the PANN paper:

https://github.com/qiuqiangkong/audioset_tagging_cnn