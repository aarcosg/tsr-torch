# Traffic Sign Recognition
This is the code for the paper

**[Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods](https://doi.org/10.1016/j.neunet.2018.01.005)**
<br>
[Álvaro Arcos-García](https://scholar.google.com/citations?user=gjecl3cAAAAJ),
[Juan Antonio Álvarez-García](https://scholar.google.com/citations?user=Qk79xk8AAAAJ),
[Luis M. Soria-Morillo](https://scholar.google.com/citations?user=poBDpFkAAAAJ)
<br>

The paper addresses the problem of traffic sign classification using a Deep Neural Network which comprises Convolutional layers and Spatial Transfomer Networks. The model reports an accuracy of 99.71% in the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=results).

We provide:
- A [pretrained model](#pretrained-model).
- Test code to [run the model on new images](#running-on-new-images).
- Instructions for [training the model](#training).

If you find this code useful in your research, please cite:

```
"Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods."
Álvaro Arcos-García, Juan A. Álvarez-García, Luis M. Soria-Morillo. Neural Networks 99 (2018) 158-165.
```
\[[link](https://doi.org/10.1016/j.neunet.2018.01.005)\]\[[bibtex](
https://scholar.googleusercontent.com/citations?view_op=export_citations&user=gjecl3cAAAAJ&citsig=AMstHGQAAAAAW8ngijLlAnMmG2C2_Weu4sB-TWRmdT1P&hl=en)\]

## Requirements
This project is implemented in [Torch](http://torch.ch/), and depends on the following packages: [torch/torch7](https://github.com/torch/torch7), [torch/nn](https://github.com/torch/nn), [torch/image](https://github.com/torch/image),  [qassemoquab/stnbhwd](https://github.com/qassemoquab/stnbhwd), [torch/cutorch](https://github.com/torch/cutorch), [torch/cunn](https://github.com/torch/cunn), [cuDNN bindings for Torch](https://github.com/soumith/cudnn.torch) and [nninit](https://github.com/Kaixhin/nninit).

After installing torch, you can install / update these dependencies by running the following:
```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
luarocks install https://raw.githubusercontent.com/qassemoquab/stnbhwd/master/stnbhwd-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/Kaixhin/nninit/master/rocks/nninit-scm-1.rockspec
```

## Pretrained model
You can download a pretrained model from [Google Drive](https://drive.google.com/uc?export=download&id=1iqX1TZBxJSEmaEoReK2ZDko03JHqw1bp).
Unzip the file `gtsrb_cnn3st_pretrained.zip` and move its content to the folder `pretrained` of this project. It contains the pretrained model that obtains an accuracy of 99.71% in the German Traffic Sign Recognition Benchmark (GTSRB) and a second file with the mean and standard deviation values computed during the training process.

## Running on new images
To run the model on new images, use the script `run_model.lua`. It will run the pretrained model on the images provided in the `sample_images` folder:
```bash
th run_model.lua
```

## Training
To train and experiment with new traffic sign classification models, please follow the following steps:
1. Use the script `download_gtsrb_dataset.lua` which will create a new folder called `GTSRB` that will include two folders: `train` and `val`.
    ```bash
    th download_gtsrb_dataset.lua
    ```
2. Use the script `main.lua` setting the training parameters described in the file `opts.lua`. For example, the following options will generate the files needed to train a model with just one spatial transformer network localized at the beginning of the main network:
    ```bash
    th main.lua -data GTSRB -save GTSRB/checkpoints -dataset gtsrb -nClasses 43 -optimizer adam -LR 1e-4 -momentum 0 -weightDecay 0 -batchSize 50 -nEpochs 15 -weightInit default -netType cnn3st -globalNorm -localNorm -cNormConv -locnet2 '' -locnet3 '' -showFullOutput
    ```
    The next example, will create a model with three spatial transformer networks:
    ```bash
    th main.lua -data GTSRB -save GTSRB/checkpoints -dataset gtsrb -nClasses 43 -optimizer rmsprop -LR 1e-5 -momentum 0 -weightDecay 0 -batchSize 50 -nEpochs 15 -weightInit default -netType cnn3st -globalNorm -localNorm -cNormConv -showFullOutput
    ```
## Acknowledgements
The source code of this project is mainly based on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) and [gtsrb.torch](https://github.com/Moodstocks/gtsrb.torch).
