require 'torch'
require 'image'
require 'cunn'
require 'cudnn'
require 'stn'
require 'nn'

print("Loading network...")
local model_path = "pretrained/gtsrb_cnn3st_model.t7"
local mean_std_path = "pretrained/gtsrb_cnn3st_mean_std.t7"
local network = torch.load(model_path)
local mean_std = torch.load(mean_std_path)
print("--- Network ---")
print(network)
print("--- Mean/Std ---")
local mean, std = mean_std[1], mean_std[2]
print("Mean:"..mean, "Std:"..std)


print("Loading sample images...")
local sample_img1 = image.load("sample_images/img1.jpg")
sample_img1 = image.scale(sample_img1, 48, 48)
local sample_img2 = image.load("sample_images/img2.jpg")
sample_img2 = image.scale(sample_img2, 48, 48)
local samples_tensor = torch.Tensor(2,sample_img1:size(1), sample_img1:size(2), sample_img1:size(3)):fill(0)
samples_tensor[1]:copy(sample_img1)
samples_tensor[2]:copy(sample_img2)

print("Applying global normalization to sample image")
samples_tensor:add(-mean)
samples_tensor:div(std)

print("Applying local contrast normalization to sample image")
local norm_kernel = image.gaussian1D(7)
local local_normalizer = nn.SpatialContrastiveNormalization(3, norm_kernel)
samples_tensor:copy(local_normalizer:forward(samples_tensor))

print("Running the network...")
samples_tensor = samples_tensor:cuda()
local scores = network:forward(samples_tensor)
print("Scores...")
print(scores)
local _, prediction1 = scores[1]:max(1)
local _, prediction2 = scores[2]:max(1)
print("Prediction class sample img 1: " .. prediction1[1] - 1)
print("Prediction class sample img 2: " .. prediction2[1] - 1)