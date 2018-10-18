local torch = require 'torch'
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
require 'stn'
local image = require 'image'

local layers = {}

-- NVIDIA CuDNN
layers.convolution = cudnn.SpatialConvolution
layers.maxPooling = cudnn.SpatialMaxPooling
layers.nonLinearity = cudnn.ReLU
layers.batchNorm = cudnn.SpatialBatchNormalization

-- CPU
--layers.convolution = nn.SpatialConvolution
--layers.maxPooling = nn.SpatialMaxPooling
--layers.nonLinearity = nn.ReLU
--layers.batchNorm = nn.SpatialBatchNormalization

-- Returns the number of output elements for a table of convolution layers and the new height and width of the image
local function convs_noutput(convs, input_size)
    input_size = input_size or baseInputSize
    -- Get the number of channels for conv that are multiscale or not
    local nbr_input_channels = convs[1]:get(1).nInputPlane or
            convs[1]:get(1):get(1).nInputPlane
    local output = torch.CudaTensor(1, nbr_input_channels, input_size, input_size)
    for _, conv in ipairs(convs) do
        conv:cuda()
        output = conv:forward(output)
    end
    return output:nElement(), output:size(3), output:size(4)
end

-- Creates a conv module with the specified number of channels in input and output
-- If multiscale is true, the total number of output channels will be:
-- nbr_input_channels + nbr_output_channels
-- Using cnorm adds the spatial contrastive normalization module
-- The filter size for the convolution can be specified (default 5)
-- The stride of the convolutions is fixed at 1
local function new_conv(nbr_input_channels,nbr_output_channels, multiscale, cnorm, filter_size)
    multiscale = multiscale or false
    cnorm = cnorm or false
    filter_size = filter_size or 5
    local padding_size = 2
    local pooling_size = 2
    local norm_kernel = image.gaussian(7)

    local conv

    local first = nn.Sequential()
    first:add(layers.convolution(nbr_input_channels,
        nbr_output_channels,
        filter_size, filter_size,
        1,1,
        padding_size, padding_size))
    first:add(layers.nonLinearity(true))
    first:add(layers.maxPooling(pooling_size, pooling_size,
        pooling_size, pooling_size))

    if cnorm then
        first:add(nn.SpatialContrastiveNormalization(nbr_output_channels, norm_kernel))
    end

    if multiscale then
        conv = nn.Sequential()
        local second = layers.maxPooling(pooling_size, pooling_size,
            pooling_size, pooling_size)

        local parallel = nn.ConcatTable()
        parallel:add(first)
        parallel:add(second)
        conv:add(parallel)
        conv:add(nn.JoinTable(1,3))
    else
        conv = first
    end

    return conv
end

-- Creates a fully connection layer with the specified size.
local function new_fc(nbr_input, nbr_output)
    local fc = nn.Sequential()
    fc:add(nn.View(nbr_input))
    fc:add(nn.Linear(nbr_input, nbr_output))
    fc:add(layers.nonLinearity(true))
    return fc
end

-- Creates a classifier with the specified size.
local function new_classifier(nbr_input, nbr_output)
    local classifier = nn.Sequential()
    classifier:add(nn.View(nbr_input))
    classifier:add(nn.Linear(nbr_input, nbr_output))
    return classifier
end

-- Creates a spatial transformer module
-- locnet are the parameters to create the localization network
-- rot, sca, tra can be used to force specific transformations
-- input_size is the height (=width) of the input
-- input_channels is the number of channels in the input
-- no_cuda due to (1) below, we need to know if the network will run on cuda
local function new_spatial_tranformer(locnet, rot, sca, tra, input_size, input_channels)
    local nbr_elements = {}
    for c in string.gmatch(locnet, "%d+") do
        nbr_elements[#nbr_elements + 1] = tonumber(c)
    end

    -- Get number of params and initial state
    local init_bias = {}
    local nbr_params = 0
    if rot then
        nbr_params = nbr_params + 1
        init_bias[nbr_params] = 0
    end
    if sca then
        nbr_params = nbr_params + 1
        init_bias[nbr_params] = 1
    end
    if tra then
        nbr_params = nbr_params + 2
        init_bias[nbr_params-1] = 0
        init_bias[nbr_params] = 0
    end
    if nbr_params == 0 then
        -- fully parametrized case
        nbr_params = 6
        init_bias = {1,0,0,0,1,0}
    end

    local st = nn.Sequential()

    -- Create a localization network same as cnn but with downsampled inputs
    local localization_network = nn.Sequential()
    local conv1 = new_conv(input_channels, nbr_elements[1], false, false)
    local conv2 = new_conv(nbr_elements[1], nbr_elements[2], false, false)
    local conv_output_size = convs_noutput({conv1, conv2}, input_size/2)
    local fc = new_fc(conv_output_size, nbr_elements[3])
    local classifier = new_classifier(nbr_elements[3], nbr_params)
    -- Initialize the localization network (see paper, A.3 section)
    classifier:get(2).weight:zero()
    classifier:get(2).bias = torch.Tensor(init_bias)

    localization_network:add(layers.maxPooling(2,2,2,2))
    localization_network:add(conv1)
    localization_network:add(conv2)
    localization_network:add(fc)
    localization_network:add(classifier)

    -- Create the actual module structure
    local ct = nn.ConcatTable()

    local branch1 = nn.Sequential()
    branch1:add(nn.Transpose({3,4},{2,4}))
    branch1:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))

    local branch2 = nn.Sequential()
    branch2:add(localization_network)
    branch2:add(nn.AffineTransformMatrixGenerator(rot, sca, tra))
    branch2:add(nn.AffineGridGeneratorBHWD(input_size, input_size))
    branch2:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', true, true))

    ct:add(branch1)
    ct:add(branch2)

    st:add(ct)
    local sampler = nn.BilinearSamplerBHWD()
    -- (1)
    -- The sampler lead to non-reproducible results on GPU
    -- We want to always keep it on CPU
    -- This does no lead to slowdown of the training
    sampler:type('torch.FloatTensor')
    -- make sure it will not go back to the GPU when we call
    -- ":cuda()" on the network later
    sampler.type = function(type)
        return self
    end
    st:add(sampler)
    st:add(nn.Copy('torch.FloatTensor','torch.CudaTensor', true, true))
    st:add(nn.Transpose({2,4},{3,4}))

    return st
end

local function createModel(opt)
    nInputChannels = 3
    baseInputSize = opt.baseInputSize or 48
    local cnorm = opt.cNormConv or false
    local nbr_elements = {}
    for c in string.gmatch(opt.cnn, "%d+") do
        nbr_elements[#nbr_elements + 1] = tonumber(c)
    end
    assert(#nbr_elements == 4,
        'opt.cnn should contain 4 comma separated values, got '..#nbr_elements)

    local conv1 = new_conv(nInputChannels, nbr_elements[1], false, cnorm, 7)
    local conv2 = new_conv(nbr_elements[1], nbr_elements[2], false, cnorm, 4)
    local conv3 = new_conv(nbr_elements[2], nbr_elements[3], false, cnorm, 4)

    local convOutputSize, _, _ = convs_noutput({conv1, conv2, conv3})

    local fc = new_fc(convOutputSize, nbr_elements[4])
    local fc_class = new_classifier(nbr_elements[4], opt.nClasses)

    local features = nn.Sequential()

    if opt.locnet1 and opt.locnet1 ~= '' then
        local st1 = new_spatial_tranformer(
            opt.locnet1, -- locnet
            false, false, false, -- rot, sca, tra
            baseInputSize, -- input_size
            nInputChannels -- input_channels
        )
        features:add(st1)
    end

    features:add(conv1)

    if opt.locnet2 and opt.locnet2 ~= '' then
        local _, currentInputSize, _ = convs_noutput({conv1})
        local st2 = new_spatial_tranformer(
            opt.locnet2, -- locnet
            false, false, false, -- rot, sca, tra
            currentInputSize, -- input_size
            nbr_elements[1] -- input_channels
        )
        features:add(st2)
    end

    features:add(conv2)

    if opt.locnet3 and opt.locnet3 ~= '' then
        local _, currentInputSize, _ = convs_noutput({conv1,conv2})

        local st3 = new_spatial_tranformer(
            opt.locnet3, -- locnet
            false, false, false, -- rot, sca, tra
            currentInputSize, -- input_size
            nbr_elements[2] -- input_channels
        )
        features:add(st3)
    end

    features:add(conv3)

    local classifier = nn.Sequential()
    classifier:add(fc)
    classifier:add(fc_class)

    local model = nn.Sequential():add(features):add(classifier)

    return model
end

return createModel

-- Test paramareters
--opt = {}
--opt.nClasses = 43
--opt.cnn = '200,250,350,400'
--opt.locnet1=  '250,250,250'
--opt.locnet2=  '150,200,300'
--opt.locnet3=  '150,200,300'
--opt.globalNorm = 'false'
--opt.localNorm = 'false'
--model = createModel(opt)
--model:cuda()
--print(model)
--
--parameters, gradParameters = model:getParameters()
--print(parameters:size())
--print(gradParameters:size())

