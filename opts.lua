--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-data',           '',         'Path to dataset')
    cmd:option('-computeMeanStd', false,    'Compute mean and std')
    cmd:option('-dataset',        'gtsrb', 'Options: gtsrb')
    cmd:option('-manualSeed',     1,          'Manually set RNG seed')
    cmd:option('-nGPU',           1,          'Number of GPUs to use by default')
    cmd:option('-backend',        'cudnn',    'Options: cudnn | cunn')
    cmd:option('-cudnn',          'default',  'Options: fastest | default | deterministic')
    cmd:option('-gen',            'gen',      'Path to save generated files')
    cmd:option('-precision',      'single',   'Options: single | double | half')
    cmd:option('-showFullOutput', false,     'Whether show full training process (true) or just final output (false)' )
    ------------- Data options ------------------------
    cmd:option('-nThreads',        1, 'number of data loading threads')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         0,       'Number of total epochs to run')
    cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       32,      'mini-batch size (1 = pure stochastic)')
    cmd:option('-testOnly',        false, 'Run on validation set only')
    cmd:option('-tenCrop',         false, 'Ten-crop testing')
    ------------- Checkpointing options ---------------
    cmd:option('-checkpoint',      false,       'Save model after each epoch')
    cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
    cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
    ---------- Optimization options ----------------------
    cmd:option('-optimizer',       'sgd', 'Options: sgd | adam | rmsprop | adagrad | lbfgs | nag')
    cmd:option('-LR',              0.01,   'initial learning rate')
    cmd:option('-momentum',        0.9,   'momentum')
    cmd:option('-weightDecay',     1e-4,  'weight decay')
    cmd:option('-nesterov',        false , 'Nesterov')
    cmd:option('-LRDecayStep',     10,    'number of steps to decay LR by 0.1')
    ---------- Model options ----------------------------------
    cmd:option('-netType',      'cnn3st', 'Options: cnn3st')
    cmd:option('-retrain',      'none',   'Path to model to retrain with')
    cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
    cmd:option('-weightInit',   'default',  'Options: default | kaiming | glorot | uniform | conv_aware')
    ---------- Model paper_conv3_st3 options ----------------------------------
    cmd:option('-cnn',          '200,250,350,400', 'Network parameters (conv1_out, conv2_out, conv3_out, fc1_out)')
    cmd:option('-locnet1',      '250,250,250',     'Localization network 1 parameters')
    cmd:option('-locnet2',      '150,200,300',     'Localization network 2 parameters')
    cmd:option('-locnet3',      '150,200,300',     'Localization network 3 parameters')
    cmd:option('-globalNorm',   false,            'Whether perform global normalization')
    cmd:option('-localNorm',    false,            'Whether perform local normalization')
    cmd:option('-cNormConv',    false,            'Whether perform contrastive normalization in conv modules')
    cmd:option('-dataAug',      false,           'Whether perform data augmentation on training dataset')
    ---------- Model options ----------------------------------
    cmd:option('-shareGradInput',  false, 'Share gradInput tensors to reduce memory usage')
    cmd:option('-optnet',          false, 'Use optnet to reduce memory usage')
    cmd:option('-resetClassifier', false, 'Reset the fully connected layer for fine-tuning')
    cmd:option('-nClasses',         0,      'Number of classes in the dataset')
    cmd:option('-baseInputSize',    48,     'Size of input images')
    cmd:text()

    local opt = cmd:parse(arg or {})

    if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
        cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
    end

    if opt.precision == nil or opt.precision == 'single' then
        opt.tensorType = 'torch.CudaTensor'
    elseif opt.precision == 'double' then
        opt.tensorType = 'torch.CudaDoubleTensor'
    elseif opt.precision == 'half' then
        opt.tensorType = 'torch.CudaHalfTensor'
    else
        cmd:error('unknown precision: ' .. opt.precision)
    end

    if opt.resetClassifier then
        if opt.nClasses == 0 then
            cmd:error('-nClasses required when resetClassifier is set')
        end
    end
    if opt.shareGradInput and opt.optnet then
        cmd:error('error: cannot use both -shareGradInput and -optnet')
    end

    print('--- Options ---')
    print(opt)

    return opt
end

return M