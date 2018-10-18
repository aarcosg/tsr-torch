--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

require 'torch'
require 'paths'
require 'optim'
require 'nn'

local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'trainer'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

-- we don't  change this to the 'correct' type (e.g. HalfTensor), because math
-- isn't supported on that type.  Type conversion later will handle having
-- the correct type.

local opt = opts.parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.nThreads)
print(opt.manualSeed)

torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
math.randomseed(opt.manualSeed)

-- Create unique checkpoint dir
dir_name = 'net-' .. opt.netType .. '__cnn-' .. opt.cnn ..
        '__locnet1-' .. opt.locnet1 .. '__locnet2-' .. opt.locnet2 .. '__locnet3-' .. opt.locnet3 ..
        '__optimizer-'.. opt.optimizer .. '__weightinit-' .. opt.weightInit ..
        os.date("__%Y_%m_%d_%X")
opt.save = paths.concat(opt.save, dir_name)
if paths.dir(opt.save) == nil then
    paths.mkdir(opt.save)
end

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

print("-- Model architecture --")
print(model)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

-- Logger
local logger = optim.Logger(paths.concat(opt.save, 'history.log'))
logger:setNames{'Train Loss', 'Train LossAbs','Train Acc',
                'Test Loss', 'Test LossAbs', 'Test Acc' }
logger:style{'+-','+-','+-','+-','+-','+-'}
logger:display(false)

if opt.testOnly then
    local top1Err, top5Err = trainer:test(0, valLoader)
    print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
    return
end

local startEpoch = opt.epochNumber --checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = 0
local bestTop5 = 0
local bestLoss = math.huge
local bestLossAbs = math.huge
local bestEpoch = math.huge
for epoch = startEpoch, opt.nEpochs do
    -- Train for a single epoch
    local trainTop1, trainTop5, trainLoss, trainLossAbs = trainer:train(epoch, trainLoader)

    -- Run model on validation set
    local testTop1, testTop5, testLoss, testLossAbs = trainer:test(epoch, valLoader)

    -- Update logger
    logger:add{trainLoss, trainLossAbs, trainTop1,
              testLoss, testLossAbs, testTop1 }
--    logger:plot()

    local bestModel = false
    if testTop1 > bestTop1 then
        bestModel = true
        bestTop1 = testTop1
        bestTop5 = testTop5
        bestLoss = testLoss
        bestLossAbs = testLossAbs
        bestEpoch = epoch
        if opt.showFullOutput then
            print(string.format(' * Best Model -- epoch:%i  top1: %6.3f  top5: %6.3f  loss: %6.3f, lossabs: %6.3f',
                bestEpoch, bestTop1, bestTop5, bestLoss, bestLossAbs))
        end
    end

    checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

--logger:plot()

print(string.format(' * Finished Best Model -- epoch:%i  top1: %6.3f  top5: %6.3f  loss: %6.3f, lossabs: %6.3f',
    bestEpoch, bestTop1, bestTop5, bestLoss, bestLossAbs))