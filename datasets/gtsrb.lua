--
--  GTSRB dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local GTSRBDataset = torch.class('GTSRBDataset', M)

function GTSRBDataset:__init(imageInfo, opt, split)
    -- imageInfo: result from gtsrb-gen.lua
    -- opt: command-line arguments
    -- split: "train" or "val"
    self.imageInfo = imageInfo[split]
    self.opt = opt
    self.split = split
    self.dir = paths.concat(opt.data, split)
    assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function GTSRBDataset:get(i)
    local path = ffi.string(self.imageInfo.imagePath[i]:data())

    local image = self:_loadImage(paths.concat(self.dir, path))
    local class = self.imageInfo.imageClass[i]

    return {
        input = image,
        target = class,
    }
end

function GTSRBDataset:_loadImage(path)
    local ok, input = pcall(function()
        return image.load(path, 3, 'float')
    end)

    -- Sometimes image.load fails because the file extension does not match the
    -- image format. In that case, use image.decompress on a ByteTensor.
    if not ok then
        local f = io.open(path, 'r')
        assert(f, 'Error reading: ' .. tostring(path))
        local data = f:read('*a')
        f:close()

        local b = torch.ByteTensor(string.len(data))
        ffi.copy(b:data(), data, b:size(1))

        input = image.decompress(b, 3, 'float')
    end

    return input
end

function GTSRBDataset:size()
    return self.imageInfo.imageClass:size(1)
end

-- Computed from GTSRB training images
local meanstd = {
    mean = { 0.341, 0.312, 0.321 },
    std = { 0.275, 0.264, 0.270 },
}

function GTSRBDataset:preprocess()
    local transforms = {t.Resize(48, 48)}
    if self.opt.globalNorm then
        table.insert(transforms, t.ColorNormalize(meanstd))
    end
    if self.opt.localNorm then
        table.insert(transforms, t.LocalContrastNorm(7))
    end

    return t.Compose(transforms)

end

return M.GTSRBDataset