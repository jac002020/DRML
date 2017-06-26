require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'loadcaffe'


local _RESIDUAL_ = true

local alexnet = loadcaffe.load('./models/deploy.prototxt', './models/bvlc_alexnet.caffemodel', 'cudnn')
alexnet:get(1).dW = 1
alexnet:get(1).dH = 1
for i = 24, 16, -1 do
    alexnet:remove()
end

local cnn = alexnet
if (_RESIDUAL_) then
    cnn:add(nn.SpatialFullConvolution(256, 3, 19, 19, 8, 8, 0, 0, 5, 5))
    cnn:add(nn.Reshape(12288))
else
    cnn:add(nn.Reshape(9216))
end

local triplet_branches = nn.Parallel(2, 2)
triplet_branches:add(cnn:clone())
triplet_branches:add(cnn:clone())
triplet_branches:add(cnn:clone())

local model = nn.Sequential()
model:add(triplet_branches)
if (_RESIDUAL_) then
    model:add(nn.Reshape(36864))
    model:add(nn.Linear(36864, 64))
else
    model:add(nn.Reshape(27648))
    model:add(nn.Linear(27648, 64))
end
model:add(nn.ReLU())
model:add(nn.Linear(64, 64))
model:add(nn.Linear(64, 1467))
model:add(nn.LogSoftMax())

model = model:cuda()
cudnn.convert(model, cudnn)

torch.save('./results/cnn_0.t7', model:clearState())