require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'loadcaffe'


local w_G = 2

local alexnet = loadcaffe.load('../models/deploy.prototxt', '../models/bvlc_alexnet.caffemodel', 'cudnn')
alexnet:get(1).dW = 1
alexnet:get(1).dH = 1
for i = 24, 16, -1 do
    alexnet:remove()
end

local cnn = alexnet
cnn:add(nn.SpatialFullConvolution(512, 3, 19, 19, 8, 8, 0, 0, 5, 5))
cnn:add(nn.Reshape(12288))

local group = function(layer, G, length, index)
    local groups = nn.ConcatTable()
    for i = 1, G do
        local s = nn.Sequential()
        if index ~= 1 then
            s:add(nn.Narrow(2, 1 + (i - 1) * length, length))
        end
        s:add(layer:clone())
        groups:add(s)
    end
    return groups
end

local wide = function(layers, indexes, G)
    for i = 1, table.getn(indexes) do
        local index = indexes[i] + 2 * (i - 1)
        local channels = layers.modules[index]['nOutputPlane']
        local inputChannels = layers.modules[index]['nInputPlane']
        layers:insert(group(layers:get(index), G, inputChannels, i), 1 + index)
        layers:insert(nn.JoinTable(2), 2 + index)
        layers:insert(cudnn.SpatialConvolution(G * channels, G * channels, 1, 1, 1, 1, 0, 0, 1), 3 + index)
        layers:remove(index)
    end
end
wide(cnn, { 1, 5, 9, 11, 13 }, w_G)

cudnn.convert(cnn, cudnn)
cnn = cnn:cuda()

local triplet_branches = nn.Parallel(2, 2)
triplet_branches:add(cnn:clone())
triplet_branches:add(cnn:clone())
triplet_branches:add(cnn:clone())

local model = nn.Sequential()
model:add(triplet_branches)
model:add(nn.Reshape(36864))
model:add(nn.Linear(36864, 64))
model:add(nn.ReLU())
model:add(nn.Linear(64, 64))
model:add(nn.Linear(64, 1467))
model:add(nn.LogSoftMax())

model = model:cuda()
cudnn.convert(model, cudnn)


print(model)
input = torch.randn(2, 3, 3, 64, 64):cuda()
output = model:forward(input)
print(#output)


torch.save('../results/cnn_0.t7', model:clearState())
