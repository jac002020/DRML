require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'


preTrainModelFile = './results/pre-training/cnn_current.t7'
local _RESIDUAL_ = true

local function residual(branches, count)
    local triplet_residual = nn.Parallel(2, 2)
    for i = 1, branches:size() do
        local m = branches:get(i)
        m:remove()
        local layer = nn.Sequential()
        for j = 1, count do
            local block = nn.Sequential()
            block:add(nn.ConcatTable()
                :add(m:clone())
                :add(nn.Sequential()
                    :add(nn.SpatialConvolution(3, 3, 1, 1, 1, 1))
                    :add(nn.SpatialBatchNormalization(3))))
            block:add(nn.CAddTable())
            block:add(nn.ReLU())
            layer:add(block)
        end
        triplet_residual:add(layer)
    end

    return triplet_residual
end

preTrainModel = torch.load(preTrainModelFile)

preTrainModel:remove(7)
preTrainModel:remove(6)
   
if (_RESIDUAL_) then
    preTrainModel:insert(residual(preTrainModel:get(1), 3), 2)
    preTrainModel:remove(1)
end

local siamese_cnns = nn.ParallelTable()
siamese_cnns:add(preTrainModel)
siamese_cnns:add(preTrainModel:clone('weight', 'bias', 'gradWeight', 'gradBias'))

local model = nn.Sequential()
model:add(nn.SplitTable(2))
model:add(siamese_cnns)
model:add(nn.CSubTable())
model:add(nn.Linear(64, 64, false))
model:add(nn.Square())
model:add(nn.Sum(2))
model:add(nn.Sqrt())

W = torch.zeros(64, 64)
for i = 1, 64 do
    for j = 1, 64 do
        if i == j then
            W[i][j] = 1
        end
    end
end
model:get(4).weight = W:cuda()

model = model:cuda()
cudnn.convert(model, cudnn)

torch.save('./results/drml_0.t7', model:clearState())