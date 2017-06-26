require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'


modelFile = './results/pre-training/cnn_current.t7'

torch.setdefaulttensortype('torch.FloatTensor')

testset = {
    batchSize = 64,
    length = 14097,
    data = torch.load('./datasets/cuhk03/cuhk03.detected.data.t7'),
    label = torch.load('./datasets/cuhk03/cuhk03.detected.label.t7'):cuda(),
}

function testset:size()
    return math.ceil(self.length / self.batchSize)
end

batchData = torch.Tensor(testset.batchSize, 3, 3, 64, 64):cuda()
batchLabel = torch.Tensor(testset.batchSize):cuda()

getBatch = function(dataset, i)
    local t = batchIndex or 0
    local size = math.min(t + testset.batchSize, testset.length) - t
    if(size ~= batchData:size(1)) then
        batchData = torch.Tensor(size, 3, 3, 64, 64):cuda();
        batchLabel = torch.Tensor(size):cuda();
    end
    for k = 1, size do
        batchData[{ k, 1, {}, {}, {} }] = dataset.data[{ t + k, {}, {1, 64}, {} }]
        batchData[{ k, 2, {}, {}, {} }] = dataset.data[{ t + k, {}, {33, 96}, {} }]
        batchData[{ k, 3, {}, {}, {} }] = dataset.data[{ t + k, {}, {65, 128}, {} }]
        batchLabel[k] = dataset.label[t + k]
    end
    batchIndex = t + size
    return {batchData, batchLabel}
end

setmetatable(testset, 
    {__index = function(t, i) 
        return getBatch(t, i)
    end}
);

model = torch.load(modelFile)
model:evaluate()

correct = 0
for i = 1, testset:size() do
    local data = testset[i]
    local prediction = model:forward(data[1])
    local _, indices = torch.sort(prediction, true)
    for j = 1, data[2]:size(1) do
       if data[2][j] == indices[j][1] then
          correct = correct + 1
       end
    end
end

print(correct .. '/' .. testset.length .. ' ' .. (100 * correct / testset.length) .. '%')