require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'


modelFile = './results/cnn_0.t7'

torch.setdefaulttensortype('torch.FloatTensor')

trainset = {
    batchSize = 1,
    length = 14096,
    data = torch.load('./datasets/cuhk03/cuhk03.labeled.data.t7'),
    label = torch.load('./datasets/cuhk03/cuhk03.labeled.label.t7'):cuda()
}

function trainset:size() 
    return math.ceil(self.length / self.batchSize)
end

shuffle = torch.randperm(trainset.length)
batchData = torch.Tensor(trainset.batchSize, 3, 3, 64, 64):cuda()
batchLabel = torch.Tensor(trainset.batchSize):cuda()

getBatch = function(dataset, i)
    local t = batchIndex or 0
    local size = math.min(t + trainset.batchSize, trainset.length) - t
    if(size ~= batchData:size(1)) then
        batchData = torch.Tensor(size, 3, 3, 64, 64):cuda();
        batchLabel = torch.Tensor(size):cuda();
    end
    for k = 1, size do
        batchData[{ k, 1, {}, {}, {} }] = dataset.data[{ shuffle[t + k], {}, {1, 64}, {} }]
        batchData[{ k, 2, {}, {}, {} }] = dataset.data[{ shuffle[t + k], {}, {33, 96}, {} }]
        batchData[{ k, 3, {}, {}, {} }] = dataset.data[{ shuffle[t + k], {}, {65, 128}, {} }]
        batchLabel[k] = dataset.label[shuffle[t + k]]
    end
    batchIndex = t + size
    return {batchData, batchLabel}
end

setmetatable(trainset, 
    {__index = function(t, i) 
        return getBatch(t, i)
    end}
);

model = torch.load(modelFile)
model = model:cuda()
cudnn.convert(model, cudnn)
model:training()

criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

trainer = nn.StochasticGradient(model, criterion)
trainer.maxIteration = 100
trainer.learningRate = 0.001
trainer.learningRateDecay = 0.00001
trainer.hookIteration = function(self, iteration, currentError)
    batchIndex = 0
    torch.save('./results/pre-training/cnn_current.t7', model:clearState())
end
print('Pre-training start ...')
io.output("./results/pre-training.log")
trainer:train(trainset)