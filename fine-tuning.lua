require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'optim'


modelFile = './results/drml_0.t7'

torch.setdefaulttensortype('torch.FloatTensor')

trainset = {
    batchSize = 1,
    classes = 485,
    data = torch.load('./datasets/cuhk01/datatrain.t7'),
    label = torch.load('./datasets/cuhk01/labeltrain.t7')
}
trainset.batchSize = 3 * 2 * trainset.batchSize
trainset.length = trainset.classes * 4

function trainset:size()
    return self.length
end

math.randomseed(os.time())

getIndex = function(i)
    local jIndex = {1, 1, 1, 2, 2, 3}
    local kIndex = {2, 3, 4, 3, 4, 4}
    local b = (i - 1) % 6
    local a = (i - 1 - b) / 6
    local j = 4 * a + jIndex[b + 1]
    local k = 4 * a + kIndex[b + 1]
    if i > trainset.classes * 6 then
        local t = k - (j - (j % 4))
        while (t >= 1 and t <= 4) do
            j = math.random(trainset.length)
            k = math.random(trainset.length)
            t = k - (j - (j % 4))
        end
    end
    return j * (trainset.length + 1) + k
end

pairData = torch.Tensor(2, 3, 3, 64, 64):cuda()

getData = function(dataset, i, index)
    index = index or getIndex(i)
    local k = index % (trainset.length + 1)
    local j = (index - k) / (trainset.length + 1)
    pairData[{ 1, 1, {}, {}, {} }] = dataset.data[{ j, {}, {1, 64}, {} }]
    pairData[{ 1, 2, {}, {}, {} }] = dataset.data[{ j, {}, {33, 96}, {} }]
    pairData[{ 1, 3, {}, {}, {} }] = dataset.data[{ j, {}, {65, 128}, {} }]
    pairData[{ 2, 1, {}, {}, {} }] = dataset.data[{ k, {}, {1, 64}, {} }]
    pairData[{ 2, 2, {}, {}, {} }] = dataset.data[{ k, {}, {33, 96}, {} }]
    pairData[{ 2, 3, {}, {}, {} }] = dataset.data[{ k, {}, {65, 128}, {} }]
    if dataset.label[j] == dataset.label[k] then
        pairLabel = 1
    else
        pairLabel = -1
    end
    return { pairData:clone(), pairLabel, j * (trainset.length + 1) + k }
end

setmetatable(trainset,
    {__index = function(t, i) 
        return getData(t, i)
    end}
);

batchData = torch.Tensor(trainset.batchSize, 2, 3, 3, 64, 64):cuda()
batchLabel = torch.Tensor(trainset.batchSize):fill(-1):cuda()

batchCount = math.ceil(trainset.classes * 6 * 2 / trainset.batchSize)
batchIndex = torch.Tensor(trainset.batchSize * batchCount):fill(0)

setBatch = function(i, flag)
    flag = flag or false
    local index = (i - 1) * trainset.batchSize

    for n = 1, trainset.batchSize / 2 do
        if not flag then
            local data1 = getData(trainset, index / 2 + n)
            local data2 = getData(trainset, trainset.classes * 6 + index / 2 + n)
            batchData[n]:copy(data1[1])
            batchLabel[n] = data1[2]
            batchIndex[index + n] = data1[3]
            batchData[trainset.batchSize / 2 + n]:copy(data2[1])
            batchLabel[trainset.batchSize / 2 + n] = data2[2]
            batchIndex[index + trainset.batchSize / 2 + n] = data2[3]
        else
            local data1 = getData(trainset, nil, batchIndex[index + n])
            local data2 = getData(trainset, nil, batchIndex[index + trainset.batchSize / 2 + n])
            batchData[n]:copy(data1[1])
            batchLabel[n] = data1[2]
            batchData[trainset.batchSize / 2 + n]:copy(data2[1])
            batchLabel[trainset.batchSize / 2 + n] = data2[2]
        end
    end
end

model = torch.load(modelFile)
model = model:cuda()
cudnn.convert(model, cudnn)
model:training()

criterion = nn.HingeEmbeddingCriterion(50)
criterion = criterion:cuda()

params, gradParams = model:getParameters()
local optimState = {
    learningRate = 0.001
}

output = torch.Tensor(trainset.batchSize * batchCount):cuda()
trueLabel = torch.Tensor(trainset.batchSize * batchCount):cuda()

print('Fine-tuning start ...')
io.output("./results/fine-tuning.log")
for epoch = 1, 160 do
    feval = function(params)
        gradParams:zero()

        for i = 1, batchCount do
            setBatch(i)
            output[{ {(i - 1) * trainset.batchSize + 1, i * trainset.batchSize} }]:copy(model:forward(batchData))
            trueLabel[{ {(i - 1) * trainset.batchSize + 1, i * trainset.batchSize} }]:copy(batchLabel)
            collectgarbage();
        end

        local loss = criterion:forward(output, trueLabel)
        local dloss_doutputs = criterion:backward(output, trueLabel)

        for i = 1, batchCount do
            setBatch(i, true)
            model:forward(batchData)
            model:backward(batchData, dloss_doutputs[{ {(i - 1) * trainset.batchSize + 1, i * trainset.batchSize} }])
            collectgarbage();
        end

        print('epoch ' .. (epoch) .. ' error: ' .. (loss))
        torch.save('./results/fine-tuning/drml_current.t7', model:clearState())

        if epoch % 10 == 0 then
            torch.save('./results/fine-tuning/drml_' .. (epoch) .. '.t7', model:clearState())
        end

        return loss, gradParams
    end

    optim.sgd(feval, params, optimState)
end