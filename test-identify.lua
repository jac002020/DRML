require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'


modelFile = './results/fine-tuning/drml_current.t7'

torch.setdefaulttensortype('torch.FloatTensor')

testset = {
    batchSize = 1,
    length = 486 * 4,
    data = torch.load('./datasets/cuhk01/datatest.t7'),
    label = torch.load('./datasets/cuhk01/labeltest.t7'):cuda()
}

getData = function(dataset, i)
    local data = torch.Tensor(1, 3, 3, 64, 64):cuda()
    data[{ 1, 1, {}, {}, {} }] = dataset.data[{ i, {}, {1, 64}, {} }]
    data[{ 1, 2, {}, {}, {} }] = dataset.data[{ i, {}, {33, 96}, {} }]
    data[{ 1, 3, {}, {}, {} }] = dataset.data[{ i, {}, {65, 128}, {} }]
    return data
end

featureSet = {
    batchSize = 2048,
    length = testset.length * (testset.length - 1) / 2,
    feature = torch.Tensor(testset.length, 64):cuda()
}

featureData =  torch.Tensor(featureSet.batchSize, 2, 64):cuda()
featureLabel = torch.Tensor(featureSet.batchSize):cuda()

getBatch = function(dataset, i)
    local t = i or 0
    local size = math.min(t + featureSet.batchSize, featureSet.length) - t
    if(size ~= featureData:size(1)) then
        featureData = torch.Tensor(size, 2, 64):cuda();
        featureLabel = torch.Tensor(size):cuda();
    end
    for k = 1, size do
        local ti = math.ceil(math.sqrt(2 * (t + k) + 0.25) - 0.5)
        local tp = (ti - 1) * ti / 2
        local tj = t + k - tp
        featureData[{ k, 1, {} }] = dataset.feature[{ testset.length - ti, {} }]
        featureData[{ k, 2, {} }] = dataset.feature[{ testset.length - ti + tj, {} }]
        if(testset.label[testset.length - ti] == testset.label[testset.length - ti + tj]) then
            featureLabel[k] = 1
        else
            featureLabel[k] = -1
        end
    end

    return {featureData, featureLabel}, t + size
end

model = torch.load(modelFile)
model:evaluate()

featurePart = model:get(2):get(1)
featurePart = featurePart:cuda()
cudnn.convert(featurePart, cudnn)

print('Feature start ...')
for i = 1, testset.length do
    local data = featurePart:forward(getData(testset, i))
    featureSet.feature[{ i, {} }] = data[{{}}]
end

metricPart = nn.Sequential()
        :add(nn.SplitTable(2))
        :add(nn.CSubTable())
        :add(model:get(4))
        :add(nn.Square())
        :add(nn.Sum(2))
        :add(nn.Sqrt())
metricPart = metricPart:cuda()
cudnn.convert(metricPart, cudnn)

print('Metric start ...')
io.output("./results/prediction.txt")
maxIndex = math.ceil(featureSet.length / featureSet.batchSize)
for i = 1, maxIndex do
    batch, index = getBatch(featureSet, index)
    local distance = metricPart:forward(batch[1]):float()
    for j = 1, distance:size(1) do
       io.write(string.format("%f", distance[j]), ' ', batch[2][j], '\n')
    end
end