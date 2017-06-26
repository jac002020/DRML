require 'image'
local matio = require 'matio'


torch.setdefaulttensortype('torch.FloatTensor')

loadCUHK03 = function()
    print('Load CUHK03 ...')
    
    local CUHK03 = matio.load('./datasets/cuhk03/cuhk-03.mat', {'detected', 'labeled'})

    local detectedCellLength = {843,440,77,58,49}
    local detectedCellIndex = {0,843,1283,1360,1418}

    local detectedIndex = 0
    local detectedData = torch.Tensor(14097, 3, 128, 64)
    local detectedLabel = torch.Tensor(detectedData:size(1))

    for cellIndex = 1, 5 do
        for row = 1, detectedCellLength[cellIndex] do
            for col = 0, 9 do
                local im = CUHK03['detected'][cellIndex][row + col * detectedCellLength[cellIndex]]
                if im ~= nil then
                    im = im:resize(im:size(3), im:size(2), im:size(1))
                    local imDeal = torch.Tensor(im:size(1), im:size(3), im:size(2))
                    for channel = 1, im:size(1) do
                        imDeal[channel] = im[channel]:t()
                    end
                    imDeal = image.scale(imDeal / 255, 64, 128)
                    detectedIndex = detectedIndex + 1
                    detectedData[detectedIndex] = imDeal:clone()
                    detectedLabel[detectedIndex] = detectedCellIndex[cellIndex] + row
                    collectgarbage();
                end
            end
        end
    end

    torch.save('./datasets/cuhk03/cuhk03.detected.data.t7', detectedData)
    torch.save('./datasets/cuhk03/cuhk03.detected.label.t7', detectedLabel)
    collectgarbage();

    local labeledCellLength = {843,440,77,58,49}
    local labeledCellIndex = {0,843,1283,1360,1418}

    local labeledIndex = 0
    local labeledData = torch.Tensor(14096, 3, 128, 64)
    local labeledLabel = torch.Tensor(labeledData:size(1))

    for cellIndex = 1, 5 do
        for row = 1, labeledCellLength[cellIndex] do
            for col = 0, 9 do
                local im = CUHK03['labeled'][cellIndex][row + col * labeledCellLength[cellIndex]]
                if im ~= nil then
                    im = im:resize(im:size(3), im:size(2), im:size(1))
                    local imDeal = torch.Tensor(im:size(1), im:size(3), im:size(2))
                    for channel = 1, im:size(1) do
                        imDeal[channel] = im[channel]:t()
                    end
                    imDeal = image.scale(imDeal / 255, 64, 128)
                    labeledIndex = labeledIndex + 1
                    labeledData[labeledIndex] = imDeal:clone()
                    labeledLabel[labeledIndex] = labeledCellIndex[cellIndex] + row
                    collectgarbage();
                end
            end
        end
    end

    torch.save('./datasets/cuhk03/cuhk03.labeled.data.t7', labeledData)
    torch.save('./datasets/cuhk03/cuhk03.labeled.label.t7', labeledLabel)
end
loadCUHK03();

loadCUHK01 = function()
    print('Load CUHK01 ...')

    local CUHK01 = { torch.Tensor(971, 3, 128, 64), torch.Tensor(971, 3, 128, 64), torch.Tensor(971, 3, 128, 64), torch.Tensor(971, 3, 128, 64) }

    for individual = 1, 971 do
        for imIndex = 1, 4 do
            local file = "./datasets/cuhk01/campus/" .. string.format("%04d", individual) .. string.format("%03d", imIndex) .. ".png"
            local f = io.open(file)
            if(f ~= nil) then
                io.close(f)
                local im = image.load(file)
                im = image.scale(im, 64, 128)
                for channel = 1, 3 do
                    CUHK01[imIndex][{ {individual}, {channel}, {}, {} }] = im[{ {channel}, {}, {} }]
                end
            end
            collectgarbage();
        end
    end

    local dataIndex = torch.randperm(971)
    local datatrain = torch.Tensor(485 * 4, 3, 128, 64)
    local labeltrain = torch.Tensor(485 * 4)
    local datatest = torch.Tensor(486 * 4, 3, 128, 64)
    local labeltest = torch.Tensor(486 * 4)

    for i = 1, 485 do
        datatrain[{ 4 * i - 3, {}, {}, {} }] = CUHK01[1][{ dataIndex[i], {}, {}, {} }]
        datatrain[{ 4 * i - 2, {}, {}, {} }] = CUHK01[2][{ dataIndex[i], {}, {}, {} }]
        datatrain[{ 4 * i - 1, {}, {}, {} }] = CUHK01[3][{ dataIndex[i], {}, {}, {} }]
        datatrain[{ 4 * i, {}, {}, {} }] = CUHK01[4][{ dataIndex[i], {}, {}, {} }]
        labeltrain[4 * i - 3] = dataIndex[i]
        labeltrain[4 * i - 2] = dataIndex[i]
        labeltrain[4 * i - 1] = dataIndex[i]
        labeltrain[4 * i] = dataIndex[i]
        collectgarbage();
    end

    for i = 486, 971 do
        datatest[{ 4 * (i - 485) - 3, {}, {}, {} }] = CUHK01[1][{ dataIndex[i], {}, {}, {} }]
        datatest[{ 4 * (i - 485) - 2, {}, {}, {} }] = CUHK01[2][{ dataIndex[i], {}, {}, {} }]
        datatest[{ 4 * (i - 485) - 1, {}, {}, {} }] = CUHK01[3][{ dataIndex[i], {}, {}, {} }]
        datatest[{ 4 * (i - 485), {}, {}, {} }] = CUHK01[4][{ dataIndex[i], {}, {}, {} }]
        labeltest[4 * (i - 485) - 3] = dataIndex[i]
        labeltest[4 * (i - 485) - 2] = dataIndex[i]
        labeltest[4 * (i - 485) - 1] = dataIndex[i]
        labeltest[4 * (i - 485)] = dataIndex[i]
        collectgarbage();
    end

    torch.save('./datasets/cuhk01/dataIndex.t7', dataIndex)
    torch.save('./datasets/cuhk01/datatrain.t7', datatrain)
    torch.save('./datasets/cuhk01/labeltrain.t7', labeltrain)
    torch.save('./datasets/cuhk01/datatest.t7', datatest)
    torch.save('./datasets/cuhk01/labeltest.t7', labeltest)
end
loadCUHK01();