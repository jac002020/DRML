clc;
clear all;

test{1} = load('./results/prediction.txt');

for i = 1 : size(test, 2)
    test{i}(find(test{i}(:, 2) == 1), 2) = 0;
    test{i}(find(test{i}(:, 2) == -1), 2) = 1;
end

for i = 1 : size(test, 2)
    prec_rec(test{i}(:, 1), test{i}(:, 2), 'holdFigure', true, 'plotROC', false);
end