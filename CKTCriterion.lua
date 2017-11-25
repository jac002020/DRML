local CKTCriterion, parent = torch.class('nn.CKTCriterion', 'nn.Criterion')

function CKTCriterion:__init(alpha)
    parent.__init(self)
    self.Lk = 25
    self.Ld = 2.718181828
    self.alpha = alpha or 0.1
    self.regular = true
    self.lambda = 0.01
end

function sgn(input, alpha)
    return torch.cdiv(input, torch.pow(input, 2):add(alpha * alpha):sqrt())
end

function CKTCriterion:updateOutput(input, target)
    self.norm = torch.div(torch.log(torch.div(input, self.Lk)), torch.log(self.Ld))
    self.output = torch.cmul(sgn(self.norm, self.alpha), sgn(target, self.alpha)):sum() / input:nElement()
    if (self.regular == nil or self.regular == true) then
        self.output = self.output + torch.dist(torch.addmm(torch.zeros(64, 64):cuda(), self.w, self.w:t()), torch.eye(64):cuda()) * self.lambda / 2
    end
    return self.output
end

function CKTCriterion:updateGradInput(input, target)
    self.gradInput = torch.cdiv(torch.mul(torch.cmul(torch.pow(torch.pow(self.norm, 2):add(self.alpha * self.alpha), -1.5), sgn(target, self.alpha)), self.alpha * self.alpha / torch.log(self.Ld) / input:nElement()), input)
    return self.gradInput
end