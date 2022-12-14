------------------------------------------------------------------------
--[[ VRMSEReward ]]--
-- Variance reduced classification reinforcement criterion.
-- input : {class prediction, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRMSEReward, nn.SelectTable(-1))
------------------------------------------------------------------------
local VRMSEReward, parent = torch.class("nn.VRMSEReward", "nn.Criterion")

function VRMSEReward:__init(module, scale, areaScale, criterion)
   parent.__init(self)
   self.module = module -- so it can call module:reinforce(reward)
   self.scale = scale or 1 -- scale of reward
   self.areaScale = areaScale or 4 -- scale of reward
   self.errorC = nn.MSECriterion()
   -- self.fillOp = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize)
   -- self.fill = torch.ones(opt.batchSize,opt.nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2])
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   self.sizeAverage = true
   self.gradInput = {torch.Tensor()}
end

function VRMSEReward:updateOutput(inputTable, target)
   assert(torch.type(inputTable) == 'table')
   local map = inputTable[2]
   local input = inputTable[1]

   -- amac ne olabilir 
   -- input ile target birbirine benzer olması lazım
   -- buna göre ödül verecek
   
   -- reward = mse
   self.reward = input:clone()
   self.reward:add(-1, target):pow(2)--target -1 ile carptı. input ile topladı
   -- karesini aldı. mse yi yaptı ama daha bölmedi
                                  
   assert(self.reward:dim() == 5)
   for i = 5,2,-1 do
      self.reward = self.reward:sum(i)
   end
   -- batchSize kadar kanalları topladı. batchSize kadar fark elde etti. mse nin üstünü elde etti.2x1x1x1
   
   self.reward:resize(self.reward:size(1)) -- batchSize kadar çıktı 
   self.reward:div(-input:size(4)*input:size(5)) -- fark/M*N yaptı self.reward:div(-input:size(3)input:size(4)) -- fark/MN yaptı
   self.reward:add(4) -- pixel ~ [-1, 1], thus error^2 are <= 4 MSE diyebiliriz
   --print('self.reward:add(4)',self.reward)
   
   local area = map:sum(4):sum(3):sum(2):div(opt.highResSize[1]*opt.highResSize[2]) -- tum map toplanip/128*128 bolunuyor
   area = area:view(-1)
   --print('area',area)
   self.reward:add(self.areaScale,area) -- 4 = area reward scale
   -- areascale ile area çarptı + reward yaptı
	--print('BURADAAAAAAAAAAAAAAA')
   self.output = self.errorC:forward(input,target) -- mse bu
   --print(self.output)
   return self.output
end

function VRMSEReward:updateGradInput(inputTable, target)
   local input = inputTable[1]
   local baseline =inputTable[3]
   
   -- reduce variance of reward using baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
   self.vrReward:add(-1, baseline)
   if self.sizeAverage then
      self.vrReward:div(input:size(1))
   end
   -- broadcast reward to modules
   self.module:reinforce(self.vrReward)  
   
   -- zero gradInput (this criterion has no gradInput for class pred)
   self.gradInput[1]:resizeAs(input):zero()
   self.gradInput[1] = self.gradInput[1]
   -- self.gradInput[1] = self.errorC:backward(input,target)

   -- learn the baseline reward
   self.gradInput[3] = self.criterion:backward(baseline, self.reward)
   self.gradInput[3] = self.gradInput[3]
   return self.gradInput
end

function VRMSEReward:type(type)
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end