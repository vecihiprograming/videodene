require 'dpnn'
require 'rnn'
require 'nngraph'
require 'optim'
require 'image'
require 'nn'
require 'VRMSERewardEA'
require 'SpatialGlimpse_inverse'
require 'torch'

util = paths.dofile('util.lua')
-- nngraph.setDebug(true)
opt = lapp[[
   --randomize                (default 1)         batch size
   -b,--batchSize             (default 2)         batch size
   --frameSize                (default 7)         frame size
   --oW                		  (default 128)         batch size
   --oH                       (default 96)         batch size
   -r,--lr                    (default 0.0002)    learning rate

   --dataset                  (default 'folder')  imagenet / lsun / folder
   --nThreads                 (default 1)         # of data loading threads to use

   --beta1                    (default 0.5)       momentum term of adam
   --ntrain                   (default math.huge) #  of examples per epoch. math.huge for full dataset
   --display                  (default 0)         display samples while training. 0 = false
   --display_id               (default 10)        display window id.
   --gpu                      (default 0)         gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   --GAN_loss_after_epoch     (default 5)
   --name                     (default 'fullmodel')
   --data_root                (default './videos/IJBC_128_96_new/GT/')
   --data_list                (default './videos/IJBC_128_96_new/train.txt')
   --checkpoints_name         (default '')        name of checkpoints for load
   --checkpoints_epoch        (default 0)         epoch of checkpoints for load
   --epoch                    (default 1)         save checkpoints every N epoch
   --nc                       (default 3)         number of input image channels (RGB/Grey)

   --niter                    (default 1)  maximum number of iterations

   --rewardScale              (default 1)     scale of positive reward (negative is 0)
   --rewardAreaScale          (default 4)     scale of aree reward
   --locatorStd               (default 0.11)  stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')

   --glimpseHiddenSize        (default 128)   size of glimpse hidden layer')
   --glimpsePatchSize         (default '60,45')     size of glimpse patch at highest res (height = width)')
   --glimpseScale             (default 1)     scale of successive patches w.r.t. original input image')
   --glimpseDepth             (default 1)     number of concatenated downscaled patches')
   --locatorHiddenSize        (default 128)   size of locator hidden layer')
   --imageHiddenSize          (default 512)   size of hidden layer combining glimpse and locator hiddens')
   --wholeImageHiddenSize     (default 256)   size of full image hidden size

   --pertrain_SR_loss         (default 2)     SR loss before training action
   --residual                 (default 1)     whether learn residual in each step
   --rho                      (default 2)    back-propagate through time (BPTT) for rho time-steps
   --hiddenSize               (default 512)   number of hidden units used in Simple RNN.
   --FastLSTM                 (default 1)     use LSTM instead of linear layer
   --BN                                       whether use BatchNormalization
   --save_im                  (default 1)                                  whether save image on test
]]

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.ntrain == 'math.huge' then opt.ntrain = math.huge end

opt.loadSize = {128, 128} -- h w olarak dusundum ve sadece bu dosyayı ona gore guncelledım
opt.highResSize = {128, 128} -- h w olarak dusundum ve sadece bu dosyayı ona gore guncelledım
-- neden ımage.load h ve w okuyor. o sekılde dusundum
opt.lowResSize = {16, 16}

-- create data loader
local DataLoader = paths.dofile('data/data2.lua')
opt.data = './videos/IJBC_128_96_new/train/'
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
opt.data = './videos/IJBC_128_96_new/test/'
local dataTest = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
--------------------------------------------------------------
-- local patch size
local PatchSize = {}
PatchSize[1], PatchSize[2] = opt.glimpsePatchSize:match("([^,]+),([^,]+)")
opt.glimpsePatchSize = {}
opt.glimpsePatchSize[1] = tonumber(PatchSize[1])
opt.glimpsePatchSize[2] = tonumber(PatchSize[2])
opt.glimpseArea = opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2]
if opt.glimpseArea == opt.highResSize[1]*opt.highResSize[2] then
  opt.unitPixels = (opt.highResSize[2] - opt.glimpsePatchSize[2]) / 2
else
  opt.unitPixels = opt.highResSize[2] / 2
end
if opt.display == 0 then opt.display = false end
-----------------------------------------------------------------------
local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') or name:find('Linear') then
      if m.weight then m.weight:normal(0.0, 0.02) end
      if m.bias then m.bias:normal(0.0, 0.02) end
    elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:normal(0.0, 0.02) end
    end
end
-----------------------------------------------------------------------
local nc = opt.nc
local fs = opt.frameSize
local rho = opt.rho
local lowResSize = opt.lowResSize
local highResSize = opt.highResSize
-----------------------------------------------------------------------
local SpatialBatchNormalization
if opt.BN then SpatialBatchNormalization = nn.SpatialBatchNormalization
else SpatialBatchNormalization = nn.Identity end
local SpatialConvolution = nn.SpatialConvolution
-----------------------------------------------------------------------

videos=data:getBatch()

--print('video_size===>\n',videos:size())
-----------------------------------------------------------
----------------------- locator net -----------------------
-- Encode the (x,y) -- coordinate of last attended patch
local locationSensor = nn.Sequential()
locationSensor:add(nn.Linear(2, opt.locatorHiddenSize))
locationSensor:add(nn.BatchNormalization(opt.locatorHiddenSize)):add(nn.ReLU(true))

-- Encode the low-resolution video images
local videoSensor = nn.Sequential()
videoSensor:add(nn.View(-1):setNumInputDims(4))
videoSensor:add(nn.Linear(fs*nc*highResSize[1]*highResSize[2],opt.wholeImageHiddenSize))
videoSensor:add(nn.BatchNormalization(opt.wholeImageHiddenSize)):add(nn.ReLU(true))

-- Encode the enhanced videos in last step
local videoErrSensor = nn.Sequential()
videoErrSensor:add(nn.View(-1):setNumInputDims(4))
videoErrSensor:add(nn.Linear(fs*nc*highResSize[1]*highResSize[2],opt.wholeImageHiddenSize))
videoErrSensor:add(nn.BatchNormalization(opt.wholeImageHiddenSize)):add(nn.ReLU(true))

-- rnn input
videoglimpse = nn.Sequential()
videoglimpse:add(nn.ParallelTable():add(locationSensor):add(videoErrSensor):add(videoSensor))
videoglimpse:add(nn.JoinTable(1,1))
videoglimpse:add(nn.Linear(opt.wholeImageHiddenSize+opt.wholeImageHiddenSize+opt.locatorHiddenSize, opt.imageHiddenSize))
videoglimpse:add(nn.BatchNormalization(opt.imageHiddenSize)):add(nn.ReLU(true))
videoglimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))
videoglimpse:add(nn.BatchNormalization(opt.hiddenSize)):add(nn.ReLU(true))

-- rnn recurrent cell
recurrent = nn.GRU(opt.hiddenSize, opt.hiddenSize)

-- recurrent neural network
local rnn = nn.Recurrent(opt.hiddenSize, videoglimpse, recurrent, nn.ReLU(true), 99999)
print("rnn----->",rnn) 
-- output the coordinate of attended patch
local locatorvideos = nn.Sequential()
locatorvideos:add(nn.Linear(opt.hiddenSize, 2))
locatorvideos:add(nn.Tanh()) -- bounds mean between -1 and 1
locatorvideos:add(nn.ReinforceNormal(2*opt.locatorStd, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
locatorvideos:add(nn.HardTanh()) -- bounds sample between -1 and 1, while reinforce recieve no gradInput
locatorvideos:add(nn.MulConstant(opt.unitPixels*2/opt.highResSize[2]))
print("locatorvideos----->",locatorvideos)

-- ----------------------- SR net -----------------------
-- globally encode the attended patch
local SR_patch_fc = nn.Sequential()
SR_patch_fc:add(nn.JoinTable(1,3))
SR_patch_fc:add(nn.View(-1):setNumInputDims(3))
SR_patch_fc:add(nn.Linear(nc*2*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2],256)):add(nn.ReLU(true))
SR_patch_fc:add(nn.Linear(256,256)):add(nn.ReLU(true))
SR_patch_fc:add(nn.Linear(256,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
SR_patch_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))
-- globally encode the image
local SR_img_fc = nn.Sequential()
SR_img_fc:add(nn.JoinTable(1,3))
SR_img_fc:add(nn.View(-1):setNumInputDims(3))
SR_img_fc:add(nn.Linear(nc*2*highResSize[1]*highResSize[2],256)):add(nn.ReLU(true))
SR_img_fc:add(nn.Linear(256,256)):add(nn.ReLU(true))
SR_img_fc:add(nn.Linear(256,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
SR_img_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))
-- transform the hidden of RNN
local SR_fc = nn.Sequential()
SR_fc:add(nn.Linear(opt.hiddenSize,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
SR_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))

-----------------SRnet = ---------------------------------------------
local SRnet = nn.Sequential() -- 5 adet (2x3x60x45) giriyoruz
SRnet:add(nn.JoinTable(1,3)) -- 2x15x60x45 yaptı
SRnet:add(SpatialConvolution(nc*5, 16, 5, 5, 1, 1, 2, 2)) --1x16x60x45 çıkış
SRnet:add(SpatialBatchNormalization(16)):add(nn.ReLU(true))
SRnet:add(SpatialConvolution(16, 32, 7, 7, 1, 1, 3, 3)) -- 1x32x60x45
SRnet:add(SpatialBatchNormalization(32)):add(nn.ReLU(true))
SRnet:add(SpatialConvolution(32, 64, 7, 7, 1, 1, 3, 3)) -- 1x64x60x45
SRnet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
SRnet:add(SpatialConvolution(64, 64, 7, 7, 1, 1, 3, 3)) -- 1x64x60x45
SRnet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
SRnet:add(SpatialConvolution(64, 64, 7, 7, 1, 1, 3, 3)) -- 1x64x60x45
SRnet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
SRnet:add(SpatialConvolution(64, 32, 7, 7, 1, 1, 3, 3)) -- 1x32x60x45
SRnet:add(SpatialBatchNormalization(32)):add(nn.ReLU(true))
SRnet:add(SpatialConvolution(32, 16, 5, 5, 1, 1, 2, 2)) -- 1x16x60x45
SRnet:add(SpatialBatchNormalization(16)):add(nn.ReLU(true))
SRnet:add(SpatialConvolution(16, nc, 5, 5, 1, 1, 2, 2)) -- 1x3x60x45

-- placeholder lar
----------------------------------
local video_loc_prev = nn.Identity()()
local video = nn.Identity()()
local video_pre = nn.Identity()()
local video_visited_map_pre = nn.Identity()() -- used for record the attened area

local frame_pre1 = nn.Identity()()
local frame_pre2 = nn.Identity()()
local frame_pre3 = nn.Identity()()
local frame_pre4 = nn.Identity()()
local frame_pre5 = nn.Identity()()
local frame_pre6 = nn.Identity()()
local frame_pre7 = nn.Identity()()
local frame1 = nn.Identity()()
local frame2 = nn.Identity()()
local frame3 = nn.Identity()()
local frame4 = nn.Identity()()
local frame5 = nn.Identity()()
local frame6 = nn.Identity()()
local frame7 = nn.Identity()()
local pat = nn.Identity()()
local onesTensor = nn.Identity()()

local oneframe = nn.Identity()()
local oneframe_pre = nn.Identity()()
-------------------------------------  

local h = rnn({video_loc_prev,video_pre,video})
local loc = locatorvideos(h)

local visited_map = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize)({video_visited_map_pre, onesTensor, loc})

local patch1 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame1, loc})--1*3*60*45
local patch2 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame2, loc})
local patch3 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame3, loc})
local patch4 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame4, loc})
local patch5 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame5, loc})
local patch6 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame6, loc})
local patch7 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame7, loc})

local patch_pre1 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame_pre1, loc})
local patch_pre2 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame_pre2, loc})
local patch_pre3 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame_pre3, loc})
local patch_pre4 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame_pre4, loc})
local patch_pre5 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame_pre5, loc})
local patch_pre6 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame_pre6, loc})
local patch_pre7 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({frame_pre7, loc})

local SR_patch_fc_o1 = SR_patch_fc({patch1, patch_pre1})
local SR_patch_fc_o2 = SR_patch_fc({patch2, patch_pre2})
local SR_patch_fc_o3 = SR_patch_fc({patch3, patch_pre3})
local SR_patch_fc_o4 = SR_patch_fc({patch4, patch_pre4})
local SR_patch_fc_o5 = SR_patch_fc({patch5, patch_pre5})
local SR_patch_fc_o6 = SR_patch_fc({patch6, patch_pre6})
local SR_patch_fc_o7 = SR_patch_fc({patch7, patch_pre7})

local SR_img_fc_o1 = SR_img_fc({frame1, frame_pre1})
local SR_img_fc_o2 = SR_img_fc({frame2, frame_pre2})
local SR_img_fc_o3 = SR_img_fc({frame3, frame_pre3})
local SR_img_fc_o4 = SR_img_fc({frame4, frame_pre4})
local SR_img_fc_o5 = SR_img_fc({frame5, frame_pre5})
local SR_img_fc_o6 = SR_img_fc({frame6, frame_pre6})
local SR_img_fc_o7 = SR_img_fc({frame7, frame_pre7})

local SR_fc_o = SR_fc(h)

local hr_patch1 = SRnet({patch1, patch_pre1,SR_patch_fc_o1,SR_img_fc_o1,SR_fc_o})
local hr_patch2 = SRnet({patch2, patch_pre2,SR_patch_fc_o2,SR_img_fc_o2,SR_fc_o})
local hr_patch3 = SRnet({patch3, patch_pre3,SR_patch_fc_o3,SR_img_fc_o3,SR_fc_o})
local hr_patch4 = SRnet({patch4, patch_pre4,SR_patch_fc_o4,SR_img_fc_o4,SR_fc_o})
local hr_patch5 = SRnet({patch5, patch_pre5,SR_patch_fc_o5,SR_img_fc_o5,SR_fc_o})
local hr_patch6 = SRnet({patch6, patch_pre6,SR_patch_fc_o6,SR_img_fc_o6,SR_fc_o})
local hr_patch7 = SRnet({patch7, patch_pre7,SR_patch_fc_o7,SR_img_fc_o7,SR_fc_o})


if opt.residual then 
	hr_patch1 = nn.Tanh()(nn.CAddTable()({hr_patch1,patch_pre1})) 
	hr_patch2 = nn.Tanh()(nn.CAddTable()({hr_patch2,patch_pre2})) 
	hr_patch3 = nn.Tanh()(nn.CAddTable()({hr_patch3,patch_pre3})) 
	hr_patch4 = nn.Tanh()(nn.CAddTable()({hr_patch4,patch_pre4})) 
	hr_patch5 = nn.Tanh()(nn.CAddTable()({hr_patch5,patch_pre5})) 
	hr_patch6 = nn.Tanh()(nn.CAddTable()({hr_patch6,patch_pre6})) 
	hr_patch7 = nn.Tanh()(nn.CAddTable()({hr_patch7,patch_pre7})) 
end


local image_next1 = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize, nil)({frame_pre1,hr_patch1,loc})
local image_next2 = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize, nil)({frame_pre2,hr_patch2,loc})
local image_next3 = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize, nil)({frame_pre3,hr_patch3,loc})
local image_next4 = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize, nil)({frame_pre4,hr_patch4,loc})
local image_next5 = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize, nil)({frame_pre5,hr_patch5,loc})
local image_next6 = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize, nil)({frame_pre6,hr_patch6,loc})
local image_next7 = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize, nil)({frame_pre7,hr_patch7,loc})

print(hrs)

knm=torch.rand(1,2)
nngraph.annotateNodes()

model = nn.gModule({video_loc_prev,video_pre,video,
				frame1,
				frame2,
				frame3,
				frame4,
				frame5,
				frame6,
				frame7,
				frame_pre1,
				frame_pre2,
				frame_pre3,
				frame_pre4,
				frame_pre5,
				frame_pre6,
				frame_pre7,
				video_visited_map_pre,
				onesTensor},
				{loc,
				image_next1,
			    image_next2,
			    image_next3,
			    image_next4,
			    image_next5,
			    image_next6,
			    image_next7,
				visited_map})
				
			
model:apply(weights_init)
model.name = 'fullmodel'
model = nn.Recursor(model, opt.rho)			
			
gt_glimpse1 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)			
gt_glimpse2 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)			
gt_glimpse3 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)			
gt_glimpse4 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)			
gt_glimpse5 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)			
gt_glimpse6 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)			
gt_glimpse7 = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)			

baseline_R = nn.Sequential()
baseline_R:add(nn.Add(1))
local REINFORCE_Criterion = nn.VRMSEReward(model, opt.rewardScale, opt.rewardAreaScale)
local MSEcriterion = nn.MSECriterion()


function table.clone(org)
	return {table.unpack(org)}
end
---------------------------------------------------------------------------
optimState = {
learningRate = opt.lr,
beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local outputs
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   model:cuda()
   baseline_R:cuda()
   MSEcriterion:cuda();      REINFORCE_Criterion:cuda();
   gt_glimpse:cuda()
end
-- model:forget()			
local parameters, gradParameters = model:getParameters()
thin_model = model:sharedClone() -- used for save checkpoint
local a, b = thin_model:getParameters()
print(parameters:nElement())
print(gradParameters:nElement())

testLogger = optim.Logger(paths.concat(opt.name, 'test.log'))
testLogger:setNames{'MSE (training set)', 'PSNR (test set)'}
testLogger.showPlot = false

if opt.display then disp = require 'display' end			
			
local fx = function(x)
	gradParameters:zero()
	model:forget()
	
	highRes=data:getBatch() 
	lowRes = highRes:clone()
	for basize=1,opt.batchSize do
		for fr=1,opt.frameSize do
			print(basize,'===',fr)
			temp = image.scale(highRes[basize][fr], opt.lowResSize[2], opt.lowResSize[1])
			lowRes[basize][fr] = image.scale(temp, opt.highResSize[2], opt.highResSize[1], 'bicubic')
		end
	end	
	-- print(lowRes)
	local ones = torch.ones(opt.batchSize,1,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2])
	local visited_map0 = torch.zeros(opt.batchSize,1,highResSize[1],highResSize[2])
	local zero_loc = torch.zeros(opt.batchSize,2)
	local zero_dummy = torch.zeros(opt.batchSize,1)
	
	local dl = {}
	local inputs = {}
	outputs = {}
	gt = {}
	err_l = 0
	err_g = 0	
	t=1
	for t = 1,rho do	
		if t == 1 then
					inputs[t] = {zero_loc, lowRes,lowRes,
						lowRes[{{},{1},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]), -- her vdeonun ilk frameları
						lowRes[{{},{2},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{3},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{4},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{5},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{6},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{7},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{1},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{2},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{3},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{4},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{5},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{6},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						lowRes[{{},{7},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
						visited_map0, ones }
						print('-=-=-')
		else
					inputs[t]=table.clone(inputs[t-1])
					--inputs[t]=inputs[t-1]
					-- outnedir=zero_loc,nextx7tane,visitedmap
					local videonexts=torch.zeros(opt.batchSize,opt.frameSize,3,highResSize[1],highResSize[2])
					-- outputs[t-1][2] bs x 3 x 128 x 96
					videonexts[{{},{1},{}}]=outputs[t-1][2]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
					videonexts[{{},{2},{}}]=outputs[t-1][3]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
					videonexts[{{},{3},{}}]=outputs[t-1][4]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
					videonexts[{{},{4},{}}]=outputs[t-1][5]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
					videonexts[{{},{5},{}}]=outputs[t-1][6]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
					videonexts[{{},{6},{}}]=outputs[t-1][7]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
					videonexts[{{},{7},{}}]=outputs[t-1][8]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
					print('vnn')
					--print(videonexts)
					inputs[t][1]=outputs[t-1][1]:clone() -- locasyon
					inputs[t][2]=videonexts:clone()  -- video_pre parametresi
					inputs[t][3]=lowRes:clone() 
					inputs[t][4]= lowRes[{{},{1},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2])
					inputs[t][5]= lowRes[{{},{2},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2])
					inputs[t][6]= lowRes[{{},{3},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2])
					inputs[t][7]= lowRes[{{},{4},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2])
					inputs[t][8]= lowRes[{{},{5},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2])
					inputs[t][9]= lowRes[{{},{6},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2])
					inputs[t][10]=lowRes[{{},{7},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2])
					inputs[t][11]=outputs[t-1][2]:clone() 
					inputs[t][12]=outputs[t-1][3]:clone() 
					inputs[t][13]=outputs[t-1][4]:clone() 
					inputs[t][14]=outputs[t-1][5]:clone() 
					inputs[t][15]=outputs[t-1][6]:clone() 
					inputs[t][16]=outputs[t-1][7]:clone() 
					inputs[t][17]=outputs[t-1][8]:clone() 
					inputs[t][18]=outputs[t-1][9]:clone() 
					inputs[t][19]=ones:clone() 
		end
		outputs[t] = model:forward(inputs[t])
		print('---')
		-- highRes bs x 7 x 3 x 128 x 96
		
		-- gt[t] = gt_glimpse:forward{highRes[{{},{1},{}}]:reshape(opt.batchSize,3,128,96), 
										-- outputs[t][1] }:clone()
		gt[t]={}
		gt[t][1]=gt_glimpse1:forward{highRes[{{},{1},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),outputs[t][1] }:clone()
		gt[t][2]=gt_glimpse2:forward{highRes[{{},{2},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),outputs[t][1] }:clone()
		gt[t][3]=gt_glimpse3:forward{highRes[{{},{3},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),outputs[t][1] }:clone()
		gt[t][4]=gt_glimpse4:forward{highRes[{{},{4},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),outputs[t][1] }:clone()
		gt[t][5]=gt_glimpse5:forward{highRes[{{},{5},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),outputs[t][1] }:clone()
		gt[t][6]=gt_glimpse6:forward{highRes[{{},{6},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),outputs[t][1] }:clone()
		gt[t][7]=gt_glimpse7:forward{highRes[{{},{7},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),outputs[t][1] }:clone()
		--print(gt[t])
		--local MSE loss
		videonew=torch.zeros(opt.batchSize,opt.frameSize,3,highResSize[1],highResSize[2])
		videonew[{{},{1},{}}]=outputs[t][2]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
		videonew[{{},{2},{}}]=outputs[t][3]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
		videonew[{{},{3},{}}]=outputs[t][4]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
		videonew[{{},{4},{}}]=outputs[t][5]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
		videonew[{{},{5},{}}]=outputs[t][6]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
		videonew[{{},{6},{}}]=outputs[t][7]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
		videonew[{{},{7},{}}]=outputs[t][8]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])			
		
		err_l = err_l + MSEcriterion:forward(videonew, highRes) -- imagenext
		--print(err_l)
		dl[t] = MSEcriterion:backward(videonew, highRes):clone()
		--print('dl[t][{{},{1},{}}]',dl[t][{{},{1},{}}])
		--print('dl',dl)
	end
		-- print(type(outputs[rho][2]))
		-- print(type(outputs[rho][9]))
		-- print(type(highRes[{{},{1},{}}]:reshape(opt.batchSize,3,128,96)))
		local curbaseline_R = baseline_R:forward(zero_dummy)
		-- err_g = REINFORCE_Criterion:forward({outputs[rho][2], outputs[rho][9], curbaseline_R}, highRes[{{},{1},{}}]:reshape(opt.batchSize,3,128,96))	
		err_g = REINFORCE_Criterion:forward({videonew, outputs[rho][9], curbaseline_R}, highRes)	
		--print(err_g)	

		--backward sequence
		local dg = REINFORCE_Criterion:backward({videonew, outputs[rho][9], curbaseline_R}, highRes)	


		for t = rho,1,-1 do
			-- zero_loc & visited_map0 are zero tensor, which is ok used as gradOutput in this case
			model:backward(inputs[t], {zero_loc, 
												 dl[t][{{},{1},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
												 dl[t][{{},{2},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
												 dl[t][{{},{3},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
												 dl[t][{{},{4},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
												 dl[t][{{},{5},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
												 dl[t][{{},{6},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
												 dl[t][{{},{7},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
												 visited_map0})
		end	
		
		-- update baseline reward
		baseline_R:zeroGradParameters()
		baseline_R:backward(zero_dummy, dg[3])
		baseline_R:updateParameters(0.01)	
		print('sdsdfsdfsdfsdfsdf')
		return err_g, gradParameters		
end

function test()
	print('test içi')
	psnr = 0
	model:evaluate()

	paths.mkdir(opt.name)	
	for st = 1,dataTest:size(),opt.batchSize do
		model:forget()
		--xlua.progress(st,dataTest:size())
		--fetch data
		local i2, quantity
		if st + opt.batchSize > dataTest:size() then 
			i2 = dataTest:size() 
		else 
			i2 = st + opt.batchSize - 1 
		end
		quantity = i2 - st + 1
		highRes, impath,scaLabels = dataTest:getIndice({st,i2})	-- batchsıze kadar
		lowRes = highRes:clone()
		for basize=1,opt.batchSize do
			for fr=1,opt.frameSize do
				temp = image.scale(highRes[basize][fr], opt.lowResSize[2], opt.lowResSize[1])
				lowRes[basize][fr] = image.scale(temp, opt.highResSize[2], opt.highResSize[1], 'bicubic')
			end
		end

		highRes = highRes -- cuda ekle
		lowRes = lowRes --cuda ekle	
		local ones = torch.ones(opt.batchSize,1,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2])
		local visited_map0 = torch.zeros(opt.batchSize,1,highResSize[1],highResSize[2])
		local zero_loc = torch.zeros(opt.batchSize,2)
        ones=ones -- cuda ekle
        visited_map0=visited_map0-- cuda ekle
        zero_loc=zero_loc-- cuda ekle

		local output_t
		local input_t

		for t = 1,rho do	
			if t == 1 then
				input_t= {zero_loc, lowRes,lowRes,
					lowRes[{{},{1},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]), -- her vdeonun ilk frameları
					lowRes[{{},{2},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{3},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{4},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{5},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{6},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{7},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{1},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{2},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{3},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{4},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{5},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{6},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					lowRes[{{},{7},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]),
					visited_map0, ones }
                print(input_t)
			else
				--input_t=input_t -- sıfırlama içindi
                inputs={}
                --inputs=table.clone(inputs)
						-- -- outnedir=zero_loc,nextx7tane,visitedmap
				local videonexts=torch.zeros(opt.batchSize,opt.frameSize,3,highResSize[1],highResSize[2])--cuda ekle
				-- outputs[t-1][2] bs x 3 x 128 x 96
				videonexts[{{},{1},{}}]=outputs[2]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
				videonexts[{{},{2},{}}]=outputs[3]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
				videonexts[{{},{3},{}}]=outputs[4]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
				videonexts[{{},{4},{}}]=outputs[5]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
				videonexts[{{},{5},{}}]=outputs[6]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
				videonexts[{{},{6},{}}]=outputs[7]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
				videonexts[{{},{7},{}}]=outputs[8]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
				table.insert(inputs, outputs[1])--1
				table.insert(inputs, videonexts)--2      
                table.insert(inputs, lowRes)--3
                table.insert(inputs, lowRes[{{},{1},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]))--4    
                table.insert(inputs, lowRes[{{},{2},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]))--5
                table.insert(inputs, lowRes[{{},{3},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]))--6 
                table.insert(inputs, lowRes[{{},{4},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]))--7 
                table.insert(inputs, lowRes[{{},{5},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]))--8      
                table.insert(inputs, lowRes[{{},{6},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]))--9 
                table.insert(inputs, lowRes[{{},{7},{}}]:reshape(opt.batchSize,3,highResSize[1],highResSize[2]))--10 
                table.insert(inputs, outputs[2])--11
                table.insert(inputs, outputs[3])--12
                table.insert(inputs, outputs[4])--13
                table.insert(inputs, outputs[5])--14
                table.insert(inputs, outputs[6])--15
                table.insert(inputs, outputs[7])--16
                table.insert(inputs, outputs[8])--17
                table.insert(inputs, outputs[9])--18
                table.insert(inputs,ones)--19
			end
			outputs = model:forward(input_t) --bs x 9
		end
        
        print(outputs)

        local videonexts=torch.zeros(opt.batchSize,opt.frameSize,3,highResSize[1],highResSize[2])--cuda ekle
        -- outputs[t-1][2] bs x 3 x 128 x 96
        videonexts[{{},{1},{}}]=outputs[2]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
        videonexts[{{},{2},{}}]=outputs[3]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
        videonexts[{{},{3},{}}]=outputs[4]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
        videonexts[{{},{4},{}}]=outputs[5]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
        videonexts[{{},{5},{}}]=outputs[6]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
        videonexts[{{},{6},{}}]=outputs[7]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])
        videonexts[{{},{7},{}}]=outputs[8]:reshape(opt.batchSize,1,3,highResSize[1],highResSize[2])        

        for i = 1,quantity do
            -- 10* log10( 255^2 / (mse * (255/2)^2) )
            psnr = psnr + 10 * math.log10(4 / MSEcriterion:forward(videonexts[i], highRes[i]))
            if opt.save_im then
                for j=1,opt.frameSize do
                    local img = videonexts[{{i},{j},{}}]:reshape(3,highResSize[1],highResSize[2])
                    img:add(1):div(2)
                    image.save(opt.name..'/'..paths.basename(impath[i]..'/' .. 'im' .. j .. '.png'), img)                   
                end

            end
        end        

		
		error('testttt')
	end
end


-- train
epoch =  0
while epoch < opt.niter do
	epoch = epoch+1
	epoch_tm:reset()
	test()
	tm:reset()
	--print('parameters1',parameters)
	collectgarbage()
	--fx()
	optim.adam(fx, parameters, optimState)
	--a:copy(parameters)
	print('bitiiiii')
	error('iiiiiiiiiiiiiiiiii')
	
	--a:copy(parameters)
	--print('a',a)
	--print('parameters2',parameters)
   --optim.adam(fx, parameters, optimState)
end
			
-- local visited_map0 = torch.zeros(opt.batchSize,1,highResSize[1],highResSize[2])	
-- local ones = torch.ones(opt.batchSize,1,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2])		
-- a1=model:forward{knm,videos,videos,
