--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('datasetyeni.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data

print("opt.data",opt.data)
opt.data = opt.data or os.getenv('DATA_ROOT') or '/data/local/imagenet-fetch/256'
if not paths.dirp(opt.data) then
    error('Did not find directory: ', opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local loadSize   = opt.loadSize
local sampleSize = loadSize

local function loadImage(path)
	--print('load image icindeyim')
   local input = image.load(path, opt.nc, 'float')
   -- print(input)
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   -- local iW = input:size(3)
   -- local iH = input:size(2)
   -- if iW < iH then
   --    input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   -- else
   --    input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   -- end
   input = image.scale(input, loadSize[2], loadSize[1])
   return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   --print('trainHook icindeyim')
   local input = loadImage(path)
   local out
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[2]--96
   local oH = sampleSize[1] --128
   if iH > oH and iw > oW then
     local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
     local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
		--print('h1,w1',h1,w1)
     out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   else
     out = input
   end
   
   --print('oW,oH',oW,oH)
   
   assert(out:size(3) == oW)
   assert(out:size(2) == oH)
   -- do hflip with probability 0.5
   -- if torch.uniform() > 0.5 then out = image.hflip(out); end
   out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
   return out
end

-- function to load the image, jitter it appropriately (random crops etc.)
local EAtrainHook = function(self, path)
   collectgarbage()
   
   local res = torch.zeros(opt.frameSize, 3, sampleSize[2], sampleSize[1])  -- samplesize1 genislik olacak
   --print(res)
   --error('sdfsdfsdfrsdf')
   for frs=1, opt.frameSize do
   --print('trainHook icindeyim')
	   local input = loadImage(path .. '/' .. 'im' .. frs .. '.png')
	   --print('input \n',#input)-- 3 x h x w ----  3 x 128 x 96
	   --error('sdfsdf')
	   -- print('2342343234353453453453')
	   -- error('sdfsdfsdfrsdf')
	   -- size(1) kanal
	   -- size(2) height
	   -- size(3) width
	   local out
	   local iW = input:size(2)--96
	   local iH = input:size(3)--128
	   --print('iW',iW)--128 height
	   --print('iH',iH)--96 width
	   
	   -- do random crop
	   -- local oW = sampleSize[2]
	   -- local oH = sampleSize[1]
	   -- if iH > oH and iw > oW then
		 -- local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
		 -- local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
			-- --print('h1,w1',h1,w1)
		 -- -- out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
		 -- res[frs]= image.crop(input, w1, h1, w1 + oW, h1 + oH)
	   -- else
		  --out[{ fr, {}, {}, {} }]:copy(input)
		  res[frs]=input:clone()
		  --print(#res[frs])
		  --print('4444444444444444444444444444444444')
		  
	   -- end
	   
	   --print('oW,oH',oW,oH)
	   
	   -- assert(res:size(3) == oW)
	   -- assert(res:size(4) == oH)   
   end
   

   -- do hflip with probability 0.5
   -- if torch.uniform() > 0.5 then out = image.hflip(out); end
   res:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
   --print(res) bs x 3 x 128 x 96
   --print(#res)
   --error('111')
   return res
end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache*****************')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = EAtrainHook
   trainLoader.sampleHookTest = EAtrainHook
   trainLoader.loadSize = {opt.nc, opt.loadSize[1], opt.loadSize[2]}
   trainLoader.sampleSize = {opt.nc, sampleSize[1], sampleSize[2]}
   print(trainLoader.loadSize[3])
else
   print('Creating train metadata---------------------')
   trainLoader = dataLoader{
      paths = {opt.data},
      forceClasses = opt.forceTable,
      loadSize = {opt.nc, loadSize[1], loadSize[2]},
      sampleSize = {opt.nc, sampleSize[1], sampleSize[2]},
      split = 100,
	  fs=opt.frameSize,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   print('saved metadata cache at', trainCache)
   trainLoader.sampleHookTrain = EAtrainHook
   trainLoader.sampleHookTest = EAtrainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end
