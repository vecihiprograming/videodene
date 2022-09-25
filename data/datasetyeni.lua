--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
   pack=true,
   help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up)
]],
   {check=function(paths)
       local out = true;
       for k,v in ipairs(paths) do
          if type(v) ~= 'string' then
             print('paths can only be of string input');
             out = false
          end
       end
       return out
   end,
    name="paths",
    type="table",
    help="Multiple paths of directories with images"},

   {name="sampleSize",
    type="table",
    help="a consistent sample size to resize the images"},

   {name="split",
    type="number",
    help="Percentage of split to go to Training"
   },

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced ",
    default = "balanced"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="loadSize",
    type="table",
    help="a size to load the images to, initially",
    opt = true},

   {name="fs",
    type="number",
    help=""},	

   {name="forceClasses",
    type="table",
    help="If you want this loader to map certain classes to certain indices, "
       .. "pass a classes table that has {classname : classindex} pairs."
       .. " For example: {3 : 'dog', 5 : 'cat'}"
       .. "This function is very useful when you want two loaders to have the same "
    .. "class indices (trainLoader/testLoader for example)",
    opt = true},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function",
    help="applied to sample during testing",
    opt = true},
}

function dataset:__init(...)

   -- argcheck
   local args =  initcheck(...)
   for k,v in pairs(args) do 
	self[k] = v 
	print('DATASETYENI-->> args parametreleri','k :',k , ', v :',v  )
   end

   if not self.loadSize then 
	self.loadSize = self.sampleSize; 
	print('DATASETYENI-->> self.loadSize', self.loadSize ) --calismadi
   end

   if not self.sampleHookTrain then 
	self.sampleHookTrain = self.defaultSampleHook
	print('DATASETYENI-->> self.sampleHookTrain', self.sampleHookTrain ) --nil geliyor	
   end
   if not self.sampleHookTest then 
	self.sampleHookTest = self.defaultSampleHook -- bu da nil gelir
   end

   -- find class names buraya girmiyor
   self.classes = {}
   local classPaths = {}
   --print('self.forceClasses',self.forceClasses) -- nil geliyor
   if self.forceClasses then
      for k,v in pairs(self.forceClasses) do
		 --print('****','k,v',k,v)
		 self.classes[k] = v
         classPaths[k] = {}
      end
   end
   
   local function tableFind(t, o) 
	for k,v in pairs(t) do 
	   --print('####','k,v',k,v)
	   if v == o then 
	     return k 
	   end 
	end 
   end
   
   -- print('k-----',k)
   -- loop over each paths folder, get list of unique class names,
   -- also store the directory paths per class
   -- for each class,
   print('self.paths',#self.paths)
   for k,path in ipairs(self.paths) do
	  print('dataset path',path) -- ./lfw_funneled_dev_128/train/
      local dirs = dir.getdirectories(path);
		  
      for k,dirpath in ipairs(dirs) do
		 --print('k,dirpath',k,dirpath) -- k sayi dirpath class adlari yolu
         local class = paths.basename(dirpath)
		 --print ('paths.basename(dirpath)',paths.basename(dirpath)) -- class adlari
         local idx = tableFind(self.classes, class) -- table ilk bos, yavas yavas dolacak
         if not idx then
            table.insert(self.classes, class)-- class adini ekledi
			--print('self.classes',self.classes[k])
            idx = #self.classes -- eklenen son klasin numarasi bu oldu 
            classPaths[idx] = {}-- onu bos yapti
         end
         if not tableFind(classPaths[idx], dirpath) then
            table.insert(classPaths[idx], dirpath); -- claspathlerinin yolu
         end
      end	  
   end
   -- buranın sonunda sadece sınıf lıstesı ve sınıf yolları var
   
  -- print(#self.classes) -- GT 2792 class
  -- print(#classPaths) -- GT 2792
	-- error('324324')
   -- Nihayetinde self.classes => class adlari var
   -- classes[1] - adam
   -- classes[2] - ali
   -- classes[3] - veli
   -- classPaths[idx] de ilgili classid ye ait classyolu var.
   -- classPaths[1][1]=./lfw_funneled_dev_128/train/adam
   -- classPaths[2][1]=./lfw_funneled_dev_128/train/ali
   -- classPaths[3][1]=./lfw_funneled_dev_128/train/veli
   
   self.classIndices = {}
   for k,v in ipairs(self.classes) do
	  --print('v',v) -- class names
	  --print('k',k) -- sayi
      self.classIndices[v] = k
      --print(#self.classIndices)
   end
   -- self.classIndices['adam']=1
   -- self.classIndices['ali']=2
   -- self.classIndices['veli']=3
   
   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   ----------------------------------------------------------------------
   -- Options for the GNU find command
   -- local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
   --local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
   local findOptions2 = '  -maxdepth 1 -mindepth 1 '
   -- local findOptions2 = ' -type d '
   -- for i=2,#extensionList do
      -- findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   -- end

   -- find the image path names
   self.imagePath = torch.CharTensor()  -- path to each image in dataset
   self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   self.classList = {}                  -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data

   print('running "find" on each class directory, and concatenate all'
         .. ' those filenames into a single file containing all image paths for a given class')
   -- so, generat
   local classFindFiles = {}
   for i=1,#self.classes do
      classFindFiles[i] = os.tmpname()
   end
   --print(#classFindFiles)--2792
   local combinedFindList = os.tmpname();
   print('combinedFindList',combinedFindList)
   -- burada tum resim yollari var

   local tmpfile = os.tmpname()
   print(tmpfile)
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes

	sinifsayisi=0
   for i, class in ipairs(self.classes) do
	  sinifsayisi=sinifsayisi+1
	  -- print('class',class) adam
	  -- print('i',i) 1
	  -- ali 2
	  -- veli 3
      sinifici=0
	  for j,path in ipairs(classPaths[i]) do -- sınıf sayısı kadar fınd komutu yazılıyor
	  -- hazırlanan fınd komutu sınıf ıcındekı klasorlerı buluyor.
		 -- print(path) classpath
		 -- print(j) bu hep bir kullanilmiyor
         --local command = find .. ' "' .. path .. '" ' .. findOptions .. ' >>"' .. classFindFiles[i] .. '" \n'
         local command = find .. ' "' .. path .. '" ' .. findOptions2 .. ' >>"' .. classFindFiles[i] .. '" \n'
		 --print(command)
		 -- print(command) --dosyalarda arama icin gecici belge olusturuyor.
		-- find "./lfw_funneled_dev_128/train/Dennis_Erickson"  -iname "*.jpg" -o -iname "*.png" -o -iname "*.JPG" -o -iname "*.PNG" -o -iname "*.JPEG" -o -iname "*.ppm" -o -iname "*.PPM" -o -iname "*.bmp" -o -iname "*.BMP" >>"/tmp/lua_TN7iQV"

		-- find "./lfw_funneled_dev_128/train/Akbar_Al_Baker"  -iname "*.jpg" -o -iname "*.png" -o -iname "*.JPG" -o -iname "*.PNG" -o -iname "*.JPEG" -o -iname "*.ppm" -o -iname "*.PPM" -o -iname "*.bmp" -o -iname "*.BMP" >>"/tmp/lua_m9FgTU"		 
         tmphandle:write(command) -- tmpfile yaziliyor o da classFindFiles[i]
      end
	  --print('sinif no:',sinifsayisi)
   end
	
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile) -- command isleniyor ve class a ait resimler classFindFiles[i] dosylarina yaziliyor
   os.execute('rm -f ' .. tmpfile)
   print('now combine all the files to a single large file')
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- concat all finds to a single large file in the order of self.classes
   for i=1,#self.classes do
      local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   --print('tmpfile',tmpfile)
-- cat "/tmp/lua_UGetaJ" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_1OD8kK" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_gRcogH" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_1LltVK" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_21aBvI" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_M0iDHH" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_0ehQbH" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_y8Qn1I" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_nxTngH" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_8lwVkH" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_nLZn1I" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_DiplPJ" >>/tmp/lua_Fe3HpL
-- cat "/tmp/lua_pXdeuK" >>/tmp/lua_Fe3HpL   4038 satir
-- bunlarin hepsinde sinifa ait goruntu yollari var
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)
 --==========================================================================
   print('load the large concatenated list of sample paths to self.imagePath')
   local maxPathLength = tonumber(sys.fexecute(wc .. " -L '"
                                                  .. combinedFindList .. "' |"
                                                  .. cut .. " -f1 -d' '")) + 1
   	--print('maxPathLength',maxPathLength)		110 geldi								  
   local length = tonumber(sys.fexecute(wc .. " -l '"
                                           .. combinedFindList .. "' |"
                                           .. cut .. " -f1 -d' '"))
	--print('length',length) --9525 uzuhnluk

   assert(length > 0, "Could not find any image file in the given input paths")
   assert(maxPathLength > 0, "paths of files are length 0?")

   self.imagePath:resize(length, maxPathLength):fill(0)
   -- print(self.imagePath) bu bir chartensor 9525x110luk

   local s_data = self.imagePath:data()
    --print('s_data',s_data)
 
   local count = 0
   for line in io.lines(combinedFindList) do
	  -- print('line',line) tek tek butun image yollari geliyor
	  -- print('line',line) tek tek butun image yollari geliyor
      ffi.copy(s_data, line)
      s_data = s_data + maxPathLength
      if self.verbose and count % 10000 == 0 then
         xlua.progress(count, length)
      end;
      count = count + 1
   end
   --print('count',count) 9525

   self.numSamples = self.imagePath:size(1)
   print('self.numSamples ',self.numSamples ) -- 9525
   if self.verbose then print(self.numSamples ..  ' samples found.') end
   --==========================================================================
   print('Updating classList and imageClass appropriately')
   self.imageClass:resize(self.numSamples)
   local runningIndex = 0
   for i=1,#self.classes do --4038
      --if self.verbose then xlua.progress(i, #(self.classes)) end
      local length = tonumber(sys.fexecute(wc .. " -l '"
                                              .. classFindFiles[i] .. "' |"
                                              .. cut .. " -f1 -d' '"))
	  --print('length',length)	--her siniftaki image sayisi										
      if length == 0 then
         error('Class has zero samples')
      else
         self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
		 --print('self.classList[i]',self.classList[i])
		 -- self.classList[1]--> [1 2]
		 -- self.classList[2]--> [3]
		 -- self.classList[3]--> [4-5-6]
		 -- print(self.classList[i])
		 
         self.imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
		 --print(self.imageClass[{{runningIndex + 1, runningIndex + length}}])
		 
		 -- self.imageClass 9525 lik bir dizi
		 -- karsilarinda sinif no var
      end
      runningIndex = runningIndex + length
   end

	--print(self.classList)


   --==========================================================================
   -- clean up temporary files
   print('Cleaning up temporary files')
   local tmpfilelistall = ''
   for i=1,#(classFindFiles) do
      tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
      if i % 1000 == 0 then
         os.execute('rm -f ' .. tmpfilelistall)
         tmpfilelistall = ''
      end
   end
   os.execute('rm -f '  .. tmpfilelistall)
   os.execute('rm -f "' .. combinedFindList .. '"')
   --==========================================================================

   if self.split == 100 then
      self.testIndicesSize = 0
	  print('self.split ')
   else
      print('Splitting training and test sets to a ratio of '
               .. self.split .. '/' .. (100-self.split))
      self.classListTrain = {}
      self.classListTest  = {}
      self.classListSample = self.classListTrain
      local totalTestSamples = 0
      -- split the classList into classListTrain and classListTest
      for i=1,#self.classes do
		 print('for i=1,#self.classes do')
         local list = self.classList[i]
         local count = self.classList[i]:size(1)
         local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
         local perm = torch.randperm(count)
         self.classListTrain[i] = torch.LongTensor(splitidx)
         for j=1,splitidx do
            self.classListTrain[i][j] = list[perm[j]]
         end
         if splitidx == count then -- all samples were allocated to train set
            self.classListTest[i]  = torch.LongTensor()
         else
            self.classListTest[i]  = torch.LongTensor(count-splitidx)
            totalTestSamples = totalTestSamples + self.classListTest[i]:size(1)
            local idx = 1
            for j=splitidx+1,count do
               self.classListTest[i][idx] = list[perm[j]]
               idx = idx + 1
            end
         end
      end
      -- Now combine classListTest into a single tensor
	  print('Now combine classListTest into a single tensor')
      self.testIndices = torch.LongTensor(totalTestSamples)
      self.testIndicesSize = totalTestSamples
      local tdata = self.testIndices:data()
      local tidx = 0
      for i=1,#self.classes do
         local list = self.classListTest[i]
         if list:dim() ~= 0 then
            local ldata = list:data()
            for j=0,list:size(1)-1 do
               tdata[tidx] = ldata[j]
               tidx = tidx + 1
            end
         end
      end
   end
end

-- size(), size(class)
function dataset:size(class, list)
   list = list or self.classList
   if not class then
      return self.numSamples
   elseif type(class) == 'string' then
      return list[self.classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

-- getByClass
function dataset:getByClass(class)
	-- print('torch.uniform()',torch.uniform()) -- buna bakalim 
	-- print('self.classListSample[class]:nElement()',self.classListSample[class]:nElement()) -- buna bakalim 
	-- print(math.ceil(torch.uniform() * self.classListSample[class]:nElement()))
   local index = math.ceil(torch.uniform() * self.classListSample[class]:nElement())
   local imgpath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
   print('imgpath',imgpath)
   return self:sampleHookTrain(imgpath)
end

-- getByClass
function dataset:EAgetByClass(class) --bs x f x 3 x w x h icin
	local index = math.ceil(torch.uniform() * self.classListSample[class]:nElement())
	local imgpath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
	-- burada klasor yolu geldi daha img alinacak
   return self:sampleHookTrain(imgpath)

end
-- converts a table of samples (and corresponding labels) to a clean tensor

local function EAtableToOutput(self, dataTable, scalarTable)
   local data, scalarLabels, labels
   local quantity = #dataTable
   assert(dataTable[1]:dim() == 4)
   -- data = torch.Tensor(quantity,self.frameSize, 3, self.oW, self.oH)
   print(self.sampleSize[1], self.sampleSize[3], self.sampleSize[2])

   data = torch.Tensor(quantity,self.fs,self.sampleSize[1], self.sampleSize[3], self.sampleSize[2])
   print('=============='..self.sampleSize[1])
   scalarLabels = torch.LongTensor(quantity):fill(-1111)
   
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
      scalarLabels[i] = scalarTable[i]
   end
   return data, scalarLabels
end
-- converts a table of samples (and corresponding labels) to a clean tensor

local function tableToOutput(self, dataTable, scalarTable)
   local data, scalarLabels, labels
   local quantity = #scalarTable
   assert(dataTable[1]:dim() == 3)
   data = torch.Tensor(quantity,self.frameSize, 3, self.oW, self.oH)
   -- data = torch.Tensor(quantity,self.fs,self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
   print('=============='..self.sampleSize[1])
   scalarLabels = torch.LongTensor(quantity):fill(-1111)
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
      scalarLabels[i] = scalarTable[i]
   end
   return data, scalarLabels
end
function dataset:getByClasses(classes)
   assert(classes)
   local dataTable = {}
   local scalarTable = {}
   for i=1,classes:size(1) do
      local class = classes[i]
      local out = self:getByClass(class)
      table.insert(dataTable, out)
      table.insert(scalarTable, class)
   end
   local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels
end
-- sampler, samples from the training set.
function dataset:sample(quantity)
   print('datasetyeni_sample_fonksiyonu_basliyor')
   assert(quantity)
   local dataTable = {}
   local scalarTable = {}
   for i=1,quantity do
      local class = torch.random(1, #self.classes)
	  --print('class',class)
      local out = self:getByClass(class)
	  --print('out',out)
      table.insert(dataTable, out)
      table.insert(scalarTable, class)
   end
   local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels
end

-- sampler, samples from the training set.
function dataset:EAsample(quantity)
   print('datasetyeni_sample_fonksiyonu_basliyor')
   assert(quantity)
   local dataTable = {}
   local scalarTable = {}
   for i=1,quantity do
      local class = torch.random(1, #self.classes)
	  --print('class',class)
      local out = self:EAgetByClass(class)
	  --print('out',out)
      table.insert(dataTable, out)
      table.insert(scalarTable, class)
   end
   local data, scalarLabels = EAtableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels
end

function dataset:get(i1, i2)
   local indices = torch.range(i1, i2);
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   local scalarTable = {}
   local impath = {}
   for i=1,quantity do
      -- load the sample
      local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
      local out = self:sampleHookTest(imgpath)
      table.insert(dataTable, out)
      table.insert(scalarTable, self.imageClass[indices[i]])
      table.insert(impath, imgpath)
   end
   local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
   return data, impath, scalarLabels
end

return dataset
