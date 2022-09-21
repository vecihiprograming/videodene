-- GT deki tum class@larin icindeki tum videolar bir dizi haline getiriliyor. Sonra shuffle yapiliyor. sonra trainsize kadari train set yapiliyor.
require 'image'
local bdataset={}
function bdataset.new(folder)
sampleSize = {128, 96}
	trainsize=0.8
	testsize=0.2
	local data =  folder

	local dirs = dir.getdirectories(data);

	classess={} --video class
	allvideos={}
	for k,dirpath in ipairs(dirs) do
		-- print('k,dirpath',k,dirpath)
		-- her klasorde birden fazla video olabilir.
		-- bunlari da almak lazim
		-- bunlardan bazılarını train ve test yapmak lazım
		-- class - videos şeklinde dizidir.
		classess[k]=dirpath
		-- bir sıralı k indexine sınıfları aldım. klasör yolu şeklinde
		-- print(classess[k])
	end

	sayac=0
	for k=1,#classess do
		diric = dir.getdirectories(classess[k]);
		for i,dirpath in ipairs(diric) do
			sayac=sayac+1
			--print(diric)
			allvideos[sayac]=dirpath
			--print(allvideos[sayac])
		end
	end
	print(sayac)
	-- tüm klasörlerdeki videolar alındı

	-- random %80 train seti
	math.randomseed(30)
	local function shuffleTable( t )
	 
		if ( type(t) ~= "table" ) then
			print( "WARNING: shuffleTable() function expects a table" )
			return false
		end
	 
		local j
	 
		for i = #t, 2, -1 do
			j = math.random( i )
			t[i], t[j] = t[j], t[i]
		end
		return t
	end

	lastdizi=shuffleTable(allvideos)
	for k=1,#lastdizi do
		--print(lastdizi[k])
	end

	--print(math.ceil(#lastdizi*0.8))
	print(#lastdizi)
	testSet, trainSet = table.splice(lastdizi, 1, math.ceil(trainsize*#lastdizi))
	--print(#trainSet)
	--print(#testSet)

	for k=1,#trainSet do
		--print(trainSet[k])
	end
	--self.trainSet=trainSet
	--self.testSet=testSet
	--print(trainSet)
	return trainSet,testSet
end

local function loadImage(path)
	print('loadImage(path) yapıldı')
   local input = image.load(path, 3, 'float')
   -- print(input)
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   -- local iW = input:size(3)
   -- local iH = input:size(2)
   -- if iW < iH then
   --    input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   -- else
   --    input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   -- end
   --input = image.scale(input, loadSize[2], loadSize[1])
   
   return input
end

local function hooks(path)
		local dataTable = {}
		for i=1,7 do
			local img=loadImage(path..'/im'..tostring(i)..'.png')
					-- img ye hook uygula
					local out
					local iW = img:size(3) --96
					local iH = img:size(2) --128
					--print('iW iH',iW,iH)
					-- do random crop
					local oW = sampleSize[2] --96
					local oH = sampleSize[1] --128
					--print('oW, oH',oW,oH)
					if iH > oH and iw > oW then
					  local h1 = math.ceil(torch.uniform(1e-2, iH-oH)) -- uniform belirlenen aralıkta rastgele sayı üretir. ceil üste yuvarlar.
					  local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
					  out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
					else
					  out=img
					end
					assert(out:size(3) == oW)
					assert(out:size(2) == oH)
					--print('out[1]_____________',out[1])
					out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
					--print('out[1]muladd_____________',out[1])
			table.insert(dataTable,out)
		end

		return dataTable

end

function bdataset.loadBatch(arr)
   local dataTable = {}
   for i=1,#arr do
		path=arr[i]
		print(path)
		local out=hooks(path)
		--image.save('bb'..tostring(i)..'.png',out)
		table.insert(dataTable, out)
   end	
   
   return dataTable
end


return bdataset