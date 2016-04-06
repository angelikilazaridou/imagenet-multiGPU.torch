--[[
from here: https://github.com/rotmanmi/word2vec.torch
--]]
require 'paths'
require 'hdf5'
torch.setdefaulttensortype('torch.FloatTensor')

local Word2vec = torch.class('Word2vec')

function Word2vec:__init(dir)

	print('Bla')
	self.w2v = {}	
	local f = dir .. '/vectors.h5'
	if not paths.filep(f .. '.t7') then
		print(string.format('Ready to load word vectors from %s', f))
	        self.w2v = self:_bintot7(f)
	else
		print(string.format('Ready to load word vectors from %s', f .. '.t7'))
        	self.w2v = torch.load(f .. '.t7')
        end
        print('Done reading word2vec data.')
end


function Word2vec:_bintot7(f)

       
	local h5_file = hdf5.open(f, 'r')
	local vecs_size = h5_file:read('/vectors'):dataspaceSize()
	local M = h5_file:read('/vectors'):partial({1,vecs_size[1]},{1,vecs_size[2]})
        --Writing Files
        word2vec = {}
        word2vec.M = M
        word2vec.w2vvocab = w2vvocab
        word2vec.v2wvocab = v2wvocab
        torch.save(f .. '.t7',word2vec)
        print('Writing t7 File for future usage.')

        return word2vec
end



function Word2vec:getVector(label)
	local ind  = self.w2v.w2vvocab[label]
	return self.w2v.M[ind]
end

