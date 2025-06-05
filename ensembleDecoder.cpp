#include "ensembleDecoder.h"
#include <algorithm>

ensembleDecoder::ensembleDecoder(std::vector<unsigned> syn){
	mostLikely = syn;
};

void ensembleDecoder::updateGuess(std::vector<unsigned> syn){
	std::count(syn.begin(), syn.end(), 1) < std::count(mostLikely.begin(), mostLikely.end(), 1) ? (void)(mostLikely = syn): void();
}

