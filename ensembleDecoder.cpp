#include "ensembleDecoder.h"
#include "stabilizerCodes.h"
#include <algorithm>
#include <cstddef>

ensembleDecoder::ensembleDecoder(std::vector<std::string> decoder_names, DecoderAttributes list, double epsilon, fileReader& fileReader): 	
	list_of_specifiers(decoder_names), list(list), main(list.n, list.m, list.k, list.codeType, fileReader, list.trained), fr(list.n, list.k, list.m, list.codeType, list.trained){
	initalize_decoders(epsilon);
};

void ensembleDecoder::updateGuess(const std::vector<unsigned>& candidate, int index) {
    auto candidate_weight = std::count_if(candidate.begin(), candidate.end(), [](unsigned x){ return x != 0; });
    auto current_weight = std::count_if(estimatedError.begin(), estimatedError.end(), [](unsigned x){ return x != 0; });

    if (candidate_weight < current_weight) {
        estimatedError = candidate;
    		bestDecoder = index;
  }
}

// Initalizes decoder for ensemble
void ensembleDecoder::initalize_decoders(double epsilon){
	// Create the same error for all decoders
	main.add_error_given_epsilon(epsilon);
	list_of_decoders.clear();
	bestDecoder = 0;
	for (const auto& name : list_of_specifiers) {
			fileReader supplier(list.n, list.k, list.m, list.codeType, list.trained, name);
			stabilizerCodes code(list.n, list.k, list.m, list.codeType, supplier, list.trained, main.getErrorString(), main.getError());
			list_of_decoders.emplace_back(code);
	}
}

// This may be something that needs to happen in the main program
std::vector<bool> ensembleDecoder::decodeAllPaths(unsigned int L, double epsilon){
	std::vector<bool> success;

	for (size_t i = 0; i < list_of_decoders.size(); ++i){
		success = list_of_decoders[i].decode(L, epsilon);
		// Initalize error size, has to happen at runtime
		if(i == 0){
			std::vector<unsigned> initialGuess(list_of_decoders[i].getSyndrome().size(), 1);
			estimatedError = initialGuess;
		}
		if(success[0]){
			updateGuess(list_of_decoders[i].getErrorHat(), i);
		}
	}
	return list_of_decoders[bestDecoder].decode(L, epsilon);
}

