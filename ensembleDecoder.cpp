#include "ensembleDecoder.h"
#include "stabilizerCodes.h"
#include <algorithm>
#include <cstddef>
#include <iostream>

ensembleDecoder::ensembleDecoder(std::vector<std::string> decoder_names, AttributesDecoder list, fileReader& fileReader): 	
	list_of_specifiers(decoder_names), list(list), main(list.n, list.m, list.k, list.codeType, fileReader, list.trained){
	list_of_decoders.clear();
};

bool ensembleDecoder::updateGuess(const std::vector<unsigned>& candidate, int index) {
    auto candidate_weight = std::count_if(candidate.begin(), candidate.end(), [](unsigned x){ return x != 0; });
    auto current_weight = std::count_if(estimatedError.begin(), estimatedError.end(), [](unsigned x){ return x != 0; });

    if (candidate_weight < current_weight) {
        estimatedError = candidate;
    		bestDecoder = index;
    		return true;
  	}
  	return false;
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

void ensembleDecoder::add_decoder(stabilizerCodes decoder) {
	// Add a new decoder to the list
	list_of_decoders.push_back(decoder);
}

// This may be something that needs to happen in the main program
std::vector<bool> ensembleDecoder::decodeAllPaths(unsigned int L, double epsilon){
	std::vector<bool> success;
	std::vector<bool> bestSuccess;

	for (size_t i = 0; i < list_of_decoders.size(); ++i){
		// Initalize error size, has to happen at runtime
		success = list_of_decoders[i].decode(L, epsilon);
		if(i == 0){
			std::vector<unsigned> initialGuess(list_of_decoders[i].getErrorHat().size(), 1);
			estimatedError = initialGuess;
			bestSuccess = success;
		}
		// This may be wrong, since we are kind of decoding every instance anyway
			if(updateGuess(list_of_decoders[i].getErrorHat(), i)){
				bestSuccess = success;
		}
	}

	
		//Best-Case, any of the decoders is right
		// for(int i = 0 ; i < list_of_decoders.size(); i++){
		// 		success = list_of_decoders[i].decode(L, epsilon);
		// 		if (success[1]) {
		// 			break;
		// 		}
		// }
	
	return bestSuccess;
}
