#include "ensembleDecoder.h"
#include "stabilizerCodes.h"
#include <algorithm>
#include <cstddef>

ensembleDecoder::ensembleDecoder() 	
{
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

void ensembleDecoder::setErrors(std::vector<std::string> errorString, std::vector<unsigned> error){
  for(size_t i = 0; i < list_of_decoders.size(); ++i){
    list_of_decoders[i]->set_error_given_epsilon(errorString, error);
  }

}

void ensembleDecoder::add_decoder(stabilizerCodes& decoder) {
	// Add a new decoder to the list
	list_of_decoders.push_back(&decoder);
}

// This may be something that needs to happen in the main program
std::vector<bool> ensembleDecoder::decodeAllPaths(unsigned int L, double epsilon){
	std::vector<bool> success;
	std::vector<bool> bestSuccess;

	for (size_t i = 0; i < list_of_decoders.size(); ++i){
		// Initalize error size, has to happen at runtime
		success = list_of_decoders[i]->decode(L, epsilon);
		if(i == 0){
			std::vector<unsigned> initialGuess(list_of_decoders[i]->getErrorHat().size(), 1);
			estimatedError = initialGuess;
			bestSuccess = success;
		}
		// This may be wrong, since we are kind of decoding every instance anyway
    if (success[0]) {
        if (updateGuess(list_of_decoders[i]->getErrorHat(), i)) {
            bestSuccess = success;
        }
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
