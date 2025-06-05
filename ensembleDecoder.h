#include "stabilizerCodes.h"
#include <vector>

class ensembleDecoder{
	public:
		ensembleDecoder(std::vector<unsigned> syn);

		std::vector<unsigned> returnGuess(){return mostLikely;};
		void updateGuess(std::vector<unsigned> syn);

	private:
		std::vector<unsigned> mostLikely;
};
