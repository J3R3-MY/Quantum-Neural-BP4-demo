#include "stabilizerCodes.h"
#include <string>
#include <vector>

struct DecoderAttributes{
	public:
		DecoderAttributes(unsigned n, unsigned k, unsigned m, stabilizerCodesType codeType, bool trained)
		        : n(n), k(k), m(m), codeType(codeType), trained(trained) {}

		unsigned n;
		unsigned k;
		unsigned m;
		stabilizerCodesType codeType;
		bool trained;		
};

class ensembleDecoder{
	public:
		ensembleDecoder(std::vector<std::string> decoder_names, DecoderAttributes list, double epsilon, fileReader& supplier);

		std::vector<unsigned> returnGuess(){return estimatedError;};
		bool updateGuess(const std::vector<unsigned>& newCandidate, int index);

		std::vector<std::string> list_of_specifiers;
		std::vector<stabilizerCodes> list_of_decoders;
		DecoderAttributes list;

		// These two objects are needed for using some stabilizerCode functionality
    stabilizerCodes main;

		std::vector<bool> decodeAllPaths(unsigned int L, double epsilon);
		bool succesfully_decoded();

	private:
		std::vector<unsigned> estimatedError;
		int bestDecoder;
		std::vector<std::vector<double>> estimatedTaus;
		void initalize_decoders(double epsilon);
};
