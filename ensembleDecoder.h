#include "stabilizerCodes.h"
#include <string>
#include <vector>

struct AttributesDecoder{
	public:
		AttributesDecoder(unsigned n, unsigned k, unsigned m, stabilizerCodesType codeType, bool trained)
		        : n(n), k(k), m(m), codeType(codeType), trained(trained) {}

		unsigned n;
		unsigned k;
		unsigned m;
		stabilizerCodesType codeType;
		bool trained;		
};

// struct TelemtryDecoder{
// 	public: 
// 		TelemtryDecoder(std::vector<std::string> error)
// 										: errorString(error) {}
// 		std::vector<std::string> errorString;
// };

class ensembleDecoder{
	public:
		ensembleDecoder(std::vector<std::string> decoder_names, AttributesDecoder list, fileReader& supplier);

		std::vector<unsigned> returnGuess(){return estimatedError;};
		bool updateGuess(const std::vector<unsigned>& newCandidate, int index);

		std::vector<std::string> list_of_specifiers;
		std::vector<stabilizerCodes> list_of_decoders;
		AttributesDecoder list;

		// These two objects are needed for using some stabilizerCode functionality
    stabilizerCodes main;

		std::vector<bool> decodeAllPaths(unsigned int L, double epsilon);
		bool succesfully_decoded();
		void add_decoder(stabilizerCodes decoder);

	private:
		std::vector<unsigned> estimatedError;
		int bestDecoder;
		std::vector<std::vector<double>> estimatedTaus;
		void initalize_decoders(double epsilon);
};
