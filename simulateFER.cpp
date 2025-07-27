/*
 * Copyright 2023 Sisi Miao, Communications Engineering Lab @ KIT
 *
 * SPDX-License-Identifier: MIT
 *
 * This file accompanies the paper
 *     S. Miao, A. Schnerring, H. Li and L. Schmalen,
 *     "Neural belief propagation decoding of quantum LDPC codes using overcomplete check matrices,"
 *     Proc. IEEE Inform. Theory Workshop (ITW), Saint-Malo, France, Apr. 2023, https://arxiv.org/abs/2212.10245
 */
#include "stabilizerCodes.h"
#include "ensembleDecoder.h"

#include "helpers.h"

#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char *argv[]) {
    unsigned n = 48;
    unsigned k = 6;
    unsigned m = 2000;

    int decIterNum = 6;
    bool trained = true;
    double ep0 = 0.3;
    stabilizerCodesType codeType = stabilizerCodesType::GeneralizedBicycle;
		AttributesDecoder list(n, k, m, codeType, trained);
		std::vector<std::string> decoder_names{"main"};

    fileReader matrix_supplier(n, k, m, codeType, trained, "Vanilla");
    fileReader matrix_supplier_dummy(n, k, m, codeType, trained, "Vanilla");
    matrix_supplier.check_symplectic();

    fileReader matrix_pruned1(n, k, m, codeType, trained, "Tick");
    fileReader matrix_pruned2(n, k, m, codeType, trained, "Trick");
    fileReader matrix_pruned3(n, k, m, codeType, trained, "Track");

    constexpr int default_max_frame_errors = 300;
    constexpr int default_max_decoded_words = 45000000;
    //    double ep_list[] =
    //    {0.14,0.13,0.12,0.11,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.009,0.008,0.007,0.006,0.005};
    const std::vector<double> default_ep_list{
        0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
    };

    const auto arguments =
        helpers::parse_arguments(argc, argv, default_ep_list, default_max_frame_errors, default_max_decoded_words);
    if (arguments.print_help) {
        helpers::print_help(arguments.progname);
        return 0;
    }
    const std::vector<double> &ep_list = std::move(arguments.epsilons);
    const auto max_frame_errors = arguments.maximum_frame_errors;
    const auto max_decoded_words = arguments.maximum_decoded_words;

    std::cout << "% [[" << n << "," << k << +"]], " << m << " checks, " << decIterNum << " iter ";
    if (trained)
        std::cout << ",trained";
    std::cout << "\n"
              << "% collect " << max_frame_errors << " frame errors or " << max_decoded_words
              << " decoded error patterns\n";

    //    omp_set_num_threads(1);
    for (double epsilon : ep_list) {
        double total_decoding = 0;
        double failure = 0;

#pragma omp parallel
        {
            while (failure <= max_frame_errors && total_decoding <= max_decoded_words) {
                std::vector<bool> success;
                stabilizerCodes errorCreator(n, k, m, codeType, matrix_supplier_dummy, trained);
          			errorCreator.add_error_given_epsilon(epsilon);

         				ensembleDecoder dude(decoder_names, list, matrix_supplier);


                stabilizerCodes Tick(n, k, m, codeType, matrix_pruned1, trained, errorCreator.getErrorString(), errorCreator.getError());
                stabilizerCodes Trick(n, k, m, codeType, matrix_pruned2, trained, errorCreator.getErrorString(), errorCreator.getError());
                stabilizerCodes Track(n, k, m, codeType, matrix_pruned3, trained, errorCreator.getErrorString(), errorCreator.getError());

                // stabilizerCodes sisi(n, k, m, codeType, matrix_supplier, trained, errorCreator.getErrorString(), errorCreator.getError());
								// dude.add_decoder(sisi);
								dude.add_decoder(Tick);
								// dude.add_decoder(Trick);
								// dude.add_decoder(Track);
								
								success = dude.decodeAllPaths(decIterNum, ep0);

								// for(int i = 0 ; i < dude.list_of_decoders.size(); i++){
								// 		success = dude.list_of_decoders[i].decode(decIterNum, ep0);
								// 		if (success[1]) {
								// 			break;
								// 		}
								// }
        				
#pragma omp critical
                {
									if (!success[1]) 
									{
											failure += 1;
											// Print each string in the error string vector, separated by spaces
											const auto& errorStrings = dude.list_of_decoders[0].getErrorString();
											for (const auto& s : errorStrings) {
													std::cout << s << " ";
											}
											std::cout << std::endl;

									}
									total_decoding += 1;
											}
            }
        }
        // std::cout << "% FE " << failure << ", total dec. " << total_decoding << "\\\\" << std::endl;
        // std::cout << epsilon << " " << (failure / total_decoding) << "\\\\" << std::endl;
						if (epsilon == 0.06){
							break;
							return 0;
				}
    }
    return 0;
}
