#include <vector>
#include <tuple>

#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <Eigen/Dense>
#include <numeric>
#include <algorithm>

#include "basis.h"
#include "tensored_basis.h"

namespace tensored_basis {
	// Private
	void tensored_basis::doublons_triplons_precompute(){
		// Allocate memory		
		
		uint expected_nonzeros = hilbert_space_dimension; 
		if(number_of_sites - NA - NB - NC >= 0){
			expected_nonzeros -= - boost::math::factorial<double>(number_of_sites) / ( boost::math::factorial<double>(NA)*boost::math::factorial<double>(NB)*boost::math::factorial<double>(NC)*boost::math::factorial<double>(number_of_sites-NA-NB-NC) );
		}
		
		total_multiple_occupancies.reserve(expected_nonzeros);
		
		uint expected_nonzeros_site_resolved;
		if(NA>=1 and NB>=1 and NC>=1){
			 expected_nonzeros_site_resolved = 	boost::math::binomial_coefficient<double>(number_of_sites-1, NA-1) *
												boost::math::binomial_coefficient<double>(number_of_sites-1, NB-1) *
												boost::math::binomial_coefficient<double>(number_of_sites-1, NC-1);
		}
		else{
			expected_nonzeros_site_resolved = 1;
		}
												
		site_resolved_triple_occupancies.resize(number_of_sites);
		for(int site=0; site<number_of_sites; site++){
			site_resolved_triple_occupancies[site].reserve(expected_nonzeros_site_resolved);
		}
		
		// Fill in the matrices
		for(uint FC=0; FC<hilbert_space_C_dimension; FC++){
			for(uint FB=0; FB<hilbert_space_B_dimension; FB++){
				for(uint FA=0; FA<hilbert_space_A_dimension; FA++){
					uint j = flatten(FA, FB, FC);
					
					uint doublons=0;
					uint triplons=0;
					
					for(int site=0; site<number_of_sites; site++){
						
						if( bA.basisStates(FA, site) == 1 and bB.basisStates(FB, site) == 1 and bC.basisStates(FC, site) == 1 ){
							triplons += 1;
							doublons += 3;
							
							site_resolved_triple_occupancies[site].push_back(j);
						}
						else if( bA.basisStates(FA, site) == 1 and bB.basisStates(FB, site) == 1 ){
							doublons += 1;
						}
						else if( bA.basisStates(FA, site) == 1 and bC.basisStates(FC, site) == 1 ){
							doublons += 1;
						}
						else if( bB.basisStates(FB, site) == 1 and bC.basisStates(FC, site) == 1 ){
							doublons += 1;
						}
					}
					
					if(doublons!=0 or triplons!=0){
						total_multiple_occupancies.push_back( std::make_tuple(j, doublons, triplons) );
					}
				}
			}
		}
				
		// Check whether the allocated memory was enough
		for(int site=0; site<number_of_sites; site++){
			if( site_resolved_triple_occupancies[site].size() > expected_nonzeros_site_resolved ){
				std::cout << "\n\nTriple occupation vector has been resized!\n\n";
			}
			
		}
		
		if( total_multiple_occupancies.size() > expected_nonzeros ){
			std::cout << "\n\nVector containing doublons/triplons has been resized!\n\n";
		}
		

		return;
	}
	
	// Public
	tensored_basis::tensored_basis(){
		return;
	}
			
	tensored_basis::tensored_basis(int na, int nb, int nc, int length, bool V){
		// Verbose mode
		verbose = V;
		
		// Basis parameters			
		NA = na;
		NB = nb;
		NC = nc;
				
		number_of_sites = length;
				
		// Basis initialization
		basis::basis tmpA(NA, number_of_sites, 1, verbose);
		bA = tmpA;
		
		basis::basis tmpB(NB, number_of_sites, 1, verbose);
		bB = tmpB;
		
		basis::basis tmpC(NC, number_of_sites, 1, verbose);
		bC = tmpC;
		
		// Total Hilbert space dimension
		hilbert_space_A_dimension = bA.getSize();
		hilbert_space_B_dimension = bB.getSize();
		hilbert_space_C_dimension = bC.getSize();
		
		hilbert_space_dimension = hilbert_space_A_dimension * hilbert_space_B_dimension * hilbert_space_C_dimension;
		
		// Check
		for(uint FC=0; FC<hilbert_space_C_dimension; FC++){
			for(uint FB=0; FB<hilbert_space_B_dimension; FB++){
				for(uint FA=0; FA<hilbert_space_A_dimension; FA++){
					uint j = flatten(FA, FB, FC);
					std::tuple<uint, uint, uint> t = deflatten(j);
					
					if( FA != std::get<0>(t) or FB != std::get<1>(t) or FC != std::get<2>(t) ){
						std::cout << "Flattening error!\n";
						exit(1);
					}
				}
			}
		}
		
		// Table out the doubly and triply occupied sites
		doublons_triplons_precompute();
		
		return;
	}

	// Flatten - deflatten
	uint tensored_basis::flatten(uint FA, uint FB, uint FC){
		return FA + hilbert_space_A_dimension * (FB + hilbert_space_B_dimension * FC);
	}

	std::tuple<uint,uint,uint> tensored_basis::deflatten(uint state_index){
		uint FA = state_index % hilbert_space_A_dimension;
		uint FB = ( state_index / hilbert_space_A_dimension ) % hilbert_space_B_dimension;
		uint FC = ( state_index / hilbert_space_A_dimension ) / hilbert_space_B_dimension;
		
		return std::tuple(FA,FB,FC);
	}

}