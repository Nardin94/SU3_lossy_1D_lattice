#include <vector>
#include <tuple>

#include <cmath>

#include <random>

#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

#include <numeric>
#include <algorithm>
#include <cmath>

#include <chrono>

#include <unordered_set>

#include <slepceps.h>
#include <petsc.h>
#include <petscmat.h>

#include "basis.h"
#include "tensored_basis.h"
#include "effective_hamiltonian.h"

using namespace std::complex_literals;

namespace effective_hamiltonian {
	// Private
		void effective_hamiltonian::print_fock(int F){
			std::tuple<uint,uint,uint> deflattened_fock = total_basis.deflatten(F);

			Eigen::VectorXi configurationA = total_basis.bA.basisStates.row( std::get<0>(deflattened_fock) );
			Eigen::VectorXi configurationB = total_basis.bB.basisStates.row( std::get<1>(deflattened_fock) );
			Eigen::VectorXi configurationC = total_basis.bB.basisStates.row( std::get<1>(deflattened_fock) );

			std::cout << "|";
			for(uint a=0; a<configurationA.size(); a++){
				std::cout << configurationA[a] << " ";
			}
			std::cout << "\b⟩⊗|";
			for(uint b=0; b<configurationB.size(); b++){
				std::cout << configurationB[b] << " ";
			}
			std::cout << "\b⟩⊗|";
			for(uint c=0; c<configurationC.size(); c++){
				std::cout << configurationC[c] << " ";
			}
			std::cout << "\b⟩\n";				

			return;
		}

		Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> effective_hamiltonian::hopping_hamiltonian_construct(){
			// Hopping hamiltonian linear dimension
			uint hilbert_space_dimension = total_basis.hilbert_space_dimension;
			
			// Each particle in any of the Fock states can at most hop to two positions
			uint ansatzSize = hilbert_space_dimension * 2 * (NA+NB+NC);
			std::vector<Eigen::Triplet<double>> list{};
			list.reserve( ansatzSize );
			
			// Set-up the Hamiltonian
            // Hop in sublattice A
            for(uint FA_right=0; FA_right<total_basis.hilbert_space_A_dimension; FA_right++){
				Eigen::VectorXi configuration = total_basis.bA.basisStates.row(FA_right);
					
				for(int oldPosition=0; oldPosition<number_of_sites; oldPosition++){
					// Look whether there is a particle at the given site
					if( configuration[oldPosition] > 0 ){
						// Hop along +x
						if(oldPosition+1 < number_of_sites){
							// Compute new position
							int newPosition = oldPosition+1;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){								
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FA_left = total_basis.bA.tagSearch( total_basis.bA.tag(configuration) );

                                // Push back all the matrix elements
                                for(uint FB=0; FB<total_basis.hilbert_space_B_dimension; FB++){
                                    for(uint FC=0; FC<total_basis.hilbert_space_C_dimension; FC++){
                                        uint left_fock = total_basis.flatten(FA_left, FB, FC);
                                        uint right_fock = total_basis.flatten(FA_right, FB, FC);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate));
                                    }
                                }

								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}

						if(periodic_boundary_conditions and oldPosition+1 == number_of_sites){
							// Compute new position
							int newPosition = 0;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FA_left = total_basis.bA.tagSearch( total_basis.bA.tag(configuration) );

                                // Push back all the matrix elements
								int fermionic_sign = (NA-1)%2 == 0 ? 1 : -1;
                                for(uint FB=0; FB<total_basis.hilbert_space_B_dimension; FB++){
                                    for(uint FC=0; FC<total_basis.hilbert_space_C_dimension; FC++){
                                        uint left_fock = total_basis.flatten(FA_left, FB, FC);
                                        uint right_fock = total_basis.flatten(FA_right, FB, FC);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate * fermionic_sign));
                                    }
                                }
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}
											
						// Hop along -x
						if(oldPosition > 0){
							// Compute new position
							int newPosition = oldPosition-1;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){								
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FA_left = total_basis.bA.tagSearch( total_basis.bA.tag(configuration) );

                                // Push back all the matrix elements
                                for(uint FB=0; FB<total_basis.hilbert_space_B_dimension; FB++){
                                    for(uint FC=0; FC<total_basis.hilbert_space_C_dimension; FC++){
                                        uint left_fock = total_basis.flatten(FA_left, FB, FC);
                                        uint right_fock = total_basis.flatten(FA_right, FB, FC);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate));
                                    }
                                }
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}

						if(periodic_boundary_conditions and oldPosition == 0){					
							// Compute new position
							int newPosition = number_of_sites-1;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){								
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FA_left = total_basis.bA.tagSearch( total_basis.bA.tag(configuration) );

                                // Push back all the matrix elements
								int fermionic_sign = (NA-1)%2 == 0 ? 1 : -1;
                                for(uint FB=0; FB<total_basis.hilbert_space_B_dimension; FB++){
                                    for(uint FC=0; FC<total_basis.hilbert_space_C_dimension; FC++){
                                        uint left_fock = total_basis.flatten(FA_left, FB, FC);
                                        uint right_fock = total_basis.flatten(FA_right, FB, FC);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate * fermionic_sign));
                                    }
                                }
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}
						
					}
				}
            }

            // Hop in sublattice B
            for(uint FB_right=0; FB_right<total_basis.hilbert_space_B_dimension; FB_right++){
				Eigen::VectorXi configuration = total_basis.bB.basisStates.row(FB_right);
					
				for(int oldPosition=0; oldPosition<number_of_sites; oldPosition++){
					// Look whether there is a particle at the given site
					if( configuration[oldPosition] > 0 ){
						// Hop along +x
						if(oldPosition+1 < number_of_sites){
							// Compute new position
							int newPosition = oldPosition+1;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){								
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FB_left = total_basis.bB.tagSearch( total_basis.bB.tag(configuration) );

                                // Push back all the matrix elements
                                for(uint FA=0; FA<total_basis.hilbert_space_A_dimension; FA++){
                                    for(uint FC=0; FC<total_basis.hilbert_space_C_dimension; FC++){
                                        uint left_fock = total_basis.flatten(FA, FB_left, FC);
                                        uint right_fock = total_basis.flatten(FA, FB_right, FC);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate));
                                    }
                                }

								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}

						if(periodic_boundary_conditions and oldPosition+1 == number_of_sites){
							// Compute new position
							int newPosition = 0;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FB_left = total_basis.bB.tagSearch( total_basis.bB.tag(configuration) );

                                // Push back all the matrix elements
								int fermionic_sign = (NB-1)%2 == 0 ? 1 : -1;
                                for(uint FA=0; FA<total_basis.hilbert_space_A_dimension; FA++){
                                    for(uint FC=0; FC<total_basis.hilbert_space_C_dimension; FC++){
                                        uint left_fock = total_basis.flatten(FA, FB_left, FC);
                                        uint right_fock = total_basis.flatten(FA, FB_right, FC);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate * fermionic_sign));
                                    }
                                }
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}
											
						// Hop along -x
						if(oldPosition > 0){
							// Compute new position
							int newPosition = oldPosition-1;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){								
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FB_left = total_basis.bB.tagSearch( total_basis.bB.tag(configuration) );

                                // Push back all the matrix elements
                                for(uint FA=0; FA<total_basis.hilbert_space_A_dimension; FA++){
                                    for(uint FC=0; FC<total_basis.hilbert_space_C_dimension; FC++){
                                        uint left_fock = total_basis.flatten(FA, FB_left, FC);
                                        uint right_fock = total_basis.flatten(FA, FB_right, FC);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate));
                                    }
                                }
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}

						if(periodic_boundary_conditions and oldPosition == 0){					
							// Compute new position
							int newPosition = number_of_sites-1;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){								
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FB_left = total_basis.bB.tagSearch( total_basis.bB.tag(configuration) );

                                // Push back all the matrix elements
								int fermionic_sign = (NB-1)%2 == 0 ? 1 : -1;
                                for(uint FA=0; FA<total_basis.hilbert_space_A_dimension; FA++){
                                    for(uint FC=0; FC<total_basis.hilbert_space_C_dimension; FC++){
                                        uint left_fock = total_basis.flatten(FA, FB_left, FC);
                                        uint right_fock = total_basis.flatten(FA, FB_right, FC);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate * fermionic_sign));
                                    }
                                }
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}
						
					}
				}
            }

            // Hop in sublattice C
            for(uint FC_right=0; FC_right<total_basis.hilbert_space_C_dimension; FC_right++){
				Eigen::VectorXi configuration = total_basis.bC.basisStates.row(FC_right);
					
				for(int oldPosition=0; oldPosition<number_of_sites; oldPosition++){
					// Look whether there is a particle at the given site
					if( configuration[oldPosition] > 0 ){
						// Hop along +x
						if(oldPosition+1 < number_of_sites){
							// Compute new position
							int newPosition = oldPosition+1;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){								
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FC_left = total_basis.bC.tagSearch( total_basis.bC.tag(configuration) );

                                // Push back all the matrix elements
                                for(uint FA=0; FA<total_basis.hilbert_space_A_dimension; FA++){
                                    for(uint FB=0; FB<total_basis.hilbert_space_B_dimension; FB++){
                                        uint left_fock = total_basis.flatten(FA, FB, FC_left);
                                        uint right_fock = total_basis.flatten(FA, FB, FC_right);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate));
                                    }
                                }

								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}

						if(periodic_boundary_conditions and oldPosition+1 == number_of_sites){
							// Compute new position
							int newPosition = 0;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FC_left = total_basis.bC.tagSearch( total_basis.bC.tag(configuration) );

                                // Push back all the matrix elements
								int fermionic_sign = (NC-1)%2 == 0 ? 1 : -1;
                                for(uint FA=0; FA<total_basis.hilbert_space_A_dimension; FA++){
                                    for(uint FB=0; FB<total_basis.hilbert_space_B_dimension; FB++){
                                        uint left_fock = total_basis.flatten(FA, FB, FC_left);
                                        uint right_fock = total_basis.flatten(FA, FB, FC_right);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate * fermionic_sign));
                                    }
                                }
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}
											
						// Hop along -x
						if(oldPosition > 0){
							// Compute new position
							int newPosition = oldPosition-1;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){								
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FC_left = total_basis.bC.tagSearch( total_basis.bC.tag(configuration) );

                                // Push back all the matrix elements
                                for(uint FA=0; FA<total_basis.hilbert_space_A_dimension; FA++){
                                    for(uint FB=0; FB<total_basis.hilbert_space_B_dimension; FB++){
                                        uint left_fock = total_basis.flatten(FA, FB, FC_left);
                                        uint right_fock = total_basis.flatten(FA, FB, FC_right);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate));
                                    }
                                }
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}

						if(periodic_boundary_conditions and oldPosition == 0){					
							// Compute new position
							int newPosition = number_of_sites-1;
								
							// Check wether I can create a new particle at the new site
							if( configuration[newPosition] == 0 ){								
								// Move the partice
								configuration[oldPosition] -= 1;
								configuration[newPosition] += 1;
									
								// Compute the tag, and look it up
								uint FC_left = total_basis.bC.tagSearch( total_basis.bC.tag(configuration) );

                                // Push back all the matrix elements
								int fermionic_sign = (NC-1)%2 == 0 ? 1 : -1;
                                for(uint FA=0; FA<total_basis.hilbert_space_A_dimension; FA++){
                                    for(uint FB=0; FB<total_basis.hilbert_space_B_dimension; FB++){
                                        uint left_fock = total_basis.flatten(FA, FB, FC_left);
                                        uint right_fock = total_basis.flatten(FA, FB, FC_right);

                                        list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate * fermionic_sign));
                                    }
                                }
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
							}
						}
						
					}
				}
            }

			
			// The number of non-zero matrix elements sets the matrix dimension
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> H(hilbert_space_dimension, hilbert_space_dimension);
			H.setFromTriplets(list.begin(), list.end());
				
			if( list.size() > ansatzSize ){
				std::cout << "\n\nAttention: a resize was needed\n\n";
			}

			return H;
		}

		Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> effective_hamiltonian::confinement_hamiltonian_construct(){
			uint hilbert_space_dimension = total_basis.hilbert_space_dimension;
            
            // Generate the hopping matrix elements
			std::vector<Eigen::Triplet<std::complex<double>>> list{};
			list.reserve( hilbert_space_dimension );

            for(uint FA=0; FA<total_basis.hilbert_space_A_dimension; FA++){
				Eigen::VectorXi configurationA = total_basis.bA.basisStates.row(FA);

				double conf_A = 0.;
				for(int position=0; position<number_of_sites; position++){
					if( configurationA[position] > 0 ){
						conf_A += cosine_potential_strength * std::cos(2.*M_PI/number_of_sites * position) * configurationA[position];
					}
				}


				for(uint FB=0; FB<total_basis.hilbert_space_B_dimension; FB++){
					Eigen::VectorXi configurationB = total_basis.bB.basisStates.row(FB);				

					double conf_B = 0.;
					for(int position=0; position<number_of_sites; position++){
						if( configurationB[position] > 0 ){
							conf_B += cosine_potential_strength * std::cos(2.*M_PI/number_of_sites * position) * configurationB[position];
						}
					}

					for(uint FC=0; FC<total_basis.hilbert_space_C_dimension; FC++){
						Eigen::VectorXi configurationC = total_basis.bC.basisStates.row(FC);

						double conf_C = 0.;
						for(int position=0; position<number_of_sites; position++){
							if( configurationC[position] > 0 ){
								conf_C += cosine_potential_strength * std::cos(2.*M_PI/number_of_sites * position) * configurationC[position];
							}
						}

                        uint fock = total_basis.flatten(FA, FB, FC);

						list.push_back(Eigen::Triplet<std::complex<double>>(fock, fock, conf_A + conf_B + conf_C));
					}
				}
			}
						
			// The number of non-zero matrix elements sets the matrix dimension
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> Hsp(hilbert_space_dimension, hilbert_space_dimension);
			Hsp.setFromTriplets(list.begin(), list.end());
			
			return Hsp;
		}

		Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> effective_hamiltonian::interaction_hamiltonian_construct(){
			uint hilbert_space_dimension = total_basis.hilbert_space_dimension;
            
            // Generate the hopping matrix elements
			std::vector<Eigen::Triplet<std::complex<double>>> list{};
			list.reserve( total_basis.total_multiple_occupancies.size() );

			for(auto& v : total_basis.total_multiple_occupancies){
                uint fock = std::get<0>(v);
                std::complex<double> matrix_element = two_body_contact * std::get<1>(v) + (three_body_contact-0.5i*loss_rate) * (double)std::get<2>(v);

                list.push_back(Eigen::Triplet<std::complex<double>>(fock, fock, matrix_element));
			}
						
			// The number of non-zero matrix elements sets the matrix dimension
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> Hsp(hilbert_space_dimension, hilbert_space_dimension);
			Hsp.setFromTriplets(list.begin(), list.end());
			
			return Hsp;
		}

		int effective_hamiltonian::flatten_particle_shifts(int da, int db, int dc){
			std::tuple<int, int, int> target = std::make_tuple(da, db, dc);

			auto it = std::find(valid_shifts.begin(), valid_shifts.end(), target);

			if(it != valid_shifts.end()){
				return std::distance(valid_shifts.begin(), it);
			}
		
			return -1; // Invalid tuple
		}

		double effective_hamiltonian::symmetric_coefficients(int i, int j, int k){
			// If they contain an odd number of 2,5,7 then the d(i,j,k) vanishes
			int count257 = 0;
			if( i==2 or i==5 or i==7 ) count257++;
			if( j==2 or j==5 or j==7 ) count257++;
			if( k==2 or k==5 or k==7 ) count257++;

			if( count257%2 != 0 ){
				return 0;
			}

			// Sort the three numbers so that they are increasing
			if(i > j) std::swap(i, j);
    		if(j > k) std::swap(j, k);
    		if(i > j) std::swap(i, j);

			// Use a look-up table. If the coefficient is not found, it is zero
			std::tuple<int, int, int> target = std::make_tuple(i, j, k);

			auto it = std::find_if(d_coeffs.begin(), d_coeffs.end(), [&](const std::pair<std::tuple<int, int, int>, double>& entry){
				return entry.first == target;
			});

			if(it != d_coeffs.end()) {
				return it->second;
			}
			
			return 0;
		}

		bool effective_hamiltonian::isSelfAdjoint(const Eigen::SparseMatrix<std::complex<double>>& mat, double tolerance) {
			if(mat.rows() != mat.cols()){
				return false;
			}

			// Compare each entry with its transpose counterpart
			for(int k = 0; k < mat.outerSize(); ++k){
				for(Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(mat, k); it; ++it){
					std::complex<double> transposeValue = std::conj( mat.coeff(it.col(), it.row()) );
					if(std::abs(it.value() - transposeValue) > tolerance){
						return false;
					}
				}
			}

			return true;
		}

		std::vector< Eigen::SparseMatrix<std::complex<double>> > effective_hamiltonian::su3_algebra_construct( std::vector<tensored_basis::tensored_basis> &extended_basis, std::vector<std::vector<int>> &flatten, std::vector<std::tuple<int, int, int, int>> &unflatten ){
			// The state is ordered as creations(A) creations(B) creations(C) |vac>
			// When acting with B*[j]A[j], the B has to pass through A[j] creations(A) -> (-1)^(NA+1)
			// When acting with C*[j]A[j], the C has to pass through A[j] creations(A) creations(B) -> (-1)^(NA+1+NB)

			// When acting with C*[j]B[j], the C has to pass through creations(A) B[j] creations(B), while the B through creations(A) -> (-1)^(NA+NB+1+NA)=(-1)^(NB+1)

			std::vector< Eigen::SparseMatrix<std::complex<double>> > lambda_representations(8);

			// Reserve space for the matrices
			uint extendedHilberSpace_size = unflatten.size();
			uint ansatzSize = extendedHilberSpace_size * 4 * (NA+NB+NC);

			std::vector<Eigen::Triplet<std::complex<double>>> list1{}, list2{}, list3{}, list4{}, list5{}, list6{}, list7{}, list8{};
			list1.reserve( ansatzSize );
			list2.reserve( ansatzSize );
			list3.reserve( extendedHilberSpace_size );
			list4.reserve( ansatzSize );
			list5.reserve( ansatzSize );
			list6.reserve( ansatzSize );
			list7.reserve( ansatzSize );
			list8.reserve( extendedHilberSpace_size );

			// Fill in Lambda1 and Lambda2: they change (NA,NB) -> (NA+1,NB-1) or (NA-1,NB+1)
			// Each particle in any of the Fock states can at most hop to two positions

            for(uint eF_right=0; eF_right<extendedHilberSpace_size; eF_right++){
				std::tuple<int,int,int,int> compressed = unflatten[eF_right];
				int da = std::get<0>(compressed);
				int db = std::get<1>(compressed);
				int dc = std::get<2>(compressed);

				int index_left = flatten_particle_shifts(da-1,db+1,dc);

				if( index_left != -1 ){
					int index_right = flatten_particle_shifts(da, db, dc);

					int F_right = std::get<3>(compressed);
					std::tuple<uint,uint,uint> deflattened_fock = extended_basis[index_right].deflatten(F_right);

					Eigen::VectorXi configurationA = extended_basis[index_right].bA.basisStates.row( std::get<0>(deflattened_fock) );
					Eigen::VectorXi configurationB = extended_basis[index_right].bB.basisStates.row( std::get<1>(deflattened_fock) );

					// Hop from A to B; the reverse process is added to the list by conjugation
					for(int site=0; site<number_of_sites; site++){
						// Look whether there is a particle at the given site in one sublattice, and none in the other
						if( configurationA[site] > 0 and configurationB[site] == 0 ){							

							// Count the number of exchanges
							int fermionic_sign, fermions = (NA+da)+1;
							for(int s=0; s<site; s++){
								fermions += configurationA[s] + configurationB[s];
							}
							fermionic_sign = fermions%2 == 0 ? 1 : -1;

							// Move the partice
							configurationA[site] -= 1;
							configurationB[site] += 1;

							// Compute the tag, and look it up
							uint FA_left = extended_basis[index_left].bA.tagSearch( extended_basis[index_left].bA.tag(configurationA) );
							uint FB_left = extended_basis[index_left].bB.tagSearch( extended_basis[index_left].bB.tag(configurationB) );

							// Push back all the matrix elements
							uint F_left = extended_basis[index_left].flatten(FA_left, FB_left, std::get<2>(deflattened_fock));
							uint eF_left = flatten[index_left][F_left];

							list1.push_back(Eigen::Triplet<std::complex<double>>(eF_left, eF_right, 0.5 * fermionic_sign));
							list1.push_back(Eigen::Triplet<std::complex<double>>(eF_right, eF_left, 0.5 * fermionic_sign));

							list2.push_back(Eigen::Triplet<std::complex<double>>(eF_left, eF_right, 0.5i * fermionic_sign));
							list2.push_back(Eigen::Triplet<std::complex<double>>(eF_right, eF_left, -0.5i * fermionic_sign));

							// Move the particle back
							configurationB[site] -= 1;
							configurationA[site] += 1;				
						}
					}

				}
            }

			lambda_representations[0].resize(extendedHilberSpace_size, extendedHilberSpace_size);
			lambda_representations[0].setFromTriplets(list1.begin(), list1.end());

			lambda_representations[1].resize(extendedHilberSpace_size, extendedHilberSpace_size);
			lambda_representations[1].setFromTriplets(list2.begin(), list2.end());

			if( list1.size() > ansatzSize or list2.size() > ansatzSize ){
				std::cout << "\n\nAttention: a resize was needed\n\n";
			}

			// Fill in Lambda4 and Lambda5: they change (NA,NC) -> (NA+1,NC-1) or (NA-1,NC+1)

            for(uint eF_right=0; eF_right<extendedHilberSpace_size; eF_right++){
				std::tuple<int,int,int,int> compressed = unflatten[eF_right];
				int da = std::get<0>(compressed);
				int db = std::get<1>(compressed);
				int dc = std::get<2>(compressed);

				int index_left = flatten_particle_shifts(da-1,db,dc+1);

				if( index_left != -1 ){
					int index_right = flatten_particle_shifts(da, db, dc);

					int F_right = std::get<3>(compressed);
					std::tuple<uint,uint,uint> deflattened_fock = extended_basis[index_right].deflatten(F_right);

					Eigen::VectorXi configurationA = extended_basis[index_right].bA.basisStates.row( std::get<0>(deflattened_fock) );
					Eigen::VectorXi configurationC = extended_basis[index_right].bC.basisStates.row( std::get<2>(deflattened_fock) );
						
					// Hop from A to C; the reverse process is added to the list by conjugation
					for(int site=0; site<number_of_sites; site++){
						// Look whether there is a particle at the given site in one sublattice, and none in the other
						if( configurationA[site] > 0 and configurationC[site] == 0 ){							

							// Count the number of exchanges
							int fermionic_sign, fermions = NA+da+NB+db+1;
							for(int s=0; s<site; s++){
								fermions += configurationA[s] + configurationC[s];
							}
							fermionic_sign = fermions%2 == 0 ? 1 : -1;
							
							// Move the partice
							configurationA[site] -= 1;
							configurationC[site] += 1;

							// Compute the tag, and look it up
							uint FA_left = extended_basis[index_left].bA.tagSearch( extended_basis[index_left].bA.tag(configurationA) );
							uint FC_left = extended_basis[index_left].bC.tagSearch( extended_basis[index_left].bC.tag(configurationC) );

							// Push back all the matrix elements
							uint F_left = extended_basis[index_left].flatten(FA_left, std::get<1>(deflattened_fock), FC_left);
							uint eF_left = flatten[index_left][F_left];

							list4.push_back(Eigen::Triplet<std::complex<double>>(eF_left, eF_right, 0.5 * fermionic_sign));
							list4.push_back(Eigen::Triplet<std::complex<double>>(eF_right, eF_left, 0.5 * fermionic_sign));

							list5.push_back(Eigen::Triplet<std::complex<double>>(eF_left, eF_right, 0.5i * fermionic_sign));
							list5.push_back(Eigen::Triplet<std::complex<double>>(eF_right, eF_left, -0.5i * fermionic_sign));
							

							// Move the particle back
							configurationC[site] -= 1;
							configurationA[site] += 1;				
						}
					}

				}
            }

			lambda_representations[3].resize(extendedHilberSpace_size, extendedHilberSpace_size);
			lambda_representations[3].setFromTriplets(list4.begin(), list4.end());

			lambda_representations[4].resize(extendedHilberSpace_size, extendedHilberSpace_size);
			lambda_representations[4].setFromTriplets(list5.begin(), list5.end());

			if( list4.size() > ansatzSize or list5.size() > ansatzSize ){
				std::cout << "\n\nAttention: a resize was needed\n\n";
			}

			// Fill in Lamdba6 and Lambda7: they change (NB,NC) -> (NB+1,NC-1) or (NB-1,NC+1)

            for(uint eF_right=0; eF_right<extendedHilberSpace_size; eF_right++){
				std::tuple<int,int,int,int> compressed = unflatten[eF_right];
				int da = std::get<0>(compressed);
				int db = std::get<1>(compressed);
				int dc = std::get<2>(compressed);

				int index_left = flatten_particle_shifts(da,db-1,dc+1);

				if( index_left != -1 ){
					int index_right = flatten_particle_shifts(da, db, dc);

					int F_right = std::get<3>(compressed);
					std::tuple<uint,uint,uint> deflattened_fock = extended_basis[index_right].deflatten(F_right);

					Eigen::VectorXi configurationB = extended_basis[index_right].bB.basisStates.row( std::get<1>(deflattened_fock) );
					Eigen::VectorXi configurationC = extended_basis[index_right].bC.basisStates.row( std::get<2>(deflattened_fock) );

					// Hop from B to C; the reverse process is added to the list by conjugation
					for(int site=0; site<number_of_sites; site++){
						// Look whether there is a particle at the given site in one sublattice, and none in the other
						if( configurationB[site] > 0 and configurationC[site] == 0 ){
							
							// Count the number of exchanges
							int fermionic_sign, fermions = NB+db+1;
							for(int s=0; s<site; s++){
								fermions += configurationB[s] + configurationC[s];
							}
							fermionic_sign = fermions%2 == 0 ? 1 : -1;

							// Move the partice
							configurationB[site] -= 1;
							configurationC[site] += 1;

							// Compute the tag, and look it up
							uint FB_left = extended_basis[index_left].bB.tagSearch( extended_basis[index_left].bB.tag(configurationB) );
							uint FC_left = extended_basis[index_left].bC.tagSearch( extended_basis[index_left].bC.tag(configurationC) );

							// Push back all the matrix elements
							uint F_left = extended_basis[index_left].flatten(std::get<0>(deflattened_fock), FB_left, FC_left);
							uint eF_left = flatten[index_left][F_left];

							list6.push_back(Eigen::Triplet<std::complex<double>>(eF_left, eF_right, 0.5 * fermionic_sign));
							list6.push_back(Eigen::Triplet<std::complex<double>>(eF_right, eF_left, 0.5 * fermionic_sign));

							list7.push_back(Eigen::Triplet<std::complex<double>>(eF_left, eF_right, 0.5i * fermionic_sign));
							list7.push_back(Eigen::Triplet<std::complex<double>>(eF_right, eF_left, -0.5i * fermionic_sign));
						
							// Move the particle back
							configurationC[site] -= 1;
							configurationB[site] += 1;
						}
					}

				}
            }

			lambda_representations[5].resize(extendedHilberSpace_size, extendedHilberSpace_size);
			lambda_representations[5].setFromTriplets(list6.begin(), list6.end());

			lambda_representations[6].resize(extendedHilberSpace_size, extendedHilberSpace_size);
			lambda_representations[6].setFromTriplets(list7.begin(), list7.end());

			if( list6.size() > ansatzSize or list7.size() > ansatzSize ){
				std::cout << "\n\nAttention: a resize was needed\n\n";
			}

			// Fill in Lambda3 and Lambda8: they do not change (NA,NB,NC)

            for(uint eF=0; eF<extendedHilberSpace_size; eF++){
				std::tuple<int,int,int,int> compressed = unflatten[eF];
				int da = std::get<0>(compressed);
				int db = std::get<1>(compressed);
				int dc = std::get<2>(compressed);

				int index = flatten_particle_shifts(da,db,dc);

				int F = std::get<3>(compressed);
				std::tuple<uint,uint,uint> deflattened_fock = extended_basis[index].deflatten(F);

				Eigen::VectorXi configurationA = extended_basis[index].bA.basisStates.row( std::get<0>(deflattened_fock) );
				Eigen::VectorXi configurationB = extended_basis[index].bB.basisStates.row( std::get<1>(deflattened_fock) );
				Eigen::VectorXi configurationC = extended_basis[index].bC.basisStates.row( std::get<2>(deflattened_fock) );

				double l3 = 0, l8 = 0;

				for(int site=0; site<number_of_sites; site++){		
					l3 += configurationA[site] - configurationB[site];
					l8 += configurationA[site] + configurationB[site] - 2.*configurationC[site];
				}
				l3 *= 0.5;
				l8 *= 0.5/sqrt(3.);

				list3.push_back(Eigen::Triplet<std::complex<double>>(eF, eF, l3));
				list8.push_back(Eigen::Triplet<std::complex<double>>(eF, eF, l8));
            }

			lambda_representations[2].resize(extendedHilberSpace_size, extendedHilberSpace_size);
			lambda_representations[2].setFromTriplets(list3.begin(), list3.end());

			lambda_representations[7].resize(extendedHilberSpace_size, extendedHilberSpace_size);
			lambda_representations[7].setFromTriplets(list8.begin(), list8.end());

			if( list3.size() > ansatzSize or list8.size() > ansatzSize ){
				std::cout << "\n\nAttention: a resize was needed\n\n";
			}

			// Check if they are correctly self-adjoint
			bool flag_exit = false;
			for(uint gen=0; gen<8; gen++){
				if( isSelfAdjoint(lambda_representations[gen]) == false ){
					std::cout << "L[" << gen << "] is not self-adjoint!!";
					flag_exit = true;
				}
			}
			if(flag_exit) exit(1);

			return lambda_representations;
		}

		void effective_hamiltonian::casimirs_construct( ){
			// 1. Construct an enlarged basis
			// 		We are interested into the following sectors: (NA,NB,NC), (NA+1,NB-1,NC), (NA+1,NB,NC-1), (NA-1,NB+1,NC), (NA-1,NB,NC+1), (NA,NB+1,NC-1), (NA,NB-1,NC+1)
			// 		We can define shift variables da, db, dc which take values in (-1,0,+1); the sum of these 3 variables is always zero
			//		We want to fold such a triple together in such a way that a contigous index is returned
			// and
			// 2. Build flattening and deflattening objects

			if(verbose){
				std::cout << "\t\tConstructing the enlarged basis... ";
				fflush(stdout);
			}

			std::vector<tensored_basis::tensored_basis> extended_basis(7);

			std::vector<std::tuple<int, int, int, int>> unflatten = {};
			std::vector<std::vector<int>> flatten(7);

			int counter = 0;
			for(int da=-1; da<2; da++){
				for(int db=-1; db<2; db++){
					for(int dc=-1; dc<2; dc++){

						int index = flatten_particle_shifts(da,db,dc);

						if( index != -1 and NA+da>=0 and NB+db>=0 and NC+dc>=0){
							tensored_basis::tensored_basis tmp(NA+da,NB+db,NC+dc,number_of_sites, false);

							extended_basis[index] = tmp;

							flatten[index].resize( tmp.hilbert_space_dimension );

							for(uint F=0; F<tmp.hilbert_space_dimension; F++){
								std::tuple<int, int, int, int> p = std::make_tuple(da, db, dc, F);
								unflatten.push_back(p);

								flatten[index][F] = counter++;           
							}
						}
					}
				}
			}

			int block_corner = flatten[flatten_particle_shifts(0,0,0)][0];

			// Check the flattening/unflattening
			for(uint uf=0; uf<unflatten.size(); uf++){
				std::tuple<int, int, int, int> p = unflatten[uf];

				int da = std::get<0>(p);
				int db = std::get<1>(p);
				int dc = std::get<2>(p);

				int index = flatten_particle_shifts(da,db,dc);

				int F = std::get<3>(p);

				if( flatten[index][F] != (int)uf ){
					std::cout << "Flattening/unflattening extended basis error!\n";
					exit(1);
				}
			}

			if(verbose) std::cout << "Done" << std::endl;

			// 3. Build representations of the Lambda matrices in this space
			if(verbose){
				std::cout << "\t\tConstructing representations of the Lambda matrices... ";
				fflush(stdout);
			}
			std::vector< Eigen::SparseMatrix<std::complex<double>> > Lambda = su3_algebra_construct( extended_basis, flatten, unflatten );
			if(verbose) std::cout << "Done" << std::endl;

			// 4. Set up the quadratic and cubic Casimirs of SU(3)
			if(verbose){
				std::cout << "\t\tSetting up the quadratic Casimir... ";
				fflush(stdout);
			}
			Eigen::SparseMatrix<std::complex<double>> extended_I2(unflatten.size(),unflatten.size());
			Eigen::SparseMatrix<std::complex<double>> extended_C2(unflatten.size(),unflatten.size());

			for(uint i=0; i<Lambda.size(); i++){
				Eigen::SparseMatrix<std::complex<double>> tmp = Lambda[i]*Lambda[i];

				extended_C2 += tmp;
				if(i<3) extended_I2 += tmp;
			}

			if(verbose) std::cout << "Done" << std::endl;

			if(verbose){
				std::cout << "\t\tSetting up the cubic Casimir... ";
				fflush(stdout);
			}
			Eigen::SparseMatrix<std::complex<double>>  extended_C3(unflatten.size(),unflatten.size());

			for(uint i=0; i<Lambda.size(); i++){
				for(uint j=0; j<Lambda.size(); j++){

					Eigen::SparseMatrix<std::complex<double>> tmp = Lambda[i] * Lambda[j];

					for(uint k=0; k<Lambda.size(); k++){
						double symm_coeff = symmetric_coefficients(i+1,j+1,k+1);

						if( std::abs(symm_coeff) > 1.e-12 ){
							extended_C3 += symm_coeff * tmp * Lambda[k];
						}
					}
				}
			}

			if(verbose) std::cout << "Done" << std::endl;

			// 5. Restrict them to the correct particle number sector
			if(verbose){
				std::cout << "\t\tRestricting them to the correct block... ";
				fflush(stdout);
			}
			I2 = extended_I2.block(block_corner, block_corner, total_basis.hilbert_space_dimension, total_basis.hilbert_space_dimension);
			I2.makeCompressed();

			C2 = extended_C2.block(block_corner, block_corner, total_basis.hilbert_space_dimension, total_basis.hilbert_space_dimension);
			C2.makeCompressed();

			C3 = extended_C3.block(block_corner, block_corner, total_basis.hilbert_space_dimension, total_basis.hilbert_space_dimension);
			C3.makeCompressed();				

			if(verbose) std::cout << "Done" << std::endl;

			return;
		}

		void effective_hamiltonian::HConstruct(){
			// Construct the Hamiltonian
			if(verbose){
                std::cout << "\tConstructing the hopping part... ";
			    fflush(stdout);
            }		
			Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor> Hfree = hopping_hamiltonian_construct();
			if(verbose) std::cout << "Done. Sparsity: " << (double)Hfree.nonZeros() / (double)Hfree.size() << std::endl;

			if(verbose){
                std::cout << "\tConstructing the confinement part... ";
			    fflush(stdout);
            }		
			Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor> Hconf = confinement_hamiltonian_construct();
			if(verbose) std::cout << "Done. Sparsity: " << (double)Hfree.nonZeros() / (double)Hfree.size() << std::endl;
			
			if(verbose){
                std::cout << "\tConstructing the interaction part... ";
			    fflush(stdout);
            }
			Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor> Hint = interaction_hamiltonian_construct();
			if(verbose) std::cout << "Done. Sparsity: " << (double)Hint.nonZeros() / (double)Hint.size() << std::endl;
			
			Heff.resize(Hfree.rows(),Hfree.cols());
			Heff = Hfree + Hconf + Hint;

			// Construct the Casimirs
			if(verbose) std::cout << "\tConstructing the Casimirs... " << std::endl;
			casimirs_construct();

			// Check that they commute with the Hamiltonian 
			if(verbose) std::cout << "\tChecking the commutators of the Hamiltonian with the Casimirs... " << std::endl;
			
			// 1) [H,C_2]
			Eigen::SparseMatrix<std::complex<double>> commutator1 = Heff * C2 - C2 * Heff;
			double residual1 = 0;
			for(int row=0; row<commutator1.rows(); row++){
				Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(commutator1, row);
				
				for(; it; ++it){
					residual1 += std::abs(it.value());
				}
		    }
			
			if(verbose) std::cout << "\t\t||[H,C_2]|| = " << residual1 << std::endl;

			// 2) [H,C_3]
			Eigen::SparseMatrix<std::complex<double>> commutator2 = Heff * C3 - C3 * Heff;
			double residual2 = 0;
			for(int row=0; row<commutator2.rows(); row++){
				Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(commutator2, row);
				
				for(; it; ++it){
					residual2 += std::abs(it.value());
				}
		    }

			if(verbose) std::cout << "\t\t||[H,C_3]|| = " << residual2 << std::endl;

			// 3) [C_2,C_3]
			Eigen::SparseMatrix<std::complex<double>> commutator3 = C2 * C3 - C3 * C2;
			double residual3 = 0;
			for(int row=0; row<commutator3.rows(); row++){
				Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(commutator3, row);
				
				for(; it; ++it){
					residual3 += std::abs(it.value());
				}
		    }

			if(verbose) std::cout << "\t\t||[C_2,C_3]|| = " << residual3 << std::endl;		

			// 4) [H,I_2]
			Eigen::SparseMatrix<std::complex<double>> commutator4 = Heff * I2 - I2 * Heff;
			double residual4 = 0;
			for(int row=0; row<commutator4.rows(); row++){
				Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(commutator4, row);
				
				for(; it; ++it){
					residual4 += std::abs(it.value());
				}
		    }

			if(verbose) std::cout << "\t\t||[H,I_2]|| = " << residual4 << std::endl;

			// 5) [C_2,I_2]
			Eigen::SparseMatrix<std::complex<double>> commutator5 = C2 * I2 - I2 * C2;
			double residual5 = 0;
			for(int row=0; row<commutator5.rows(); row++){
				Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(commutator5, row);
				
				for(; it; ++it){
					residual5 += std::abs(it.value());
				}
		    }

			if(verbose) std::cout << "\t\t||[C_2,I_2]|| = " << residual5 << std::endl;	

			// 5) [C_3,I_2]
			Eigen::SparseMatrix<std::complex<double>> commutator6 = C3 * I2 - I2 * C3;
			double residual6 = 0;
			for(int row=0; row<commutator6.rows(); row++){
				Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(commutator6, row);
				
				for(; it; ++it){
					residual6 += std::abs(it.value());
				}
		    }

			if(verbose) std::cout << "\t\t||[C_3,I_2]|| = " << residual6 << std::endl;	

			if( residual1>1.e-6 or residual2>1.e-6 or residual3>1.e-6 or residual4>1.e-6 or residual5>1.e-6 or residual6>1.e-6 ){
				std::cout << "\nThe operators do not commute!!! Check the code\n";
				exit(1);
			}

			return;			
		}

		void effective_hamiltonian::HConstructPetsc(Mat *A){
			HConstruct();
			
			// Construct a perturbed Hamiltonian
			Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor> Hpert = Heff + 1.e-1 * C2 + 1.e-2 * C3 + 1.e-3 * I2;

			// Converting it to a PETSc object
			if(verbose){
                std::cout << "\tConverting the " << Hpert.rows() << "x" << Hpert.cols() << " Hamiltonian to a PETSc object... ";
                fflush(stdout);
            }

			PetscInt size = Hpert.rows(); // Number of rows/columns
			
			PetscInt *nz; // (Expected) number of nonzeros per row
			nz = (PetscInt*)malloc(size * sizeof(PetscInt));		

			for(PetscInt row=0; row<size; row++){
				Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(Hpert, row);
				int nonZeroCount = 0;
				for(; it; ++it){
					++nonZeroCount;
				}
					
				nz[row] = nonZeroCount;
			}

			PetscCallVoid( MatCreateSeqAIJ(PETSC_COMM_SELF, size, size, 0, nz, A) );

			for(int k = 0; k < Hpert.outerSize(); ++k){
				for( Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor>::InnerIterator it(Hpert, k); it; ++it ){
					MatSetValue(*A, it.row(), it.col(), it.value(), INSERT_VALUES);
				}
			}

			MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
			MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);

			free(nz);

			if(verbose) std::cout << "Done\n";
			
			return;
		}

		std::pair<int,int> effective_hamiltonian::get_pq_from_C2C3(double c2, double c3){
			int N = NA+NB+NC;

			for(int p=0; p<N+1; p++){
				for(int q=0; q<std::ceil(0.5*(N-p))+1; q++){
					double tmp_c2 = (p*p + q*q + 3.*p + 3.*q + p*q) / 3.;
					double tmp_c3 = (p-q) * (3.+p+2.*q) * (3.+q+2.*p) / 18.;

					if( std::abs(tmp_c2 - c2)<1.e-4 and std::abs(tmp_c3 - c3)<1.e-4){
						return std::make_pair(p,q);
					}
				}
			}

			return std::make_pair(-1,-1);
		}
		
	// Public		
		effective_hamiltonian::effective_hamiltonian(){
			return;
		}
				
		effective_hamiltonian::effective_hamiltonian(int na, int nb, int nc, int length, double J, double U0, double U2, double U3, double gamma, bool pbc, bool print_info){
			verbose = print_info;

			// Set the system's parameters
			NA = na;
			NB = nb;
			NC = nc;
					
			number_of_sites = length;

			hopping_rate = J;
			cosine_potential_strength = U0;
			two_body_contact = U2;
			three_body_contact = U3;
			loss_rate = gamma;

			periodic_boundary_conditions = pbc;

			// Build the total basis: notice that n=0 corresponds to the largest particle number
			if(verbose){
				std::cout << "Initializing the basis... ";
				fflush(stdout);
			}
			tensored_basis::tensored_basis tmp(NA,NB,NC,number_of_sites, false);
			total_basis = tmp;
			
            if(verbose){
				std::cout << "Done" << std::endl;
			}

			// Initialize PETSc and SLEPc
			PetscInitialize(NULL, NULL, PETSC_NULLPTR, PETSC_NULLPTR);
			SlepcInitialize(NULL, NULL, PETSC_NULLPTR, PETSC_NULLPTR);

			return;
		}
	
		// Destructor
		effective_hamiltonian::~effective_hamiltonian(){
			PetscFinalize();
			SlepcFinalize();
		}        

		// Diagonalize the effective Hamiltonian
		void effective_hamiltonian::hamiltonian_diagonalization( uint numberOfEigenvalues ){			
			if(verbose) std::cout << "Constructing the " << total_basis.hilbert_space_dimension << "x" << total_basis.hilbert_space_dimension << " Hamiltonian... " << std::endl;

			if( total_basis.hilbert_space_dimension > 6000 ){
				Mat A;
				HConstructPetsc(&A);

				// Diagonalize it
				if(verbose){
					std::cout << "Diagonalizing it (Kyrlov-Schur)... ";
					fflush(stdout);
				}

				// Solver
				EPS eps;
				EPSCreate(PETSC_COMM_SELF, &eps);
				
				EPSSetType(eps, EPSKRYLOVSCHUR);
				EPSSetOperators(eps, A, PETSC_NULLPTR );
				EPSSetProblemType(eps, EPS_NHEP);
				EPSSetWhichEigenpairs(eps, EPS_LARGEST_IMAGINARY);
				
				uint nev = numberOfEigenvalues;
				uint ncv = 5 * numberOfEigenvalues;
				uint mpd = ncv;
				
				if( numberOfEigenvalues > total_basis.hilbert_space_dimension ){
					nev = total_basis.hilbert_space_dimension;
				}
				if( ncv > total_basis.hilbert_space_dimension ){
					ncv = nev + 1;
					mpd = ncv;
				}
				
				EPSSetDimensions(eps, nev, ncv, mpd);
				
				PetscReal tol = 1e-12;
				PetscInt max_it = 1000000;
				EPSSetTolerances(eps, tol, max_it);

				auto time_start = std::chrono::high_resolution_clock::now();
				
				EPSSolve(eps);
				
				auto time_end = std::chrono::high_resolution_clock::now();
						
				std::chrono::duration<double> interval = time_end - time_start;

				PetscInt iterations;
				EPSGetIterationNumber(eps, &iterations);			

				if(verbose) std::cout << "Done. Time taken  (solver converged after " << iterations << " iterations): " << interval.count() << " s\n";

				// Retrieve the results
				if(verbose){
					std::cout << "Retrieving the results... ";
					fflush(stdout);
				}

				PetscScalar eval;
				Vec evec;
				VecCreateSeq(PETSC_COMM_SELF, total_basis.hilbert_space_dimension, &evec);

				PetscScalar *evecArray;
				evecArray = (PetscScalar*)malloc(total_basis.hilbert_space_dimension * sizeof(PetscScalar));			

				PetscInt nconv;
				EPSGetConverged(eps,&nconv);

				evals.resize( nconv );
				evecs.resize( total_basis.hilbert_space_dimension, nconv );
				
				for(int e=0; e<nconv; e++){
					EPSGetEigenpair(eps, e, &eval, PETSC_NULLPTR, evec, PETSC_NULLPTR);
					
					evals[e] = eval;
					
					VecGetArray(evec, &evecArray);
					
					for(uint state=0; state<total_basis.hilbert_space_dimension; state++){
						evecs(state, e) = evecArray[state];
					}
					
					VecRestoreArray(evec, &evecArray);
				}

				// Destroy and finalize
				free(evecArray);
				MatDestroy(&A);
				VecDestroy(&evec);
				EPSDestroy(&eps);
				
				std::cout << "Done\n";

				if( (uint)nconv < nev ){
					std::cout << "\n\tNot all the targeted eigenvalues converged to the prescribed tolerance...";
					std::cout << "\tTargeted eigenvalues: " << nev << "\t" << "Converged: " << nconv << "\n\n";
				}
				else{
					std::cout << "\tTargeted eigenvalues: " << nev << "\t" << "Converged: " << nconv << "\n\n";
				}
		
				if(verbose){
					std::cout << "Getting out the energies and SU(3) representations quantum numbers... ";
					fflush(stdout);
				}


				Eigen::VectorXcd heff_evals( nconv );
				Eigen::VectorXd i2_evals( nconv );
				Eigen::VectorXd c2_evals( nconv );
				Eigen::VectorXd c3_evals( nconv );

				//Eigen::VectorXcd var_heff( nconv );
				//Eigen::VectorXd var_i2( nconv );
				//Eigen::VectorXd var_c2( nconv );
				//Eigen::VectorXd var_c3( nconv );

				std::vector<std::pair<int,int>> pq( nconv );

				for(int e=0; e<nconv; e++){
					Eigen::VectorXcd eigenvector = evecs.col(e);

					heff_evals[e] = eigenvector.adjoint() * ( Heff * eigenvector ); 

					std::complex<double> i2 = eigenvector.adjoint() * ( I2 * eigenvector );
					std::complex<double> c2 = eigenvector.adjoint() * ( C2 * eigenvector );
					std::complex<double> c3 = eigenvector.adjoint() * ( C3 * eigenvector );

					if( std::imag(i2) > 1.e-6 or std::imag(c2) > 1.e-6 or std::imag(c3) > 1.e-6 ){
						std::cout << "\nImaginary I2, C2 or C3 expectation values?!\n";
						std::cout << "\tI2 = " << i2 << "\tC2 = " << c2 << "\n\tC3 = " << c3 << "\n";
					}

					i2_evals[e] = std::real(i2);
					c2_evals[e] = std::real(c2);
					c3_evals[e] = std::real(c3);

					pq[e] = get_pq_from_C2C3(std::real(c2), std::real(c3));

					//std::complex<double> squared_h = eigenvector.adjoint() * ( Heff * Heff * eigenvector );
					//std::complex<double> squared_i2 = eigenvector.adjoint() * ( I2 * I2 * eigenvector );
					//std::complex<double> squared_c2 = eigenvector.adjoint() * ( C2 * C2 * eigenvector );
					//std::complex<double> squared_c3 = eigenvector.adjoint() * ( C3 * C3 * eigenvector );				

					//var_heff[e] = squared_h - heff_evals[e]*heff_evals[e];
					//var_i2[e] = std::real(squared_i2) - i2_evals[e]*i2_evals[e];
					//var_c2[e] = std::real(squared_c2) - c2_evals[e]*c2_evals[e];
					//var_c3[e] = std::real(squared_c3) - c3_evals[e]*c3_evals[e];
				}

				output.eigenvalues = heff_evals;
				output.su2subalgebra_casimir_eigenvalues = i2_evals;
				output.quadratic_casimir_eigenvalues = c2_evals;
				output.cubic_casimir_eigenvalues = c3_evals;

				//output.heff_variances = var_heff;
				//output.I2_variances = var_i2;
				//output.C2_variances = var_c2;
				//output.C3_variances = var_c3;

				output.pq = pq;
				output.eigenvectors = evecs;    

				if(verbose){
					std::cout << "Done\n\n ";
					fflush(stdout);
				} 
			}
			else{
				HConstruct();
				Eigen::MatrixXcd Hpert = (Heff + 1.e-1 * C2 + 1.e-2 * C3 + 1.e-3 * I2).toDense();

				// Diagonalize it
				if(verbose){
					std::cout << "Diagonalizing it (full diagonalization)... ";
					fflush(stdout);
				}

				auto time_start = std::chrono::high_resolution_clock::now();

				Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
				ces.compute(Hpert);
				
				auto time_end = std::chrono::high_resolution_clock::now();
						
				std::chrono::duration<double> interval = time_end - time_start;

				if(verbose) std::cout << "Done. Time taken: " << interval.count() << " s\n";

				// Retrieve the results
				if(verbose){
					std::cout << "Retrieving the results... ";
					fflush(stdout);
				}

				evals = ces.eigenvalues();
				evecs = ces.eigenvectors();
			
				std::cout << "Done\n";
		
				if(verbose){
					std::cout << "Getting out the energies and SU(3) representations quantum numbers... ";
					fflush(stdout);
				}


				Eigen::VectorXcd heff_evals( evals.size() );
				Eigen::VectorXd i2_evals( evals.size() );
				Eigen::VectorXd c2_evals( evals.size() );
				Eigen::VectorXd c3_evals( evals.size() );

				//Eigen::VectorXcd var_heff( evals.size() );
				//Eigen::VectorXd var_i2( evals.size() );
				//Eigen::VectorXd var_c2( evals.size() );
				//Eigen::VectorXd var_c3( evals.size() );

				std::vector<std::pair<int,int>> pq( evals.size() );

				for(int e=0; e<evals.size(); e++){
					Eigen::VectorXcd eigenvector = evecs.col(e);

					heff_evals[e] = eigenvector.adjoint() * ( Heff * eigenvector ); 

					std::complex<double> i2 = eigenvector.adjoint() * ( I2 * eigenvector );
					std::complex<double> c2 = eigenvector.adjoint() * ( C2 * eigenvector );
					std::complex<double> c3 = eigenvector.adjoint() * ( C3 * eigenvector );		

					if( std::imag(i2) > 1.e-6 or std::imag(c2) > 1.e-6 or std::imag(c3) > 1.e-6 ){
						std::cout << "\nImaginary I2, C2 or C3 expectation values?!\n";
						std::cout << "\tI2 = " << i2 << "\tC2 = " << c2 << "\n\tC3 = " << c3 << "\n";
					}

					i2_evals[e] = std::real(i2);
					c2_evals[e] = std::real(c2);
					c3_evals[e] = std::real(c3);

					pq[e] = get_pq_from_C2C3(std::real(c2), std::real(c3));


					//std::complex<double> squared_h = eigenvector.adjoint() * ( Heff * Heff * eigenvector );
					//std::complex<double> squared_i2 = eigenvector.adjoint() * ( I2 * I2 * eigenvector );
					//std::complex<double> squared_c2 = eigenvector.adjoint() * ( C2 * C2 * eigenvector );
					//std::complex<double> squared_c3 = eigenvector.adjoint() * ( C3 * C3 * eigenvector );				

					//var_heff[e] = squared_h - heff_evals[e]*heff_evals[e];
					//var_i2[e] = std::real(squared_i2) - i2_evals[e]*i2_evals[e];
					//var_c2[e] = std::real(squared_c2) - c2_evals[e]*c2_evals[e];
					//var_c3[e] = std::real(squared_c3) - c3_evals[e]*c3_evals[e];
				}

				output.eigenvalues = heff_evals;
				output.su2subalgebra_casimir_eigenvalues = i2_evals;
				output.quadratic_casimir_eigenvalues = c2_evals;
				output.cubic_casimir_eigenvalues = c3_evals;

				//output.heff_variances = var_heff;
				//output.I2_variances = var_i2;
				//output.C2_variances = var_c2;
				//output.C3_variances = var_c3;

				output.pq = pq;
				output.eigenvectors = evecs;

				if(verbose){
					std::cout << "Done\n\n ";
					fflush(stdout);
				} 
			}

			return;
		}

        // Print the eigenvalues
        void effective_hamiltonian::spectrum_print(){
            if( output.eigenvalues.size() == 0 ){
                std::cout << "Printing the spectrum... Did you forgot to run the solver?" << std::endl;

                return;
            }

            std::cout << "Spectrum of the non-hermitian effective Hamiltonian..." << std::endl;
            for(int ev=0; ev<output.eigenvalues.size(); ev++){
				int p = output.pq[ev].first;
				int q = output.pq[ev].second;

				//std::complex<double> E = output.eigenvalues[ev];
				//std::complex<double> E2 = output.heff_variances[ev];

				double i2 = output.su2subalgebra_casimir_eigenvalues[ev];
				double c2 = output.quadratic_casimir_eigenvalues[ev];
				double c3 = output.cubic_casimir_eigenvalues[ev];

				//double var_i2 = output.I2_variances[ev];
				//double var_c2 = output.C2_variances[ev];
				//double var_c3 = output.C3_variances[ev];				

				printf("\t E[%4d] = %+.6lf%+.6lfi\t\tI2 = %+.1lf\t(C2,C3)=(%+.2lf,%+.2lf)\t(p,q)=(%d,%d)\n", ev, std::real(output.eigenvalues[ev]), std::imag(output.eigenvalues[ev]), i2, c2, c3, p, q );
				//printf("\t⟨H⟩[%d] = %+.6lf%+.6lfi\t⟨H²⟩=%+.6lf%+.6lfi\n\t\tI2±σ(I2) = %+.1lf±%.1lf\tC2=%+.2lf±%+.2lf\tC3=%+.2lf±%+.2lf\t(p,q)=(%d,%d)\n\n", ev, std::real(E), std::imag(E), std::real(E2), std::imag(E2), i2, sqrt(var_i2), c2, sqrt(var_c2), c3, sqrt(var_c3), p, q );
            }
            std::cout << std::endl;

            std::cout << "Dark states..." << std::endl;
			int counter_dark = 0;
			int counter_singlet = 0;

			std::vector<Eigen::VectorXcd> dark_singlets;
			dark_singlets.reserve(output.eigenvalues.size());

			std::map<std::pair<int, int>, std::vector<int>> grouped_eigenvalues;

			for(int ev = 0; ev < output.eigenvalues.size(); ev++){
				int p = output.pq[ev].first;
				int q = output.pq[ev].second;

				if(std::abs(std::imag(output.eigenvalues[ev])) < 1.e-9){
					counter_dark++;

					if(p == 0 and q == 0){
						counter_singlet++;
						dark_singlets.push_back(output.eigenvectors.col(ev));
					}

					// Group the eigenvalue by (p, q)
					grouped_eigenvalues[std::make_pair(p, q)].push_back(ev);
				}
			}

			for(const auto& [key, indices] : grouped_eigenvalues){
				int p = key.first;
				int q = key.second;
				printf("Group (p=%d, q=%d): contains %lu dark states:\n", p, q, indices.size());

				for(int ev : indices){
					double i2 = output.su2subalgebra_casimir_eigenvalues[ev];
					double c2 = output.quadratic_casimir_eigenvalues[ev];
					double c3 = output.cubic_casimir_eigenvalues[ev];
					
					printf("\t E[%4d] = %+.6lf%+.6lfi\t\tI2 = %+.1lf\t(C2,C3)=(%+.2lf,%+.2lf)\t(p,q)=(%d,%d)\n", ev, std::real(output.eigenvalues[ev]), std::imag(output.eigenvalues[ev]), i2, c2, c3, p, q );
				}
				std::cout << std::endl;
			}

            std::cout << std::endl;

			std::cout << "Total number of dark states (among the found ones): " <<	counter_dark << std::endl;
			std::cout << "Among which there are " << counter_singlet << " singlets" << std::endl;

			/*
			std::cout << "Printing the dark (0,0) states..." << std::endl;
			for(uint ds=0; ds<dark_singlets.size(); ds++){
				std::cout << "State #: " << ds << "\n";

				for(uint F=0; F<dark_singlets[ds].size(); F++){
					if( std::abs(dark_singlets[ds][F]) > 1.e-6 ){
						printf("\t(%+.4lf%+.4lfi) ", std::real(dark_singlets[ds][F]), std::imag(dark_singlets[ds][F]));
						print_fock(F);
					}
				}
			}*/

            return;
        }

		// Retrieve results
		results effective_hamiltonian::retrieve_results(){
			return output;
		}
}
