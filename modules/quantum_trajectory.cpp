#include <vector>
#include <tuple>
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

#include "basis.h"
#include "tensored_basis.h"
#include "quantum_trajectory.h"

using namespace std::complex_literals;

namespace quantum_trajectory {
	// Private
		// Hopping
		Eigen::SparseMatrix<double, Eigen::RowMajor> quantum_trajectory::hopping_hamiltonian_construct( int particle_number, basis::basis &fock_space ){		
			// Hopping hamiltonian linear dimension
			uint hilbert_space_dimension = fock_space.getSize();
			
			// Each particle in any of the Fock states can at most hop to two positions
			uint ansatzSize = hilbert_space_dimension * 2 * particle_number;
			std::vector<Eigen::Triplet<double>> list{};
			list.reserve( ansatzSize );
			
			// Set-up the Hamiltonian
			for(uint right_fock=0; right_fock<hilbert_space_dimension; right_fock++){
				// Make a copy of the basis vector
				Eigen::VectorXi configuration = fock_space.basisStates.row(right_fock);
					
				// Loop over all the sites
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
								uint left_fock = fock_space.tagSearch( fock_space.tag(configuration) );
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
									
								// Push it back
								list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate));
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
								uint left_fock = fock_space.tagSearch( fock_space.tag(configuration) );
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
									
								// Push it back
								int fermionic_sign = (particle_number-1)%2 == 0 ? 1 : -1;
								list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate * fermionic_sign));
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
								uint left_fock = fock_space.tagSearch( fock_space.tag(configuration) );
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
									
								// Push it back
								list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate));
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
								uint left_fock = fock_space.tagSearch( fock_space.tag(configuration) );
									
								// Move the particle back
								configuration[newPosition] -= 1;
								configuration[oldPosition] += 1;
									
								// Push it back
								int fermionic_sign = (particle_number-1)%2 == 0 ? 1 : -1;
								list.push_back(Eigen::Triplet<double>(left_fock, right_fock, -hopping_rate * fermionic_sign));
							}
						}
						
					}
				}
			}
			
			// The number of non-zero matrix elements sets the matrix dimension
			Eigen::SparseMatrix<double, Eigen::RowMajor> H(hilbert_space_dimension, hilbert_space_dimension);
			H.setFromTriplets(list.begin(), list.end());
				
			if( list.size() > ansatzSize ){
				std::cout << "\n\nAttention: a resize was needed\n\n";
			}

			return H;
		}
		
		void quantum_trajectory::hopping_hamiltonian_diagonalization(int particle_number){
			if( particle_number > 0 ){				
				// Construct the basis
				basis::basis tmp(particle_number, number_of_sites, 1);
				
				// Construct the Hamiltonian
				Eigen::SparseMatrix<double, Eigen::RowMajor> H = hopping_hamiltonian_construct( particle_number, tmp );

				// Diagonalize it
				Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double, Eigen::RowMajor>> eig;
				eig.compute(H);
								
				// Retrieve the spectrum
				Eigen::VectorXd spectrum = eig.eigenvalues();
				Eigen::MatrixXd eigenvectors = eig.eigenvectors();
				
				// Compute the propagator
				Eigen::MatrixXcd propagator(eigenvectors.rows(), eigenvectors.cols());				
								
				for(uint fock_left=0; fock_left<eigenvectors.rows(); fock_left++){
					for(uint fock_right=0; fock_right<eigenvectors.rows(); fock_right++){
						propagator(fock_left, fock_right) = 0;
												
						for(uint evec=0; evec<eigenvectors.cols(); evec++){
							propagator(fock_left, fock_right) += eigenvectors(fock_left, evec) * std::exp(-1i * spectrum[evec] * time_step) * std::conj( eigenvectors(fock_right, evec) );
						}
					}
				}
				
				hopping_green_function[particle_number] = propagator;
			}
			else{
				Eigen::MatrixXcd propagator(1, 1);
				propagator(0,0) = 1;
				
				hopping_green_function[0] = propagator;
			}
			
			return;
		}
		
		// Non-Hermitian time evolution
		void quantum_trajectory::second_order_trotterized_time_evolution(int sector){			
			// (!) Sector n=0 corresponds to the largest number of particles					
			// 1) Apply exp(-i Hu t / 2)
			for(auto& v : total_basis[sector].total_multiple_occupancies){
				state_vector[ std::get<0>(v) ] *= std::exp(-0.5i * ( U2_dt * std::get<1>(v) + U3_dt * (double)std::get<2>(v) ));
			}
			
			// 2) Apply exp(-i Hh(C) t) exp(-i Hh(B) t) exp(-i Hh(A) t)
			Eigen::Stride<1, Eigen::Dynamic> stride_B(1, HA_size);
			Eigen::Stride<1, Eigen::Dynamic> stride_C(1, HA_size * HB_size);

			for(uint FC=0; FC<HC_size; FC++){
				for(uint FB=0; FB<HB_size; FB++){

					uint base_offset = total_basis[sector].flatten(0, FB, FC);
					Eigen::Map<Eigen::VectorXcd> vec_A(&state_vector[base_offset], HA_size);
					
					tmp_rotated_A = hgf_A * vec_A;

					vec_A = tmp_rotated_A;
				}
			}

			for(uint FC=0; FC<HC_size; FC++){
				for(uint FA=0; FA<HA_size; FA++){

					size_t base_offset = total_basis[sector].flatten(FA, 0, FC);
					Eigen::Map<Eigen::VectorXcd, 0, Eigen::Stride<1, Eigen::Dynamic>> vec_B(&state_vector[base_offset], HB_size, stride_B);

					tmp_rotated_B = hgf_B * vec_B;
						
					vec_B = tmp_rotated_B;
				}
			}

			for(uint FB=0; FB<HB_size; FB++){
				for(uint FA=0; FA<HA_size; FA++){

					size_t base_offset = total_basis[sector].flatten(FA, FB, 0);
					Eigen::Map<Eigen::VectorXcd, 0, Eigen::Stride<1, Eigen::Dynamic>> vec_C(&state_vector[base_offset], HC_size, stride_C);

					tmp_rotated_C = hgf_C * vec_C;

					vec_C = tmp_rotated_C;
				}
			}

			// 3) Apply exp(-i Hu t / 2)
			for(auto& v : total_basis[sector].total_multiple_occupancies){	
				state_vector[ std::get<0>(v) ] *= std::exp(-0.5i * ( U2_dt * std::get<1>(v) + U3_dt * (double)std::get<2>(v) ));
			}		

			return;
		}
		
		// Construction of the Casimirs
		bool quantum_trajectory::isSelfAdjoint(const Eigen::SparseMatrix<std::complex<double>>& mat, double tolerance) {
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

		int quantum_trajectory::flatten_particle_shifts(int da, int db, int dc){
			std::tuple<int, int, int> target = std::make_tuple(da, db, dc);

			auto it = std::find(valid_shifts.begin(), valid_shifts.end(), target);

			if(it != valid_shifts.end()){
				return std::distance(valid_shifts.begin(), it);
			}
		
			return -1; // Invalid tuple
		}

		double quantum_trajectory::symmetric_coefficients(int i, int j, int k){
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

		std::vector< Eigen::SparseMatrix<std::complex<double>> > quantum_trajectory::su3_algebra_construct( std::vector<tensored_basis::tensored_basis> &extended_basis, std::vector<std::vector<int>> &flatten, std::vector<std::tuple<int, int, int, int>> &unflatten,  int sector ){
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
							int fermions = ((NA-sector)+da)+1;
							for(int s=0; s<site; s++){
								fermions += configurationA[s] + configurationB[s];
							}
							double fermionic_sign = fermions%2 == 0 ? 1. : -1.;

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
							int fermions = (NA-sector) + da + (NB-sector) + db + 1;
							for(int s=0; s<site; s++){
								fermions += configurationA[s] + configurationC[s];
							}
							double fermionic_sign = fermions%2 == 0 ? 1 : -1;
							
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
							int fermions = (NB-sector)+db+1;
							for(int s=0; s<site; s++){
								fermions += configurationB[s] + configurationC[s];
							}
							double fermionic_sign = fermions%2 == 0 ? 1 : -1;

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

		void quantum_trajectory::casimirs_construct( int sector, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &opC2, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &opC3 ){
			// 1. Construct an enlarged basis
			// 		We are interested into the following sectors: (NA,NB,NC), (NA+1,NB-1,NC), (NA+1,NB,NC-1), (NA-1,NB+1,NC), (NA-1,NB,NC+1), (NA,NB+1,NC-1), (NA,NB-1,NC+1)
			// 		We can define shift variables da, db, dc which take values in (-1,0,+1); the sum of these 3 variables is always zero
			//		We want to fold such a triple together in such a way that a contigous index is returned
			// and
			// 2. Build flattening and deflattening objects
			std::vector<tensored_basis::tensored_basis> extended_basis(7);

			std::vector<std::tuple<int, int, int, int>> unflatten = {};
			std::vector<std::vector<int>> flatten(7);

			int counter = 0;
			for(int da=-1; da<2; da++){
				for(int db=-1; db<2; db++){
					for(int dc=-1; dc<2; dc++){

						int index = flatten_particle_shifts(da,db,dc);

						if( index != -1 and 
							(NA-sector)+da>=0 and (NA-sector)+da<=number_of_sites and 
							(NB-sector)+db>=0 and (NB-sector)+da<=number_of_sites and 
							(NC-sector)+dc>=0 and (NC-sector)+da<=number_of_sites){
							tensored_basis::tensored_basis tmp((NA-sector)+da, (NB-sector)+db, (NC-sector)+dc, number_of_sites, false);

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

			int center_index = flatten_particle_shifts(0,0,0);
			int block_corner = flatten[center_index][0];

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

			// 3. Build representations of the Lambda matrices in this space
			std::vector< Eigen::SparseMatrix<std::complex<double>> > Lambda = su3_algebra_construct( extended_basis, flatten, unflatten, sector );

			// 4. Set up the quadratic and cubic Casimirs of SU(3)
			Eigen::SparseMatrix<std::complex<double>> extended_C2(unflatten.size(),unflatten.size());

			for(uint i=0; i<Lambda.size(); i++){
				Eigen::SparseMatrix<std::complex<double>> tmp = Lambda[i]*Lambda[i];

				extended_C2 += tmp;
			}

			Eigen::SparseMatrix<std::complex<double>> extended_C3(unflatten.size(),unflatten.size());

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

			// 5. Restrict them to the correct particle number sector				
			opC2 = extended_C2.block(block_corner, block_corner, extended_basis[center_index].hilbert_space_dimension, extended_basis[center_index].hilbert_space_dimension);
			opC2.makeCompressed();

			opC3 = extended_C3.block(block_corner, block_corner, extended_basis[center_index].hilbert_space_dimension, extended_basis[center_index].hilbert_space_dimension);
			opC3.makeCompressed();

			return;
		}

		// Collapse
		void quantum_trajectory::loss_probabilities_construct(std::vector<double> &loss_probabilities, int sector){
			
			for(int site=0; site<number_of_sites; site++){
				loss_probabilities[site] = 0;
				for(uint j :  total_basis[sector].site_resolved_triple_occupancies[site]){
					loss_probabilities[site] += std::norm(state_vector[j]);

				}
				loss_probabilities[site] *= loss_rate*time_step;
			}
			
			return;
		}
		
		void quantum_trajectory::state_collapse(int sector, int site){
			// The state is ordered as creations(A) creations(B) creations(C) |vac>
			// As a consequence, when collapsing with L_i = a_{C,i} a_{B,i} a_{A,i}
			// 1) We anticommute a_{A,i}: this will give a contribution sum_{j=0}^{i-1} N_{A,j}
			// 2) We anticommute a_{B,i}: this will give a contribution (NA-1) + sum_{j=0}^{i-1} N_{B,j}
			// 3) We anticommute a_{C,i}: this will give a contribution (NA-1) + (NB-1) + sum_{j=0}^{i-1} N_{C,j}
			// Overall: 2(NA-1)+(NB-1)+sum_{sigma=A,B,C} sum_{j=0}^{i-1} N_{sigma,j}
			// The first piece is always even, and can thus be neglected

			// Resize temporary variables used during the Trotterized time evolution
			tmp_rotated_A.resize(total_basis[sector+1].hilbert_space_A_dimension);
			tmp_rotated_B.resize(total_basis[sector+1].hilbert_space_B_dimension);
			tmp_rotated_C.resize(total_basis[sector+1].hilbert_space_C_dimension);

			hgf_A = hopping_green_function[total_basis[sector+1].NA];
			hgf_B = hopping_green_function[total_basis[sector+1].NB];
			hgf_C = hopping_green_function[total_basis[sector+1].NC];

			HA_size = total_basis[sector+1].hilbert_space_A_dimension;
			HB_size = total_basis[sector+1].hilbert_space_B_dimension;
			HC_size = total_basis[sector+1].hilbert_space_C_dimension;

			// Collapse: loop over all the Fock states ABC having a triple occupancy at the given collapse site
			std::vector<std::complex<double>> new_state_vector( total_basis[sector+1].hilbert_space_dimension, 0 );
			for(uint j :  total_basis[sector].site_resolved_triple_occupancies[site]){
				std::tuple<uint,uint,uint> FAFBFC = total_basis[sector].deflatten( j );
				
				// Sublattice A
				// Extracting the Fock state
				Eigen::VectorXi vA = total_basis[sector].bA.basisStates.row( std::get<0>(FAFBFC) );
				// Annihilating the fermion
				vA[site]--;
				// Searching its index into the NA-1 particles sector
				uint FAp = total_basis[sector+1].bA.tagSearch( total_basis[sector+1].bA.tag( vA ) );

				// Sublattice B
				// Extracting the Fock state				
				Eigen::VectorXi vB = total_basis[sector].bB.basisStates.row( std::get<1>(FAFBFC) );
				// Annihilating the fermion
				vB[site]--;
				// Searching its index into the NA-1 particles sector				
				uint FBp = total_basis[sector+1].bB.tagSearch( total_basis[sector+1].bB.tag( vB ) );

				// Sublattice C
				// Extracting the Fock state				
				Eigen::VectorXi vC = total_basis[sector].bC.basisStates.row( std::get<2>(FAFBFC) );
				// Annihilating the fermion
				vC[site]--;
				// Searching its index into the NA-1 particles sector				
				uint FCp = total_basis[sector+1].bC.tagSearch( total_basis[sector+1].bC.tag( vC ) );
				
				
				// Flatten the indices we found
				uint jp = total_basis[sector+1].flatten(FAp, FBp, FCp);
				
				// Finally, get the phase out
				int NABC_site = total_basis[sector+1].NB;
				for(int l=0; l<site; l++){
					NABC_site += vA[l] + vB[l] + vC[l];
				}

				// Copy the coefficient
				if( NABC_site%2==0 ){
					new_state_vector[jp] += std::sqrt(loss_rate)*state_vector[j];
				}
				else{
					new_state_vector[jp] -= std::sqrt(loss_rate)*state_vector[j];
				}
			}
			
			// Update the state vector
			double norm = 0;
			for(uint index=0; index<new_state_vector.size(); index++){
				norm += std::norm(new_state_vector[index]);
			}
			norm = 1. / std::sqrt(norm);

			state_vector.resize( new_state_vector.size() );
			for(uint index=0; index<new_state_vector.size(); index++){	
				state_vector[index] = new_state_vector[index] * norm;
			}
			

			return;
		}
			
	// Public
		quantum_trajectory::quantum_trajectory(){
			return;
		}
				
		quantum_trajectory::quantum_trajectory(int na, int nb, int nc, int length, double J, double U2, double U3, double gamma, int NT, double dt, int N_timestep, std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi> IC, bool pbc, bool print_info){
			verbose = print_info;

			// Set the system's parameters
			NA = na;
			NB = nb;
			NC = nc;
					
			number_of_sites = length;

			hopping_rate = J;
			two_body_contact = U2;
			three_body_contact = U3;
			loss_rate = gamma;

			initial_condition = IC;

			// Set the stepper parameters
			number_of_trajectories = NT;
			time_step = dt;
			Number_time_step = N_timestep;
			
			periodic_boundary_conditions = pbc;

			// Initialize the output vectors
			Ntot_vector.resize( Number_time_step + 1);
			Ntot2_vector.resize( Number_time_step + 1);
			Ntot3_vector.resize( Number_time_step + 1);
			
			C2_vector.resize( Number_time_step + 1);
			C3_vector.resize( Number_time_step + 1);
			NC2_vector.resize( Number_time_step + 1); 

			dNdt_vector.resize( Number_time_step + 1); 

			// Build all the Casimir operators needed
			if( verbose ){
				std::cout << "Setting up the Casmirs...";
				fflush(stdout);
			}
			C2_operators.resize( std::min({NA,NB,NC})+1 );
			C3_operators.resize( std::min({NA,NB,NC})+1 );
						
			for(int n=0; n<=std::min( {NA,NB,NC} ); n++){
				casimirs_construct( n, C2_operators[n], C3_operators[n] );
			}

			if( verbose ){
				std::cout << " Done" << std::endl;
				//for(int n=0; n<=std::min( {NA,NB,NC} ); n++){
				//	printf("\t%d\tC2(%dx%d)\tC3(%dx%d)\n", n, C2_operators[n].rows(), C2_operators[n].cols(), C3_operators[n].rows(), C3_operators[n].cols());
				//}
			}

			// Set some constants which do not need to be computed every time
			U2_dt = two_body_contact * time_step;
			U3_dt = (three_body_contact-0.5i*loss_rate) * time_step;

			// Build the total basis: notice that n=0 corresponds to the largest particle number
			if(verbose){
				std::cout << "Initializing the basis... ";
				fflush(stdout);
			}
			total_basis.resize( std::min( {NA,NB,NC} ) + 1 );
			for(int n=0; n<=std::min( {NA,NB,NC} ); n++){
				tensored_basis::tensored_basis tmp(NA-n,NB-n,NC-n,number_of_sites, false);
				total_basis[n] = tmp;
			}
			if(verbose){
				std::cout << "Done" << std::endl;				
			}

			// Running some checks on the IC
			// 1. Dimensions should match
			Eigen::VectorXi FA = std::get<0>(initial_condition);
			Eigen::VectorXi FB = std::get<1>(initial_condition);
			Eigen::VectorXi FC = std::get<2>(initial_condition);

			if( FA.size() != number_of_sites or FB.size() != number_of_sites or FC.size() != number_of_sites ){
				std::cout << "Size mismatch in the initial condition\n";
				exit(1);
			}
			
			// 2. Number of particles should be correct			
			int tmp_na=0, tmp_nb=0, tmp_nc=0;
			for(int site=0; site<number_of_sites; site++){
				tmp_na += FA[site];
				tmp_nb += FB[site];
				tmp_nc += FC[site];
			}
			if( tmp_na != NA or tmp_nb != NB or tmp_nc != NC ){
				std::cout << "Initial condition error!\n";
				exit(1);
			}

			// Build and diagonalize the hopping Hamiltonian: notice that n=0 corresponds to the zero particle sector
			if(verbose){
				std::cout << "Diagonalizing the hopping Hamiltonians... ";
				fflush(stdout);
			}			
			hopping_green_function.resize( std::max({NA,NB,NC})+1 );
			
			int tmp_NA = NA, tmp_NB = NB, tmp_NC = NC;

			std::unordered_set<int> seenElements;
			seenElements.reserve(std::max({NA,NB,NC}));

			while( tmp_NA>=0 and tmp_NB>=0 and tmp_NC>=0 ){
				if(seenElements.find(tmp_NA) == seenElements.end()){
					hopping_hamiltonian_diagonalization(tmp_NA);
					seenElements.insert( tmp_NA-- );
				}
				if(seenElements.find(tmp_NB) == seenElements.end()){
					hopping_hamiltonian_diagonalization(tmp_NB);
					seenElements.insert( tmp_NB-- );
				}
				if(seenElements.find(tmp_NC) == seenElements.end()){
					hopping_hamiltonian_diagonalization(tmp_NC);
					seenElements.insert( tmp_NC-- );
				}
			}

			if(verbose){
				std::cout << "Done. Sectors: ";
				for(auto s : seenElements){
					std::cout << s << " ";
				}				
				std::cout << std::endl;
			}			
					
			return;
		}
		
		// Initialize the state vector
		void quantum_trajectory::state_vector_initialization( Eigen::VectorXi &FA, Eigen::VectorXi &FB, Eigen::VectorXi &FC ){		
			// Begin by reserving the appropriate dimensions
			state_vector.resize( total_basis[0].hilbert_space_dimension );
			
			for(uint j=0; j<total_basis[0].hilbert_space_dimension; j++){
				state_vector[j] = 0;
			}
	
			// Initialize the state vector
			uint fa = total_basis[0].bA.tagSearch( total_basis[0].bA.tag(FA) );
			uint fb = total_basis[0].bB.tagSearch( total_basis[0].bB.tag(FB) );
			uint fc = total_basis[0].bC.tagSearch( total_basis[0].bC.tag(FC) );
			
			uint j = total_basis[0].flatten(fa, fb, fc);
			
			state_vector[j] = 1.;

			// Reserve the appropriate sizes to temporary variables
           		tmp_rotated_A.resize(total_basis[0].hilbert_space_A_dimension);
            		tmp_rotated_B.resize(total_basis[0].hilbert_space_B_dimension);
            		tmp_rotated_C.resize(total_basis[0].hilbert_space_C_dimension); 

			hgf_A = hopping_green_function[total_basis[0].NA];
			hgf_B = hopping_green_function[total_basis[0].NB];
			hgf_C = hopping_green_function[total_basis[0].NC];

			HA_size = total_basis[0].hilbert_space_A_dimension;
			HB_size = total_basis[0].hilbert_space_B_dimension;
			HC_size = total_basis[0].hilbert_space_C_dimension;

			return;
		}
	
		//Trajectory compute
		void quantum_trajectory::compute_trajectories( ){
			std::random_device rd;
    			std::mt19937 gen(rd());
			std::uniform_real_distribution<> dis(0.0, 1.0);

			if(verbose){
				std::cout << "Initial Hilbert-Space dimensions (" << total_basis[0].hilbert_space_A_dimension << "x" << total_basis[0].hilbert_space_B_dimension << "x" << total_basis[0].hilbert_space_C_dimension << ")" << std::endl;
				std::cout << "Computing the trajectories..." << std::endl;
			}

			for(int traj=0; traj<number_of_trajectories; traj++){
				auto start = std::chrono::high_resolution_clock::now();

				//Construct the initial state vector
				state_vector_initialization(std::get<0>(initial_condition), std::get<1>(initial_condition), std::get<2>(initial_condition));

				// Number of quantum jumps
				quantum_jumps_number = 0;

				// The loss probabilities
				std::vector<double> loss_probabilities_0(number_of_sites);
				std::vector<double> loss_probabilities_1(number_of_sites);

				// Initialize the observables
				double tmp_N = (double)(NA+NB+NC);		

				std::complex<double> complexified_c2 = state_vector.adjoint() * (C2_operators[quantum_jumps_number] * state_vector);
				std::complex<double> complexified_c3 = state_vector.adjoint() * (C3_operators[quantum_jumps_number] * state_vector);
				double tmp_c2 = std::real( complexified_c2 );
				double tmp_c3 = std::real( complexified_c3 );

				double tmp_NA_NB_NC_expectation = 0;
				for(auto& v : total_basis[quantum_jumps_number].total_multiple_occupancies){
					if( std::get<2>(v) != 0 ){
						tmp_NA_NB_NC_expectation += std::norm( state_vector[ std::get<0>(v) ] ) * std::get<2>(v);
					}
				}

				double tmp_dndt = - 3. * loss_rate * tmp_NA_NB_NC_expectation;

				Ntot_vector[0] += tmp_N / number_of_trajectories;
				Ntot2_vector[0] += std::pow(tmp_N, 2) / number_of_trajectories;
				Ntot3_vector[0] += std::pow(tmp_N, 3) / number_of_trajectories;

				C2_vector[0] += tmp_c2 / number_of_trajectories;
				C3_vector[0] += tmp_c3 / number_of_trajectories;
				NC2_vector[0] += tmp_c2 * tmp_N / number_of_trajectories;
	
				dNdt_vector[0] += tmp_dndt / number_of_trajectories;

				// Run the trajectory
				double min_dp = 1, max_dp = 0;
				for(int time_index=1; time_index<Number_time_step+1;time_index++){						
					loss_probabilities_construct(loss_probabilities_0, quantum_jumps_number);

					second_order_trotterized_time_evolution(quantum_jumps_number);
					loss_probabilities_construct(loss_probabilities_1, quantum_jumps_number);
				
					double loss_probability = 0;
					for(uint site=0; site<number_of_sites;site++){
						loss_probability += 0.5 * (loss_probabilities_0[site] + loss_probabilities_1[site]);
					}
					
					if(dis(gen) > loss_probability){
						//Normalize the state
						double norm=0;
						
						for(uint j=0;j<state_vector.size(); j++){
							norm+=std::norm(state_vector[j]);
						}

						norm = 1./std::sqrt(norm);

						for(uint j=0; j<state_vector.size(); j++){
							state_vector[j] = state_vector[j] * norm;
						}
					}
					else{
						double cumulative_loss_probability = 0;

						double random_number = dis(gen);

						for(int site=0; site<number_of_sites; site++){
							cumulative_loss_probability += 0.5 * (loss_probabilities_0[site] + loss_probabilities_1[site]) / loss_probability;

							if(random_number < cumulative_loss_probability){
								state_collapse(quantum_jumps_number++, site);
								break;
							}
						}
					}
					
					// Update observables
					// Total number of particles
					tmp_N = double(NA+NB+NC-3*quantum_jumps_number);

					// Casimirs
					complexified_c2 = state_vector.adjoint() * (C2_operators[quantum_jumps_number] * state_vector);
					complexified_c3 = state_vector.adjoint() * (C3_operators[quantum_jumps_number] * state_vector);
					tmp_c2 = std::real( complexified_c2 );
					tmp_c3 = std::real( complexified_c3 );

					// Average number of multiple occupancies
					tmp_NA_NB_NC_expectation = 0;

					for(auto& v : total_basis[quantum_jumps_number].total_multiple_occupancies){
						if( std::get<2>(v) != 0 ){
							tmp_NA_NB_NC_expectation += std::norm( state_vector[ std::get<0>(v) ] ) * std::get<2>(v);
						}
					}

					tmp_dndt = - 3. * loss_rate * tmp_NA_NB_NC_expectation;

					// Update all the observables
					Ntot_vector[time_index] += tmp_N / number_of_trajectories;
					Ntot2_vector[time_index] += std::pow(tmp_N,2) / number_of_trajectories;
					Ntot3_vector[time_index] += std::pow(tmp_N,3) / number_of_trajectories;

					C2_vector[time_index] += tmp_c2 / number_of_trajectories;
					C3_vector[time_index] += tmp_c3 / number_of_trajectories;
					NC2_vector[time_index] += tmp_c2 * tmp_N / number_of_trajectories;

					dNdt_vector[time_index] += tmp_dndt / number_of_trajectories;
					
					// Update the averaged dp
					min_dp = std::min(loss_probability, min_dp);
					max_dp = std::max(loss_probability, max_dp);
				}
				
				auto end = std::chrono::high_resolution_clock::now(); 
				std::chrono::duration<double> diff = end - start;
				//auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

				if(verbose){
					printf("\tTrajectory: %d\tMax and min loss probability: (%.4lf,%.4lf)\tNumber of jumps: %d\tTrajectory time: %.3lfs", traj, min_dp, max_dp, quantum_jumps_number, diff.count());
					std::cout << std::endl;
				}
			}

			return;
		}

		// Retrieve results
		results quantum_trajectory::retrieve_results(){
			results averages;
			averages.time.resize( Ntot_vector.size() );

			averages.particle_number.resize( Ntot_vector.size() );
			averages.particle_number_squared.resize( Ntot2_vector.size() );
			averages.particle_number_cubed.resize( Ntot3_vector.size() );

			averages.casimir_C2.resize( Ntot_vector.size() );
			averages.casimir_C3.resize( Ntot_vector.size() );
			averages.N_casimir_C2.resize( Ntot_vector.size() );
			averages.dN_dt.resize( Ntot_vector.size() );

			for(uint time=0; time<Ntot_vector.size(); time++){
				averages.time[time] = time * time_step;

				averages.particle_number[time] = Ntot_vector[time];
				averages.particle_number_squared[time] = Ntot2_vector[time];
				averages.particle_number_cubed[time] = Ntot3_vector[time];

				averages.casimir_C2[time] = C2_vector[time];
				averages.casimir_C3[time] = C3_vector[time];
				averages.N_casimir_C2[time] = NC2_vector[time];
				averages.dN_dt[time] = dNdt_vector[time];
			}
			
			return averages;
		}
}


/*

		void quantum_trajectory::timing_test(){
			state_vector_initialization(std::get<0>(initial_condition), std::get<1>(initial_condition), std::get<2>(initial_condition));
			quantum_jumps_number=0;			
			
			for(int reps=0; reps<10; reps++){
				auto start = std::chrono::high_resolution_clock::now();
				
				for(int time_index=1; time_index<Number_time_step+1;time_index++){
					std::cout << "t = " << time_index << "\t\r";
					second_order_trotterized_time_evolution(quantum_jumps_number);
				}
				std::cout << "\n";
				
				auto end = std::chrono::high_resolution_clock::now(); 
				std::chrono::duration<double> diff = end - start;
				
				printf("\tTime: %.3lfs\n\n\n", diff.count());
				fflush(stdout);
			}
			
			return;
		} 
 
*/
