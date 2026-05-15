#include <vector>
#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <Eigen/Dense>
#include <numeric>
#include <algorithm>

#include "basis.h"

namespace basis {
	// Members of the basis::basis class
		// Private
		// Basis construction
		void basis::generateArrangements(std::vector<std::vector<int>>& arrangements, std::vector<int>& arrangement, int pos, int count){
			if (pos == sz) {
				// All boxes have been filled
				if (count == N) {
					// We have used all the objects
					arrangements.push_back(arrangement);
				}
				return;
			}

			for (int i = 0; i <= std::min(maxParticlesPerSite, N - count); i++) {
				// Try putting i objects in the current box
				arrangement[pos] = i;
				generateArrangements(arrangements, arrangement, pos + 1, count + i);
			}
		}

		// Prime generation
		std::vector<int> basis::SieveOfEratosthenes(int targetNumberOfPrimes){	
			int sieveBound = - targetNumberOfPrimes * boost::math::lambert_wm1( - exp(-2.) / targetNumberOfPrimes);
					
			std::vector<int> primes;
			primes.reserve(sieveBound+1);
			
			std::vector<bool> isPrime(sieveBound+1, true);
		   
			for(int i = 2; i*i <= sieveBound; i++){
				// If i is prime, all the multiples of i (except for i) should be marked as non-prime
				if(isPrime[i] == true){
					for (int j = i*2; j <= sieveBound; j += i){
						isPrime[j] = false;
					}
				}
			}
			
			// Collect all the primes in a vector, starting from i=2 (i=0, i=1 are not primes)
			for(int i = 2; i <= sieveBound; i++){
				if(isPrime[i]){
					primes.push_back(i);
				}
			}
			
			// Check whether the primes are enough
			if( primes.size() < sz ){
				std::cout << "Prime generation error\n";
				exit(1);
			}			
			
			return primes;
		}
		
		// Basis re-ordering
		void basis::sort(std::vector<std::vector<int>> &arrangements, Eigen::MatrixXi &b){
			// Instead of exchanging the rows of the matrix multiple times, I 
			// initialize a vector with increasing integer numbers. I order this vector
			// and I end up with a map from the initial unordered rows to the final ones
			// I then copy the rows from the old matrix to the correct place in a new matrix

			// Map vector
			std::vector<int> map(arrangements.size());
			std::iota(map.begin(), map.end(), 0);

			// Sort according to the magnitude of the tags. Lowest is smallest
			std::sort(map.begin(), map.end(),
					  [&](int a, int b) { return tags[a] < tags[b]; });

			// Copy in a reordered way
			std::vector<double> tmpTags(tags.size());
			
			for(uint state=0; state<arrangements.size(); state++){
				tmpTags[state] = tags[map[state]];
				
				for(int site=0; site<sz; site++){
					b(state, site) = arrangements[map[state]][site];					
				}
			}
			
			tags = tmpTags;
			
			return;
		}
		
		// Public
		// Tag function
		double basis::tag(const Eigen::VectorXi &configuration){
			double tag = 0.;
			
			for(int site=0; site<sites; site++){
				tag += prime_log[site] * configuration[site];
			}
			
			return tag;
		}

		double basis::tag(const std::vector<int> &configuration){
			double tag = 0.;
			
			for(uint site=0; site<configuration.size(); site++){
				tag += prime_log[site] * configuration[site];
			}
			
			return tag;
		}

		int basis::tagSearch(double tag){	
			uint size = tags.size();
			if(tag < tags[0] or tag > tags[size-1]){
				std::cout << "Tag error\n";
				exit(1);
			}
			
			// Finds the tag by bisection
			int A = 0, B = size-1;
			int C = (int)(0.5 * (A+B));

			double tagA = tags[A];
			double tagB = tags[B];
			double tagC = tags[C];
			
			if( tagA == tag ){
				return A;
			}
			else if( tagB == tag ){
				return B;
			}
			else if( tagC == tag){
				return C;
			}

			while( true ){
				if( (tagA-tag) * (tagC-tag) <= 0 ){ 				// Solution is in [A,C]
					B = C;														// Restric [A,B] -> [A,C]
					tagB = tagC;											// New B is the old C; tagA does not change
					
					C = (int)(0.5 * (A+B));								// Compute new average point
					tagC = tags[C];										// I recompute f[C]
				}
				else{															// Solution is in [C, B]
					A = C;														// Restrict [A, B] -> [C, B]
					tagA = tagC;											// New A is the old C; tagB does not change
					
					C = (int)(0.5 * (A+B));								// Compute new average point
					tagC = tags[C];										// I recompute f[C]
				}
				
				if( tagA == tag ){
					return A;
				}
				else if( tagB == tag ){
					return B;
				}
				else if( tagC == tag){
					return C;
				}
			}
			
		}

		// Retrieve infos
		int basis::getSize(){
			return basisStates.rows();
		}

		// Check
		void basis::check(){
			std::cout << "\nPrinting informations on the basis..." << std::endl;
			std::cout << "Number of particles: " << N << std::endl;
			std::cout << "Maximum number of particles per site: " << maxParticlesPerSite << std::endl;
			std::cout << "Number of lattice sites: " << sz << std::endl;
			
			std::cout << "\nNumber of basis states: " << configurations << std::endl;
			
			std::cout << "Now printing the logarithms of primes..." << std::endl;
			for(uint prime=0; prime<prime_log.size(); prime++){
				std::cout << prime << "\t" << prime_log[prime] << "\n";
			}
			
			std::cout << "\nNow printing the " << configurations << "x" << sites << " basis..." << std::endl;
			for(int state=0; state<configurations; state++){
				std::cout << state << "\t";
				for(int site=0; site<sites; site++){
					std::cout << basisStates(state, site) << " ";
				}
				std::cout << "\t" << tags[state] << "  " << tag(basisStates.row(state)) << std::endl;
			}
			
			std::cout << "\nCheck done" << std::endl;
						
			return;
		}
	
		// Basis construction
		basis::basis(){

			return;
		}

		basis::basis(int numberOfParticles, int numberOfLatticeSites, int maxNumberOfParticlesPerSite, bool V) : N(numberOfParticles), sz(numberOfLatticeSites), maxParticlesPerSite(maxNumberOfParticlesPerSite){
			// Verbose mode
			verbose = V;
						
			// Generate the basis
			double guessSize;
			if( numberOfParticles != 0 ){
				guessSize = 10. * boost::math::binomial_coefficient<double>(numberOfLatticeSites, numberOfParticles);
			}
			else{
				guessSize = 1;
			}
			
			std::vector<std::vector<int>> arrangements;
			arrangements.reserve( (int)guessSize );
			std::vector<int> arrangement(sz, 0);
			
			if( verbose ) std::cout << "\tGenerating arrangements of N = " << numberOfParticles << " particles in M = " << numberOfLatticeSites << " lattice sites, allowing for " << maxNumberOfParticlesPerSite << " particles per site... ";
			fflush(stdout);
			generateArrangements(arrangements, arrangement, 0, 0);
			if( verbose ) std::cout << "Done" << std::endl;
						
			// Generate the tags
			if( verbose ) std::cout << "\tGenerating the primes... ";
			fflush(stdout);			
			std::vector<int> primesLists = SieveOfEratosthenes( sz );
			
			prime_log.resize(primesLists.size());
			for(uint p=0; p<primesLists.size(); p++){
				prime_log[p] = std::log((double)primesLists[p]);
			}
			if( verbose ) std::cout << "Done" << std::endl;

			if( verbose ) std::cout << "\tGenerating the tags... ";
			fflush(stdout);					
			tags.resize(arrangements.size());
			for(uint state=0; state<arrangements.size(); state++){
				tags[state] = tag(arrangements[state]);
			}
			if( verbose ) std::cout << "Done" << std::endl;
			
			// Sort the basis so that we get increasing hash codes						
			if( verbose ) std::cout << "\tSorting... ";
			fflush(stdout);	

			Eigen::MatrixXi b(arrangements.size(), sz);	
								
			sort(arrangements, b);

			basisStates = b;
			configurations = b.rows();
			sites = b.cols();
			
			if( verbose ) std::cout << "Done" << std::endl;
			
			// And finally, check for self-consistency
			if( verbose ) std::cout << "\tCheck... ";
			fflush(stdout);				
			for(int state=0; state<configurations; state++){
				double hash_code = tag(basisStates.row(state));
				
				if( state != tagSearch(hash_code) ){
					std::cout << "Tag search does not correspond to the state. Check source code." << std::endl;
					exit(1);
				}
			}
			if( verbose ) std::cout << "Done\n" << std::endl;
			
			if( verbose ) check();
			
			return;
		}

		// Overloading the assignment operator
		basis& basis::operator=(const basis& otherBasis) {
			// Check for self-assignment
			if (this == &otherBasis) {
				return *this; // Return the current object
			}

			// Perform the assignment
			N = otherBasis.N;
			maxParticlesPerSite = otherBasis.maxParticlesPerSite;
			sz = otherBasis.sz;
			
			prime_log = otherBasis.prime_log;
			tags = otherBasis.tags;

			configurations = otherBasis.configurations;
			sites = otherBasis.sites;
			
			basisStates = otherBasis.basisStates;
			
			return *this; // Return the updated object
		}			

}
