#ifndef TENSORED_BASIS_H
#define TENSORED_BASIS_H

namespace tensored_basis{
	class tensored_basis{
		private:
			// Print infos by default?
			bool verbose;
					
			void doublons_triplons_precompute();
						
		public:
			// Class declaration
			tensored_basis();
			tensored_basis(int na, int nb, int nc, int length, bool V = false);

			// Parameters
			int NA, NB, NC;
			int number_of_sites;
			
			uint hilbert_space_A_dimension, hilbert_space_B_dimension, hilbert_space_C_dimension;
			uint hilbert_space_dimension;
			
			// Store doubly and triply occupied states
			// The first number in the pair is the flattened (FA, FB, FC); the second is the total number of doublons, the third the one of triplons
			std::vector<std::vector<uint>> site_resolved_triple_occupancies;
			std::vector<std::tuple<uint, uint, uint>> total_multiple_occupancies;
			
			// The basis
			basis::basis bA, bB, bC;
			
			// Rectify and de-rectify
			uint flatten(uint FA, uint FB, uint FC);
			std::tuple<uint,uint,uint> deflatten(uint state_index);
	};
}

#endif
