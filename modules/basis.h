#ifndef BASIS_H
#define BASIS_H

namespace basis {
	class basis{
		private:
			// Print infos by default?
			bool verbose;
			
			// Parameters
			int N;
			int maxParticlesPerSite;
			int sz;
			
			// Private functions for building the basis vectors (occupations of lattice sites)
			void generateArrangements(std::vector<std::vector<int>>& arrangements, std::vector<int>& arrangement, int pos, int count);
			
			// Prime functions for the tag function
			std::vector<double> prime_log;
			std::vector<int> SieveOfEratosthenes(int targetNumberOfPrimes);
			
			// Sorting
			void sort(std::vector<std::vector<int>> &arrangements, Eigen::MatrixXi &b);
			
		public:
			// Basis vectors and informations
			Eigen::MatrixXi basisStates;
			int configurations, sites;
			
			// Tags
			std::vector<double> tags;
			double tag(const Eigen::VectorXi &configuration);
			double tag(const std::vector<int> &configuration);
			int tagSearch(double tag);
			
			// Check
			void check();
			
			// Get infos
			int getSize();
			
			// Class declaration
			basis();
			basis(int numberOfParticles, int numberOfLatticeSites, int maxNumberOfParticlesPerSite, bool V = false);
			
			// Overloading the assignment operator
			basis& operator=(const basis& basisObject);
	};
}

#endif
