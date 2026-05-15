
#ifndef EFFECTIVE_HAMILTONIAN_H
#define EFFECTIVE_HAMILTONIAN_H

namespace effective_hamiltonian{
	struct results{
		Eigen::VectorXcd eigenvalues;
		Eigen::MatrixXcd eigenvectors;

		Eigen::VectorXd su2subalgebra_casimir_eigenvalues;
		Eigen::VectorXd quadratic_casimir_eigenvalues;
		Eigen::VectorXd cubic_casimir_eigenvalues;

		//Eigen::VectorXcd heff_variances;
		//Eigen::VectorXd I2_variances;
		//Eigen::VectorXd C2_variances;
		//Eigen::VectorXd C3_variances;		

		std::vector<std::pair<int,int>>	pq;
	};
	
	class effective_hamiltonian{
		private:
			bool verbose;

			// Print a Fock state
			void print_fock(int F);

			// su(3) symmetric coefficients
			std::vector<std::pair<std::tuple<int, int, int>, double>> d_coeffs = {
				{{1, 1, 8}, 1./sqrt(3.)},
				{{2, 2, 8}, 1./sqrt(3.)},
				{{3, 3, 8}, 1./sqrt(3.)},
				{{8, 8, 8}, - 1./sqrt(3.)},
				{{4, 4, 8}, - 0.5/sqrt(3.)},
				{{5, 5, 8}, - 0.5/sqrt(3.)},
				{{6, 6, 8}, - 0.5/sqrt(3.)},
				{{7, 7, 8}, - 0.5/sqrt(3.)},
				{{3, 4, 4}, 0.5},
				{{3, 5, 5}, 0.5},
				{{3, 6, 6}, - 0.5},
				{{3, 7, 7}, - 0.5},
				{{2, 4, 7}, - 0.5},
				{{1, 4, 6}, 0.5},
				{{1, 5, 7}, 0.5},
				{{2, 5, 6}, 0.5},
			};

			double symmetric_coefficients(int i, int j, int k);			

			// System information
			int NA, NB, NC;
			int number_of_sites;
			double hopping_rate, cosine_potential_strength, two_body_contact, three_body_contact, loss_rate;
			
			bool periodic_boundary_conditions;
            						
			// Total tensor-producted basis, and the wavefunction
			tensored_basis::tensored_basis total_basis;

			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> hopping_hamiltonian_construct( );
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> confinement_hamiltonian_construct( );
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> interaction_hamiltonian_construct( );
			
			std::vector<std::tuple<int, int, int>> valid_shifts = {
				{-1, 1, 0},
				{-1, 0, 1},
				{0, -1, 1},
				{0, 0, 0},
				{0, 1, -1},
				{1, 0, -1},
				{1, -1, 0}
			};

			int flatten_particle_shifts(int da, int db, int dc);
			
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> Heff;
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> I2;
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> C2;
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> C3;
			
			void casimirs_construct( );
			bool isSelfAdjoint(const Eigen::SparseMatrix<std::complex<double>>& mat, double tolerance = 1e-10);
			std::vector< Eigen::SparseMatrix<std::complex<double>> > su3_algebra_construct( std::vector<tensored_basis::tensored_basis> &extended_basis, std::vector<std::vector<int>> &flatten, std::vector<std::tuple<int, int, int, int>> &unflatten );

			void HConstruct();
            void HConstructPetsc(Mat *A);
            
            Eigen::VectorXcd evals;
            Eigen::MatrixXcd evecs;
            results output;

			std::pair<int,int> get_pq_from_C2C3(double c2, double c3);

		public:		
			// Class declaration
			effective_hamiltonian();
			effective_hamiltonian(int na, int nb, int nc, int length, double J, double U0, double U2, double U3, double gamma, bool pbc = true, bool print_info = false);
            ~effective_hamiltonian();

			// Diagonalize the effective Hamiltonian
			void hamiltonian_diagonalization( uint numberOfEigenvalues = 10 );
			
            // Print the spectrum
            void spectrum_print();

			// Retrieve results
			results retrieve_results();
	};
}

#endif
