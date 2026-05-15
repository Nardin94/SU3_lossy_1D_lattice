#ifndef QUANTUM_TRAJECTORY_H
#define QUANTUM_TRAJECTORY_H

namespace quantum_trajectory{
	struct results{
		Eigen::VectorXd time;
		Eigen::VectorXd particle_number;
		Eigen::VectorXd particle_number_squared;
		Eigen::VectorXd particle_number_cubed;
		Eigen::VectorXd casimir_C2;
		Eigen::VectorXd casimir_C3;
		Eigen::VectorXd N_casimir_C2;
		Eigen::VectorXd dN_dt;
	};
	
	class quantum_trajectory{
		private:
			bool verbose;
			
			// System information
			int NA, NB, NC;
			int number_of_sites;
			double hopping_rate, two_body_contact, three_body_contact, loss_rate;
			std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi> initial_condition;
			
			// Stepper information
			int number_of_trajectories;
			double time_step;
			int Number_time_step;
			bool periodic_boundary_conditions;
            
			double U2_dt;
            std::complex<double> U3_dt;

			// Matrix representations of the Casimir operators
			bool isSelfAdjoint(const Eigen::SparseMatrix<std::complex<double>>& mat, double tolerance = 1e-10);

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

			std::vector< Eigen::SparseMatrix<std::complex<double>> > su3_algebra_construct( std::vector<tensored_basis::tensored_basis> &extended_basis, std::vector<std::vector<int>> &flatten, std::vector<std::tuple<int, int, int, int>> &unflatten,  int sector );
			
			void casimirs_construct( int sector, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &opC2, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> &opC3 );
			
			std::vector<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> C2_operators, C3_operators;	

			// Evolution of Ntot and Ntot^2 as function of time in average
			std::vector<double> Ntot_vector, Ntot2_vector, Ntot3_vector;
			std::vector<double> C2_vector, C3_vector, NC2_vector, dNdt_vector; 
			
			// Total tensor-producted basis, and the wavefunction
			std::vector<tensored_basis::tensored_basis> total_basis;
			Eigen::VectorXcd state_vector;
        	Eigen::VectorXcd tmp_rotated_A, tmp_rotated_B, tmp_rotated_C;

			int quantum_jumps_number;

			// Hopping Hamiltonian construction, diagonalization and green function
			std::vector<Eigen::MatrixXcd> hopping_green_function;
			Eigen::MatrixXcd hgf_A, hgf_B, hgf_C;
			
			uint HA_size, HB_size, HC_size;

			Eigen::SparseMatrix<double, Eigen::RowMajor> hopping_hamiltonian_construct( int particle_number, basis::basis &fock_space );
			void hopping_hamiltonian_diagonalization( int N );
			
			// Time evolution
			void second_order_trotterized_time_evolution(int sector);
			
			// Collapse
			void loss_probabilities_construct(std::vector<double> &loss_probabilities, int sector);
			void state_collapse(int sector, int site);
			
							
		public:		
			// Class declaration
			quantum_trajectory();
			quantum_trajectory(int na, int nb, int nc, int length, double J, double U2, double U3, double gamma, int NT, double dt, int N_timestep, std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi> IC,  bool pbc = true, bool print_info = false);

			// Wavefunction initialization
			void state_vector_initialization( Eigen::VectorXi &FA, Eigen::VectorXi &FB, Eigen::VectorXi &FC );	
			
			// Trajectory compute
			void compute_trajectories( );
			
			// Retrieve results
			results retrieve_results();
	};
}

#endif
