#include <iostream>
#include <vector>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/CXX11/Tensor>

#include <slepceps.h>
#include <petsc.h>
#include <petscmat.h>

//#include <mpi.h>
#include <omp.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "./modules/basis.h"
#include "./modules/tensored_basis.h"
#include "./modules/quantum_trajectory.h"
#include "./modules/effective_hamiltonian.h"

using namespace std::complex_literals;

int L = 3;
int domain_wall = 1;

int NA = 3, NB = 2, NC = 2;

bool pbc = true;

double J = 1;
double U0 = 0.5;
double U2 = 0.;
double U3 = 0;
double g = 0.1;

int number_of_trajectories = 250;
int total_trajectories_number;

double time_step = 0.1;
double final_time = 500.;

int timesteps_number = std::ceil( final_time / time_step );

// Output

void outputInizialization(){
	char fileName[256];

	struct stat st = {0};	

	sprintf(fileName, "../output");
	if(stat(fileName, &st) == -1){
		mkdir(fileName, 0700);
	}

	sprintf(fileName, "../output/J=%g_U2=%g_U3=%g_g=%g", J, U2, U3, g);
	if(stat(fileName, &st) == -1){
		mkdir(fileName, 0700);
	}

	sprintf(fileName, "../output/J=%g_U2=%g_U3=%g_g=%g/L=%d_NA=%d_NB=%d_NC=%d", J, U2, U3, g, L, NA, NB, NC);
	if(stat(fileName, &st) == -1){
		mkdir(fileName, 0700);
	}
	
						
	return;
}

void save_data(Eigen::VectorXd &n, Eigen::VectorXd &n2, Eigen::VectorXd &n3, Eigen::VectorXd &C2, Eigen::VectorXd &C3, Eigen::VectorXd &NC2, Eigen::VectorXd &dNdt){
	FILE *fp;
	char fileName[256];	

	sprintf(fileName, "../output/J=%g_U2=%g_U3=%g_g=%g/L=%d_NA=%d_NB=%d_NC=%d/dt=%g_NT=%d_", J, U2, U3, g, L, NA, NB, NC, time_step, total_trajectories_number);

	std::string name(fileName);
	if( pbc == true ){
		std::string boundary_conditions = "pbc";
		name += boundary_conditions;
	}
	else{
		std::string boundary_conditions = "obc";
		name += boundary_conditions;		
	}

	fp = fopen(name.c_str(), "w");
	
	for(uint t=0; t<n.size(); t++){
		fprintf(fp, "%lf\t%.8lf\t%.8lf\t%.8lf\t%.8lf\t%.8lf\t%.8lf\t%.8lf\n", t*time_step, n[t], n2[t], n3[t], C2[t], C3[t], NC2[t], dNdt[t]);
	}
	
	fclose(fp);
	
	return;
}

// Initial condition

std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi> initial_condition_vector(){
	// The initial condition - in this case it's a product of three Fock state
	Eigen::VectorXi vA(L);
	Eigen::VectorXi vB(L);
	Eigen::VectorXi vC(L);
		
	for(int site=0; site<L; site++){
		//*
		if( site>=domain_wall ){
			vA[site] = 1;
			vB[site] = 1;
			vC[site] = 1;
		}
		else{
			vA[site] = 1;
			vB[site] = 0;
			vC[site] = 0;
		}
		//*/
		/*
		if( site%3==0 ){
			vA[site] = 1;
			vB[site] = 0;
			vC[site] = 0;
		}
		else if( site%3==1 ){
			vA[site] = 0;
			vB[site] = 1;
			vC[site] = 0;			
		}
		else if( site%3==2 ){
			vA[site] = 0;
			vB[site] = 0;
			vC[site] = 1;			
		}
		//*/
	}
	
	return std::make_tuple(vA, vB, vC);
}
			
// Main

int main( int argc, char** argv ){	

	//*	
	outputInizialization();

	std::cout << "System's parameters...";
	std::cout << "Length: " << L << "\n";
	std::cout << "(NA,NB,NC) = (" << NA << ", " << NB << ", " << NC << ")\n\n";
		
	std::cout << "Hopping rate = " << J << "\n";
	std::cout << "On site two-body interaction = " << U2 << "\n";
	std::cout << "On site three-body interaction = " << U3 << "\n";
	std::cout << "Loss rate = " << g << "\n\n";
		
	std::cout << "Number of trajectories per cpu = " << number_of_trajectories << "\n";
	std::cout << "Time step = " << time_step << "\n";
	std::cout << "Final time = " << final_time << "\n\n";
	
	Eigen::VectorXd global_n( timesteps_number+1 ), global_n2( timesteps_number+1 ), global_n3( timesteps_number+1 );
	Eigen::VectorXd global_C2( timesteps_number+1 ), global_C3( timesteps_number+1 ), global_NC2( timesteps_number+1 );
	Eigen::VectorXd global_dNdt( timesteps_number+1 );

	for(int t=0; t<global_n.size(); t++){
		global_n[t] = 0;
		global_n2[t] = 0;
		global_n3[t] = 0;

		global_C2[t] = 0;
		global_C3[t] = 0;
		global_NC2[t] = 0;

		global_dNdt[t] = 0;
	}
	
	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		int num_threads = omp_get_num_threads();

		if( thread_id == 0 ){
			std::cout << "Number of spawned threads " << num_threads << " threads\n";
			std::cout << "For a total of " << num_threads * number_of_trajectories << " trajectories\n\n";
			
			total_trajectories_number = num_threads * number_of_trajectories;
		}
		
		std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi> initial_condition = initial_condition_vector();

		bool verbose = false;

		if( thread_id == 0 ){
			verbose = true;
		}

		quantum_trajectory::quantum_trajectory qt(NA, NB, NC, L, J, U2, U3, g, number_of_trajectories, time_step, timesteps_number, initial_condition, pbc, verbose );
		qt.compute_trajectories();
		
		quantum_trajectory::results local_averages = qt.retrieve_results();
		Eigen::VectorXd local_n = local_averages.particle_number;
		Eigen::VectorXd local_n2 = local_averages.particle_number_squared;
		Eigen::VectorXd local_n3 = local_averages.particle_number_cubed;

		Eigen::VectorXd local_C2 = local_averages.casimir_C2;
		Eigen::VectorXd local_C3 = local_averages.casimir_C3;
		Eigen::VectorXd local_NC2 = local_averages.N_casimir_C2;

		Eigen::VectorXd local_dNdt = local_averages.dN_dt;

		// Average over nthreads
		#pragma omp critical
		{
			for(int t=0; t<local_n.size(); t++){
				global_n[t] += local_n[t] / num_threads;
				global_n2[t] += local_n2[t] / num_threads;
				global_n3[t] += local_n3[t] / num_threads;

				global_C2[t] += local_C2[t] / num_threads;
				global_C3[t] += local_C3[t] / num_threads;
				global_NC2[t] += local_NC2[t] / num_threads;

				global_dNdt[t] += local_dNdt[t] / num_threads;
			}
		}		
	}
					
	save_data(global_n, global_n2, global_n3, global_C2, global_C3, global_NC2, global_dNdt);
	//*/

	/*
	effective_hamiltonian::effective_hamiltonian NH_H(NA, NB, NC, L, J, U0, U2, U3, g, pbc, true );
	NH_H.hamiltonian_diagonalization( 50 );

	NH_H.spectrum_print();
	//*/

	return 0;
}
