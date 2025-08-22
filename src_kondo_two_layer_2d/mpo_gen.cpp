//
// Created by Haoxin Wang on 3/7/2025.
//
/*
 * 2-layer 2-leg Kondo lattice model MPO generator
 * This program generates and dumps the MPO tensors for the Kondo model
 */

 #include "qlten/qlten.h"
 #include "qlmps/qlmps.h"
 #include "../src_kondo_1D/kondo_hilbert_space.h"
 #include "params_case.h"
 #include "../src_single_orbital/myutil.h"
 
 using namespace qlmps;
 using namespace qlten;
 using namespace std;
 
 int main(int argc, char *argv[]) {
   MPI_Init(nullptr, nullptr);
   MPI_Comm comm = MPI_COMM_WORLD;
   int rank, mpi_size;
   MPI_Comm_size(comm, &mpi_size);
   MPI_Comm_rank(comm, &rank);
 
   CaseParams params(argv[1]);
   size_t Lx = params.Lx; // Lx should be even number, for N/4 should on electron site for measure
   size_t Ly = 2;
   double t = params.t, Jk = params.JK, U = params.U;
   double Jperp = params.Jperp;
   double t2 = params.t2;
   size_t N = 4 * Ly * Lx; // 4 for double layer times two orbital (localized & itinerate)
   // order of sites for fixed Lx :
   // (layer0, ly0) ---> (layer1, ly0)
   // ---> (layer0, ly1) ----> (layer1, ly1)
 
   /*** Print the model parameter Info ***/
   if (rank == 0) {
     cout << "Lx = " << Lx << endl;
     cout << "Ly = " << Ly << endl;
     cout << "N = " << N << endl;
     cout << "t = " << t << endl;
     cout << "t2 = " << t2 << endl;
     cout << "Jk = " << Jk << endl;
     cout << "U = " << U << endl;
     cout << "Jperp = " << Jperp << endl;
     cout << "Geometry = " << params.Geometry << endl;
   }
 
   clock_t startTime, endTime;
   startTime = clock();
 
   std::vector<IndexT> pb_set = std::vector<IndexT>(N);
   for (size_t i = 0; i < N; ++i) {
     if (i % 2 == 0) pb_set[i] = pb_outE;   // even site is extended electron
     if (i % 2 == 1) pb_set[i] = pb_outL;   // odd site is localized electron
   }
   const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(pb_set);
   auto mpo_gen = MPOGenerator<TenElemT, QNT>(sites);
 
   HubbardOperators<TenElemT, QNT> hubbard_ops;
   auto &ops = hubbard_ops;
   SpinOneHalfOperatorsU1U1 local_spin_ops;
   auto f = hubbard_ops.f;
   
   //hopping along x direction
   for (size_t i = 0; i < N - 4 * Ly; i = i + 2) {
     size_t site1 = i, site2 = i + 4 * Ly;//4 for double layer times two orbital (localized & itinerate)
     std::vector<size_t> inst_op_idxs; //even sites between site1 and site2
     for (size_t j = site1 + 2; j < site2; j += 2) {
       inst_op_idxs.push_back(j);
     }
     mpo_gen.AddTerm(-t, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2, f, inst_op_idxs);
     mpo_gen.AddTerm(t, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2, f, inst_op_idxs);
     mpo_gen.AddTerm(-t, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2, f, inst_op_idxs);
     mpo_gen.AddTerm(t, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2, f, inst_op_idxs);
   }
   
   //hopping along y direction, assume Ly = 2
   for (size_t i = 0; i < N - 6; i += 4 * Ly) {
     size_t site1 = i, site2 = i + 4; // 0-th layer
     mpo_gen.AddTerm(-t, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2, f, {site1 + 2});
     mpo_gen.AddTerm(t, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2, f, {site1 + 2});
     mpo_gen.AddTerm(-t, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2, f, {site1 + 2});
     mpo_gen.AddTerm(t, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2, f, {site1 + 2});
 
     site1 = i + 2, site2 = i + 6;   // 1-th layer
     mpo_gen.AddTerm(-t, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2, f, {site1 + 2});
     mpo_gen.AddTerm(t, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2, f, {site1 + 2});
     mpo_gen.AddTerm(-t, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2, f, {site1 + 2});
     mpo_gen.AddTerm(t, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2, f, {site1 + 2});
   }
 
   // Hubbard U term on extended electron sites
   for (size_t i = 0; i < N; i += 2) {
     mpo_gen.AddTerm(U, hubbard_ops.nupndn, i);
   }
 
   // Kondo coupling between extended and localized electrons
   for (size_t i = 0; i < N - 1; i = i + 2) {
     mpo_gen.AddTerm(Jk, hubbard_ops.sz, i, local_spin_ops.sz, i + 1);
     mpo_gen.AddTerm(Jk / 2, hubbard_ops.sp, i, local_spin_ops.sm, i + 1);
     mpo_gen.AddTerm(Jk / 2, hubbard_ops.sm, i, local_spin_ops.sp, i + 1);
   }
 
   //inter layer AFM coupling
   for (size_t i = 1; i < N - 2; i = i + 4) { // for each unit cell
     mpo_gen.AddTerm(Jperp, local_spin_ops.sz, i, local_spin_ops.sz, i + 2);
     mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sp, i, local_spin_ops.sm, i + 2);
     mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sm, i, local_spin_ops.sp, i + 2);
   }
 
   qlmps::MPO<Tensor> mpo = mpo_gen.Gen();
 
   // Dump MPO tensors to files
   if (rank == 0) {
     std::string mpo_path = "mpo";
     if (!IsPathExist(mpo_path)) {
       system(("mkdir -p " + mpo_path).c_str());
     }
     
     for (size_t i = 0; i < N; i++) {
       std::string filename = mpo_path + "/mpo_ten" + std::to_string(i) + ".qlten";
       std::ofstream file(filename, std::ios::binary);
       file << mpo[i];
       file.close();
       std::cout << "Dumped MPO tensor " << i << " to " << filename << std::endl;
     }
     
     endTime = clock();
     cout << "MPO generation and dumping completed!" << endl;
     cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
   }
 
   MPI_Finalize();
   return 0;
 }