#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>

// MPI
#include "mpi.h"

// Utils
#include "utils.h"

using namespace std;

    
    class V_Merger {
    private:
      int m_verbosity{0};
      size_t m_min_npart_pcell{0};
      double m_min_dp_pcell[3];
      int m_mpi_rank{-1};
      int m_mpi_size{-1};
      std::vector<size_t> m_mask_indexes;
      bool m_auto_binning{false};
      size_t m_total_part_processed{0};
      std::vector<size_t> m_mask_array;

    public:
      V_Merger(int rank, int size);
      virtual ~V_Merger(){;}
      
      void merge(part_kine &pp, const size_t* x_bins, const size_t* p_bins);   
      void p_cartesian(part_kine &pp, const size_t* x_bins, const size_t* p_bins, std::vector<size_t>* c_indexes);      

      std::tuple<size_t, size_t> p_reduction(part_kine &pp, const size_t* p_bins, const double* p_min,
				       const double* dp, std::vector<size_t>* c_indexes);
      part_kine get_reduced_kine(part_kine& pp);
      void setVerbose(int val){m_verbosity=val;}
      void setMinNpartPerCell(int val){m_min_npart_pcell=val;}
      void setMinDpPerCell(int i, double val){m_min_dp_pcell[i]=val;}
      void setMpiInfo(int i, int j){m_mpi_rank=i;m_mpi_size=j;}
      std::vector<size_t> get_mask_indexes(){return m_mask_indexes;}
      std::vector<size_t> get_mask_array(){return m_mask_array;}      
    };

