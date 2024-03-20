#include "vranic_merger.h"
#include <chrono>
#include <omp.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

  
    V_Merger::V_Merger(int rank, int size)
      : m_verbosity(0)			   
      , m_min_npart_pcell(4)
      , m_min_dp_pcell{1e-10, 1e-10, 1e-10} 
    {
      m_mpi_rank=rank;
      m_mpi_size=size;
    }

    void V_Merger::merge(part_kine &pp, const size_t* x_bins, const size_t* p_bins){    

      //  
      // Particle resampling, Vranic and al, method
      //

      // First check consistency of kinematic store
      size_t npart = pp.len;
      for(size_t i=0; i < NDIMS ; i++)  assert( pp.x[i].size() == npart);
	  
      // create  mask array   
      m_mask_array.resize(pp.len);  
      for (size_t s_index=0; s_index < pp.len; s_index++) m_mask_array[s_index]=0;
            
      if (m_verbosity>1 && 0 == m_mpi_rank){
	std::cout << std::endl;
	std::cout << "Merger: rank: "<< m_mpi_rank
		  << " (x,y,z)   binning -> (" << x_bins[0] << "," << x_bins[1] << "," << x_bins[2] <<  ")"
		  << " (px,py,pz) binning -> (" << p_bins[0] << "," << p_bins[1] << "," << p_bins[2] << ")"
	          << " n_part/process: " << pp.len  
		  << std:: endl;
	std::cout << std::endl;
      }

      // Get parallel min max
      auto t1 = high_resolution_clock::now();
      
      // Find boundaries
      double xmin[3]={0.0,0.0,0.0};
      double xmax[3]={0.0,0.0,0.0};      
      
      for (size_t i=0; i < NDIMS; i++){        
	xmax[i] = *std::max_element(pp.x[i].begin(),pp.x[i].end());
	xmin[i] = *std::min_element(pp.x[i].begin(),pp.x[i].end());
        assert( (xmax[i]-xmin[i]) != 0 );
      }
            
      auto t2 = high_resolution_clock::now();      
      duration<double, std::milli> ms_double1 = t2 - t1;
      
      if ((m_verbosity>1) && (0 == m_mpi_rank)){ 
	std::cout << " npart: " << pp.len << std::endl;  
	std::cout << " Find min-max timing: "  << ms_double1.count() << "ms\n"
		  << std::endl;     	
	for (size_t i=0;i<NDIMS;i++){
	  std::cout << " i: " <<  i << " xmin: " << xmin[i]
		    << " xmax: "<< xmax[i] <<std::endl; 
	}
      }


      // (x,y,z) Binning 
      size_t dims = x_bins[0]*x_bins[1]*x_bins[2]; 
      double delta_x = fabs(xmax[0]-xmin[0])/x_bins[0];
      double delta_y = fabs(xmax[1]-xmin[1])/x_bins[1];
      double delta_z = fabs(xmax[2]-xmin[2])/x_bins[2];
      
      std::vector<size_t> cells_indexes[dims];
             
      for (size_t i=0;i<x_bins[0];i++){
	for(size_t j=0;j<x_bins[1]; j++){
	  for(size_t k=0; k<x_bins[2]; k++){
	    size_t  cell_index  = ( i * x_bins[1] + j ) * x_bins[2] + k;

	      if ((m_verbosity>1) && (0 == m_mpi_rank)){ 
		std::cout << "3D linear cell index i: " << i
			  << " j: " << j  << " k: " << k 
			  << " lin: " << cell_index
			  << std::endl;
		size_t i_x = (size_t) ((cell_index)/(x_bins[2]*x_bins[1])); 
		size_t i_y = (size_t) (cell_index/x_bins[2]) % x_bins[1];
		size_t i_z = (size_t) (cell_index % x_bins[2]);
		std::cout << "3D recalcuted  cell_index--> "
			  <<  " i_x: " << i_x 
			  <<  " i_y: " << i_y
			  <<  " i_z: " << i_z
			  << std::endl;
	      }
	      
	      double x_inf = xmin[0] + i * delta_x;
	      double x_sup = xmin[0] + (i+1) * delta_x;
	      double y_inf = xmin[1] + j * delta_y;
	      double y_sup = xmin[1] + (j+1) * delta_y;
	      double z_inf = xmin[2] + k* delta_z;
	      double z_sup = xmin[2] + (k+1) * delta_z;	      
	      
	      for (size_t l=0; l < npart;l++){
		if ( is_inside<double>(pp.x[0][l], x_inf, x_sup) &&
		     is_inside<double>(pp.x[1][l], y_inf, y_sup) &&
		     is_inside<double>(pp.x[2][l], z_inf, z_sup) 
		     )
		cells_indexes[cell_index].push_back(l);
	      }//!l
	    }//!k
	  }//!j
	}//!i
       
      size_t sum_points = 0;      
      if ((m_verbosity>1) && (0==m_mpi_rank)){
	for(size_t  i = 0; i < dims ; ++i){
	  size_t i_x = (size_t) ((i)/(x_bins[2]*x_bins[1])); 
	  size_t i_y = (size_t) (i/x_bins[2]) % x_bins[1];
	  size_t i_z = (size_t) (i % x_bins[2]);

	  std::cout << " cell_index : "<< i
		    << " x[" << xmin[0] + i_x*delta_x  << "," << xmin[0] + (i_x+1)*delta_x << "["
		    << " y[" << xmin[1] + i_y*delta_y << "," <<  xmin[1] + (i_y+1)*delta_y << "["
		    << " z[" << xmin[2] + i_z*delta_z << "," <<  xmin[2] + (i_z+1)*delta_z << "["	    	    
	            <<" contains npart:  "
		    << cells_indexes[i].size() << std::endl;
	  
	  sum_points+=cells_indexes[i].size(); 	   
	}
      }

      // Cross check mapping
      assert(sum_points != pp.len && "non-consistent particle to mesh mapping!");

      // Cartesian momentum merging using Vranic & al. method  
      p_cartesian(pp, x_bins, p_bins, cells_indexes);
      
    }//! merge


    void  V_Merger::p_cartesian(part_kine &pp, const size_t* x_bins, const size_t* p_bins,
				 std::vector<size_t> c_indexes[]){
      //
      // Momentum cell analysis
      //

      if ((m_verbosity>1) && (0 == m_mpi_rank)){  
      std::cout << "p_cartesian rank: " << m_mpi_rank << std::endl;
	for(size_t  i = 0; i < x_bins[0]*x_bins[1]*x_bins[2] ; ++i){
	  std::cout << " cell_index: "
		    << i <<" contains npart: "
		    << c_indexes[i].size() << std::endl;
	}		
      }

      size_t dims  = x_bins[0]*x_bins[1]*x_bins[2];
      size_t pdims = p_bins[0]*p_bins[1]*p_bins[2];

      size_t total_cells{0};
      size_t merged_cells{0};
      size_t total_part_processed{0};
      
      //Main loop over linearized cell indexes
      for(size_t  c_index = 0; c_index < dims ; c_index++){
	// Check nb. of particles in the cell
	size_t npart_pcell = c_indexes[c_index].size();
        // Select p_cell with at least min. particles	
	if (npart_pcell < m_min_npart_pcell)
	  { 
	    if ((m_verbosity>1) && (0 == m_mpi_rank)){
	      std::cout << " icell Linear: " << c_index
			<< " removing npart/cell:" << npart_pcell
			<< std::endl;
	    }
	    continue;		    
	  }	  

	// Define momentum tiles per cell 
	std::vector<double> p_cell[NDIMS];
	for(size_t i=0; i < NDIMS; i++) p_cell[i].resize(npart_pcell);
        for(size_t ic=0;ic<npart_pcell; ic++){
	  p_cell[0][ic] = pp.p[0][c_indexes[c_index][ic]];
	  p_cell[1][ic] = pp.p[1][c_indexes[c_index][ic]];
	  p_cell[2][ic] = pp.p[2][c_indexes[c_index][ic]];    	    
        }

	if ((m_verbosity>1) && (0 == m_mpi_rank)){
	  std::cout << " icell Linear: " << c_index
	       	    << " npart/cell:" << npart_pcell
		    << std::endl;
	  
	   for(size_t ii=0;ii<npart_pcell;ii++) std::cout 
					      << " px: " << p_cell[0][ii] 
					      << " py: " << p_cell[1][ii]
					      << " pz: " << p_cell[2][ii]
					      << std::endl;					      	  	  
	}

	//Define new binning in  momentum space  
	double p_min[3], p_max[3], dp[3], inv_dp[3];
	double p_rebin[3];
	bool p_binning[3]={true,true,true};
	size_t n_rebin[3]={0,0,0};
	
	for(size_t i=0; i < NDIMS ;i++){
	  p_min[i]=p_max[i]=dp[i]=inv_dp[i]=p_rebin[i]=0.0;
	}	

	// Check if auto_binning on
        //if (pdims<0 && !m_auto_binning) m_auto_binning=true;  
	//if (!m_auto_binning) m_auto_binning=true;

	
	if (m_auto_binning){	  
	  for(size_t i=0; i<NDIMS ;i++){
	    auto mm = minmax_element(p_cell[i].begin(), p_cell[i].end());
	    p_min[i] = *mm.first;
	    p_max[i] = *mm.second;
	    // Mark anomaly in direction space
	    if (fabs(p_max[i]-p_min[i]) == 0) p_binning[i] = false;	  
	    p_rebin[i] = re_binning(p_cell[i]);
            if (!p_binning[i]) p_rebin[i]=0; 	    
	    n_rebin[i] = (p_rebin[i]==0) ? 1 : std::ceil((p_max[i]-p_min[i])/p_rebin[i]);
	    if (p_binning[i]){
	      dp[i] = fabs(p_max[i] - p_min[i])/(n_rebin[i]);
	      inv_dp[i] = 1.0/dp[i];
	    }else{
	      dp[i]=0.0;
	      inv_dp[i]=1.0;
	    }
	    
	    if ((m_verbosity>1) && (0 == m_mpi_rank)){
	      std::cout << " icell Linear: " << c_index
			<< " dir= " << i
			<< " pmin: " << p_min[i]
			<< " pmax: " << p_max[i]
			<< " n_pbins: " << n_rebin[i]
			<< " inv_dp: " << inv_dp[i] 
			<< " p_binning: " << p_binning[i] 
			<< std::endl;	    
	    }	  
	  }//!for(3_D)
	  
	  if ((m_verbosity>1) && (0 == m_mpi_rank))
	    std::cout << "Auto p_binning: " << n_rebin[0] << " : "
		      << n_rebin[1] << " : " << n_rebin[2] << std::endl;	  
	}else{
	  for(size_t i=0; i<NDIMS ;i++){
	    auto mm = minmax_element(p_cell[i].begin(), p_cell[i].end());
	    p_min[i] = *mm.first;
	    p_max[i] = *mm.second;
	    // Mark anomaly in direction space
	    if (fabs(p_max[i]-p_min[i]) == 0) p_binning[i] = false;	  
	    // take p_binning from user input
	    n_rebin[i]=p_bins[i];
	    if (!p_binning[i]) n_rebin[i]=1;	    
	    if (p_binning[i]){
	      dp[i] = fabs(p_max[i] - p_min[i])/(n_rebin[i]);
	      inv_dp[i] = 1.0/dp[i];
	    }else{
	      dp[i]=0.0;
	      inv_dp[i]=1.0;
	    }
	  }
	  
	  for(size_t i=0; i < NDIMS; i++) 
	    if ((m_verbosity>1) && (0 == m_mpi_rank))
	      std::cout << "User defined p_binning : " <<  n_rebin[0] << " : "
			<< n_rebin[1] << " : " << n_rebin[2] << std::endl;
	}//!else m_auto_binning

	
	// Do the mapping p_cell <-> epoch_trk_indexes
	pdims=n_rebin[2]*n_rebin[1]*n_rebin[0];
	std::vector<size_t> p_cells_indexes[pdims];	
	for (size_t i=0;i<n_rebin[0];i++){
	  for(size_t j=0;j<n_rebin[1]; j++){
	    for(size_t k=0;k<n_rebin[2]; k++){
	      //Row major mapping 
	      size_t p_cell_index = (i*n_rebin[1]*n_rebin[2])+(j*n_rebin[2])+k;
	      
	      double px_inf = p_min[0] + i*dp[0];
	      double px_sup = p_min[0] + (i+1) * dp[0];
	      double py_inf = p_min[1] + j* dp[1];
	      double py_sup = p_min[1] + (j+1) * dp[1];
	      double pz_inf = p_min[2] + k* dp[2];
	      double pz_sup = p_min[2] + (k+1) * dp[2];	      

	      if ((m_verbosity>1) && (0 == m_mpi_rank)){ 
		std::cout << "3D linear cell index i: " << i
			  << " j: " << j  << " k: " << k 
			  << " lin: " << p_cell_index
			  << std::endl;
		size_t i_px = (size_t) ((p_cell_index)/(n_rebin[2]*n_rebin[1])); 
		size_t i_py = (size_t) (p_cell_index/n_rebin[2]) % n_rebin[1];
		size_t i_pz = (size_t) (p_cell_index % n_rebin[2]);
		std::cout << "3D recalcuted  p_index--> "
			  <<  " i_px: " << i_px 
			  <<  " i_py: " << i_py
			  <<  " i_pz: " << i_pz
			  << std::endl;
	      }
	      
	      for (size_t l=0; l < npart_pcell; l++){

		if ((m_verbosity>1) && (0 == m_mpi_rank)){	
		  std::cout << "3D Bornes  px: "<< p_cell[0][l]
			    << " px_inf: " << px_inf << " px_sup: "
			    << px_sup << std::endl;
		  std::cout << "3D Bornes  py: "<< p_cell[1][l]
			    << " py_inf: " << px_inf << " py_sup: "
			    << px_sup << std::endl;
		  std::cout << "3D Bornes  pz: "<< p_cell[2][l]
			    << " pz_inf: " << px_inf << " pz_sup: "
			    << px_sup << std::endl;		
		}
		
		if ( is_inside<double>(p_cell[0][l], px_inf, px_sup) &&
		     is_inside<double>(p_cell[1][l], py_inf, py_sup) &&
		     is_inside<double>(p_cell[2][l], pz_inf, pz_sup) 
		     )
		  p_cells_indexes[p_cell_index].push_back(c_indexes[c_index][l]);
	      }//!l
	      if ((m_verbosity>1) && (0 == m_mpi_rank))
		std::cout << "p_cell_index: " << p_cell_index
			  << " contains n_part: "
			  <<  p_cells_indexes[p_cell_index].size() << std::endl;
	      total_part_processed+= p_cells_indexes[p_cell_index].size();	      
	    }//!k
	  }//!j
	}//!i

	if ((m_verbosity>1) && (0 == m_mpi_rank))
	  std::cout << "p_reduction for cell c_index: " << c_index
		    << " containing n_particles/geom cell: "
		    << c_indexes[c_index].size()
	            << " containing n_particle/mom cell: "
	            <<   total_part_processed 
		    <<  std::endl;

      
	// Do particle reduction
	auto [total_pcell, merged_pcell] =  p_reduction(pp,n_rebin,p_min,dp,p_cells_indexes);

	total_cells  += total_pcell;
	merged_cells += merged_pcell;
	
	// Check particle distribution after Mapping 
	for (size_t i_pcell=0; i_pcell<pdims; i_pcell++){
	  if ((m_verbosity>1) && (0 == m_mpi_rank)){
	    if (p_cells_indexes[i_pcell].size()==0) continue;
	    std::cout << "After Reduction: Particle distribution: i_pcell: "
		      << i_pcell << " npart/cell: "
		      << p_cells_indexes[i_pcell].size() << std::endl;
	    for (size_t p=0; p<p_cells_indexes[i_pcell].size(); p++)
	      std::cout << " p_index: " << p
			<< " px: " << pp.p[0][p_cells_indexes[i_pcell][p]]
			<< " py: " << pp.p[1][p_cells_indexes[i_pcell][p]]
			<< " pz: " << pp.p[2][p_cells_indexes[i_pcell][p]]
			<< std::endl;
	  }
	}
       
	
      }//!for(c_index)

	if ((m_verbosity>0) && (0 == m_mpi_rank)){	 
	  std::cout << "p_cartesian statistics: total_cells: "
		    << total_cells << " n_merged_cells: "
		    << merged_cells<< " %(merged)_cells: "
		    << (merged_cells/(double)total_cells)*100. << " %" << std::endl;
	}

	
    }//!p_cartesian

        
    std::tuple<size_t, size_t>
    V_Merger::p_reduction(part_kine &pp, const size_t* p_bins, const double* p_min,
			   const double* dp, std::vector<size_t> p_indexes[]){
      
      //
      // Particle Merging Algorithm
      //          M. Vranic et al., CPC, 191 65-73 (2015)
      //          inspired by the SMILEI pic code  implementation
      //          https://smileipic.github.io/Smilei/Understand/particle_merging.html 
      //


    
      // Photon case not treated for the moment 
      if (!p_indexes[0].empty() && pp.mass == 0) return std::make_tuple(0,0);
      
      // P-Cell geometry
      int pdims{0};       
      // Total number of P-bins
      pdims = p_bins[0]*p_bins[1]*p_bins[2];

      int n_cell_merged=0;
      int total_part_processed=0;
      
      // Check particle distribution after Mapping 
      for (int i_pcell=0; i_pcell<pdims; i_pcell++){
	size_t npart_per_cell=p_indexes[i_pcell].size();

	if ((m_verbosity>1) && (0 == m_mpi_rank)){	 
	  std::cout << "before min_part_cell selection p_reduction: i_pcell: "
		    << i_pcell << " npart/cell: "
		    << npart_per_cell << std::endl;
	}
	
	// Selected only enough populated cells.
	if (npart_per_cell<m_min_npart_pcell) continue;

	// Increment for statitistic  	
	n_cell_merged++;
	total_part_processed+=npart_per_cell;
	m_total_part_processed+=npart_per_cell;
	
	if ((m_verbosity>1) && (0 == m_mpi_rank)){	 
	  std::cout << "p_reduction: selected i_pcell: "
		    << i_pcell << " npart/cell: "
		    << npart_per_cell <<  " total_part_processed: "
		    << total_part_processed <<std::endl;	  
	}
	
	// Init total cell quantities
	double tot_w{0.0};
	double tot_px{0.0};
	double tot_py{0.0};
	double tot_pz{0.0};
	double tot_e{0.0};   
	double mo{0.0};
	
	for (size_t p=0; p<npart_per_cell; p++)
	  {
	    // Total weight (wt)
	    tot_w  += pp.w[p_indexes[i_pcell][p]];	    
	    // total momentum  (pt)
	    tot_px += pp.p[0][p_indexes[i_pcell][p]]*pp.w[p_indexes[i_pcell][p]];
	    tot_py += pp.p[1][p_indexes[i_pcell][p]]*pp.w[p_indexes[i_pcell][p]];
	    tot_pz += pp.p[2][p_indexes[i_pcell][p]]*pp.w[p_indexes[i_pcell][p]];	    

	    // gamma = sqrt( 1 + (p/mo*c)**2)
	    mo = pp.mass;
	    double fac = pp.p[0][p_indexes[i_pcell][p]] * pp.p[0][p_indexes[i_pcell][p]]	      
	               + pp.p[1][p_indexes[i_pcell][p]] * pp.p[1][p_indexes[i_pcell][p]]
	               + pp.p[2][p_indexes[i_pcell][p]] * pp.p[2][p_indexes[i_pcell][p]];
	    fac = fac / (mo * mo * C_LIGHT * C_LIGHT); 
	    double gamma = sqrt( 1 + fac );

	    // E_tot ( check me !)
	    tot_e +=  gamma * pp.w[p_indexes[i_pcell][p]] * mo * C_LIGHT * C_LIGHT;
	   	    
	    if ((m_verbosity>1) && (0 == m_mpi_rank)){	 
	      std::cout.precision(8);
	      std::cout << " p_index: " << p
			<< " px: " << pp.p[0][p_indexes[i_pcell][p]] << " [kg m s^-1] "
			<< " py: " << pp.p[1][p_indexes[i_pcell][p]] << " [kg m s^-1] "
			<< " pz: " << pp.p[2][p_indexes[i_pcell][p]] << " [kg m s^-1] "
		        << " gamma: " << gamma   
			<< " me: " << pp.mass << " [kg] "
			<< std::endl;
	    }
	  }//!for(particle/cell)
	
	// 3D index map <check-me>
	size_t i_px = (size_t) ((i_pcell)/(p_bins[2]*p_bins[1])); 
	size_t i_py = (size_t) (i_pcell/p_bins[2]) % p_bins[1];
	size_t i_pz = (size_t) (i_pcell % p_bins[2]);
	
	if ((m_verbosity>1) && (0 == m_mpi_rank)){	 
	  std::cout << "P-cell dim: " << p_bins[0]*p_bins[1]*p_bins[2]
		    << " linear: " << i_pcell 
		    << " i_px: " << i_px
		    << " i_py: "  << i_py
		    << " i_pz: "  << i_pz	      
		    << " re_linear 2D: " << i_px*p_bins[1] + i_py
		    << " re_linear 3D: " << (i_px*p_bins[1]*p_bins[2])+(i_py*p_bins[2])+i_pz	      
		    << " tot_w: " << tot_w
		    << " tot_px: " << tot_px 
		    << " tot_py: " << tot_py 
		    << " tot_pz: " << tot_pz
	            << " tot_e: " << tot_e
		    << " rest_mo: " << mo
		    << std::endl; 
	}
	
	// Vranic et al. : epsilon_a, pa e2= (pc)2 + (m0c2)2
	double eps_a = tot_e/tot_w;
	double pa = std::sqrt(std::pow(eps_a,2)-std::pow(mo*C_LIGHT*C_LIGHT,2))/C_LIGHT;
	
	// Total p_norm
	double tot_p_norm = std::sqrt(
				      std::pow(tot_px,2)
				      +std::pow(tot_py,2)
				      +std::pow(tot_pz,2)
				      );
	
	// Vranic et al: angle between pa and pt, pb and pt 
	double cos_w = std::min(tot_p_norm / (tot_w*pa),1.0);
	double sin_w = std::sqrt(1 - cos_w*cos_w);
	
	// Inverse total p
	double inv_tot_p_norm = 1./tot_p_norm;
	
	// Computation of u1 unit vector
	double u1_x = tot_px*inv_tot_p_norm;
	double u1_y = tot_py*inv_tot_p_norm;
	double u1_z = tot_pz*inv_tot_p_norm; //0. in 2D case
	
	// Vranic et al. vec_d vector
	double d_vec_x = p_min[0] + (i_px+0.5)*dp[0];
	double d_vec_y = p_min[1] + (i_py+0.5)*dp[1];
	double d_vec_z = p_min[2] + (i_pz+0.5)*dp[2];//0. in 2D case
	
	// u3 = u1 x d_vec
	double u3_x = u1_y*d_vec_z - u1_z*d_vec_y; // 0. in 2D case 
	double u3_y = u1_z*d_vec_x - u1_x*d_vec_z; // 0. in 2D case
	double u3_z = u1_x*d_vec_y - u1_y*d_vec_x; // (!=0) ,along Z. dir
	
	
	  if ((m_verbosity>1) && (0 == m_mpi_rank)){	 
	    std::cout << " eps_a: " << eps_a
		      << " pa: " << pa
		      << " tot_p_norm: " << tot_p_norm 
		      << " cos_w: " << cos_w
		      << " sin_w: " << sin_w
	              << " u1_x: " << u1_x
	              << " u1_y: " << u1_y
	              << " u1_z: " << u1_z
	              << " d_x: " <<  d_vec_x
	              << " d_y: " <<  d_vec_y
	              << " d_z: " <<  d_vec_z
	              << " u3_x: " << u3_x
	              << " u3_y: " << u3_y
	              << " u3_z: " << u3_z	      
	              << std::endl; 
	  }
	  
	  // All particle momenta are not null
	  if (fabs(u3_x*u3_x + u3_y*u3_y + u3_z*u3_z) > 0)
	    {
	      
	      double u2_x = u1_y*u3_z - u1_z*u3_y;
	      double u2_y = u1_z*u3_x - u1_x*u3_z;
	      double u2_z = u1_x*u3_y - u1_y*u3_x;
	      
	      double u2_norm = 1./sqrt(u2_x*u2_x + u2_y*u2_y + u2_z*u2_z);
	      
	      // u2 normalized 
	      u2_x = u2_x * u2_norm;
	      u2_y = u2_y * u2_norm;
	      u2_z = u2_z * u2_norm;
	      
              // Select only first particle
	      // Tagging all others to be removed ...
	      
	      if ((m_verbosity>1) && (0 == m_mpi_rank)){
		std::cout << " cond. fabs() > 0 "
			  << " u2_x: " << u2_x
			  << " u2_y: " << u2_y
			  << " u2_z: " << u2_z
		          << std::endl; 
	      }
	      
	      // Update momentum of 2 first particles in the cell
              // First particle
	      pp.p[0][p_indexes[i_pcell][0]] = pa*(cos_w*u1_x + sin_w*u2_x);
	      pp.p[1][p_indexes[i_pcell][0]] = pa*(cos_w*u1_y + sin_w*u2_y);
	      pp.p[2][p_indexes[i_pcell][0]] = pa*(cos_w*u1_z + sin_w*u2_z);
	      pp.w[p_indexes[i_pcell][0]]  = 0.5*tot_w;
	      
	      // Second particle
	      pp.p[0][p_indexes[i_pcell][1]] = pa*(cos_w*u1_x - sin_w*u2_x);
	      pp.p[1][p_indexes[i_pcell][1]] = pa*(cos_w*u1_y - sin_w*u2_y);
	      pp.p[2][p_indexes[i_pcell][1]] = pa*(cos_w*u1_z - sin_w*u2_z);
	      pp.w[p_indexes[i_pcell][1]]  = 0.5*tot_w;

	      
	      // Mask the other indexes in the cell
	      for (size_t p=2; p<npart_per_cell; p++){
		m_mask_array[p_indexes[i_pcell][p]]=-1;
	      }
	    }//!(fabs()>0)
      }//!for(p-cells)

      
      if ((m_verbosity>1) && (0 == m_mpi_rank)){	 
	std::cout << "p_reduction: n_tot_pcell: "
		  << pdims << " n_merged_cells: "
		  << n_cell_merged
	          << " total_part_processed " << m_total_part_processed     
		  << " masked indexes: " << m_mask_array.size()
		  << std::endl;
      }
	
	return  std::make_tuple(pdims, n_cell_merged);
    }//! p_reduction
    

    part_kine V_Merger::get_reduced_kine(part_kine& pp){
      // reduce kine arrays
      part_kine out;
      out.charge = pp.mass;  
      out.mass = pp.charge;
      size_t count{0};
      for (size_t part_index=0;part_index<pp.len; part_index++){
	if (m_mask_array[part_index]== 0){
	  // Do Mapping (x, p, w) for internal struct.
	  for(size_t i=0; i < NDIMS ; i++){	  
	    out.x[i].push_back(pp.x[i][part_index]);
	    out.p[i].push_back(pp.p[i][part_index]);
	    if (i==0) out.w.push_back(pp.w[part_index]);
	  }
	    count++;	  
	}
      }//!part_index

      // check consistency
     
      for(size_t i=0; i < NDIMS ; i++){
	assert(out.x[i].size() == count);
	assert(out.p[i].size() == count);	
	if (i==0) assert(out.w.size() == count );
      }
      // assign new length
      out.len=count;
      return out;
    }
   
