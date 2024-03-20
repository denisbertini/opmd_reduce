#include <openPMD/openPMD.hpp>

#include <stdio.h> 
#include <unistd.h> 
#include <iostream>
#include <memory>
#include <cstddef>
#include <string>
#include <typeinfo>

#include "opmd_reducer.h"

using std::cout;
using namespace openPMD;

void Opmd_Reducer::get_fullinfo(const std::string& fname){

  Series o = Series(fname, Access::READ_ONLY);

    std::cout << "Read iterations ";
    for( auto const& val : o.iterations )
        std::cout << '\t' << val.first;
    
    std::cout << "\n Read attributes in the root:\n";
    for( auto const& val : o.attributes() )
        std::cout << '\t' << val << '\n';
    std::cout << '\n';

    std::cout << "basePath - " << o.basePath() << '\n'
              << "iterationEncoding - " << o.iterationEncoding() << '\n'
              << "iterationFormat - " << o.iterationFormat() << '\n'
              << "meshesPath - " << o.meshesPath() << '\n'
              << "openPMD - " << o.openPMD() << '\n'
              << "openPMDextension - " << o.openPMDextension() << '\n'
      //  << "particlesPath - " << o.particlesPath() << '\n'
              << '\n';

    std::cout << "Read attributes in basePath:\n";
    for( auto const& a : o.iterations.attributes() )
        std::cout << '\t' << a << '\n';
    std::cout << '\n';

    std::cout << "Read iterations in basePath:\n";
    for( auto const& i : o.iterations )
        std::cout << '\t' << i.first << '\n';
    std::cout << '\n';

    for( auto const& i : o.iterations )
    {
        std::cout << "Read attributes in iteration " << i.first << ":\n";
        for( auto const& val : i.second.attributes() )
            std::cout << '\t' << val << '\n';
        std::cout << '\n';

        std::cout << i.first << ".time - " << i.second.time< float >() << '\n'
                  << i.first << ".dt - " << i.second.dt< float >() << '\n'
                  << i.first << ".timeUnitSI - " << i.second.timeUnitSI() << '\n'
                  << '\n';

        std::cout << "Read attributes in meshesPath in iteration " << i.first << ":\n";
        for( auto const& a : i.second.meshes.attributes() )
            std::cout << '\t' << a << '\n';
        std::cout << '\n';

        std::cout << "Read meshes in iteration " << i.first << ":\n";
        for( auto const& m : i.second.meshes )
            std::cout << '\t' << m.first << '\n';
        std::cout << '\n';

    }
}


void Opmd_Reducer::get_kinematics(const std::string& filename, const std::string& species_name){

      int mpi_size{-1};
      int mpi_rank{-1};
      
      MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
      
      try{	
	// Open series
	std::string tag = "Processing: " + filename;    
	Timer kk(tag, mpi_rank);
	Series series = Series(filename, Access::READ_ONLY, MPI_COMM_WORLD);
	
	//Check one iteration in the file
	int n_iter = series.iterations.size();
	if (n_iter != 1) throw std::runtime_error("-E- Only 1 iteration/opmd_file allowed !"); 
	
	// Get iteration
	int i_step{0};
	for( auto const& val : series.iterations ){
	  i_step=val.first;
	  m_istep=val.first;
	  if (0==mpi_rank) std::cout << "-I- Found iteration: "
				     << i_step << " : " << m_istep  << std::endl;
	}
	
	// Get unique iteration
	Iteration iteration = series.iterations[i_step];
	
	if (0 == mpi_rank)
	  {
	    std::cout << "  " << series.iterationEncoding() << std::endl;
	    std::cout << "-I- Scanning  file: " <<  filename << std::endl;
	    std::cout << "-I- Found  n_iter: " << n_iter << std::endl;
	    cout << "-I- Iteration: " << i_step << " contains " << iteration.meshes.size() << " meshes:";
	    
	    for (auto const &m : iteration.meshes)
	      cout << "\n\t" << m.first;
	    cout << '\n';
	    
	    cout << "-I- Iteration: " << i_step << " contains " << iteration.particles.size()
		 << " particle species:";
	    
	    for (auto const &ps : iteration.particles)
	      {
		cout << "\n\t" << ps.first;            
		/* attributes
		   for (auto const &r : ps.second)
		   {
		   cout << "\n\t" << r.first;
		   cout << '\n';
		   }
		*/
	      }
	    std::cout << std::endl;
	  }
	
	// Create corresponding particle species
	ParticleSpecies species = iteration.particles[species_name.c_str()]; 
	
	// Define kinematics storage
	std::array<std::shared_ptr<double>, 3> part_kine_x;
	std::array<std::shared_ptr<double>, 3> part_kine_p;
	std::array<std::shared_ptr<double>, 1> part_kine_w;        
	std::array<Extent, NDIMS> extents;
	
	// Get number of particles
	std::string const &position = m_part_attr[0];
	std::string const &x_comp = m_dimensions[0];      
	Record part_rec = species[position.c_str()];      
	RecordComponent part_comp = part_rec[x_comp.c_str()];
	Extent part_Extent = part_comp.getExtent();
	size_t n_total_part=part_Extent[0];
	
	if (0==mpi_rank){
	  std::cout <<"-I- rank#: " << mpi_rank << " n_total_part#: " << n_total_part
		    << " mpi_size: " << mpi_size << " dn_part: " << part_Extent[0]/mpi_size
		    << " : " << floor(part_Extent[0]/mpi_size) << std::endl;
	}
	
	// Read particles information in chunk     
	size_t dn_part = (size_t) (part_Extent[0]/mpi_size);
	size_t chunk_start{0};
	size_t chunk_end{0};
	chunk_start = mpi_rank * dn_part;
	if (mpi_rank < (mpi_size - 1 )){
	  chunk_end = dn_part;
	}else{
	  chunk_end = part_Extent[0] - ((mpi_size-1) * dn_part);
	}

	//if (0==mpi_rank)
	std::cout <<"-I- rank#: " << mpi_rank << " dn_part#: " << dn_part << " offset: "
		  << chunk_start << " extent: " << chunk_end << std::endl;
	
	// Prepare the 
	Offset chunk_offset = {chunk_start};
	Extent chunk_extent = {chunk_end};
	
	// Load full kinematics
	// <DB> Improve moving flush an fill() inside loop
	for (size_t i = 0; i < NDIMS; i++){
	  Record part_record = species[m_part_attr[i].c_str()];
	  if (m_part_attr[i] != "weighting"){
	    for (size_t j = 0; j < NDIMS; j++){
	      std::string const &dim = m_dimensions[j];
	      RecordComponent rc = part_record[dim];
	      if(m_part_attr[i] == "position"){
		part_kine_x[j] = rc.loadChunk<double>(chunk_offset, chunk_extent);
	      }else
		part_kine_p[j] = rc.loadChunk<double>(chunk_offset, chunk_extent);	      	      
	    }
	  } else
	    {
	      RecordComponent wc = part_record[openPMD::RecordComponent::SCALAR];
	      part_kine_w[0] =  wc.loadChunk<double>(chunk_offset, chunk_extent);
			  
	    }
	}//!(records)

	std::shared_ptr<double> charge =
	  species["charge"][openPMD::RecordComponent::SCALAR].loadChunk<double>(chunk_offset, chunk_extent);
	std::shared_ptr<double> mass =
	  species["mass"][openPMD::RecordComponent::SCALAR].loadChunk<double>(chunk_offset, chunk_extent);

	
	// fsync() && close() to release ressources
	series.flush();
	iteration.close();
	
	if (0 == mpi_rank){
	  std::cout <<"-I- rank#: " << mpi_rank << " dn_part: " 
		    << dn_part << " mass: " << *mass.get() << " charge: " << *charge.get() << std::endl;	
	  std::cout << " chunk_extent[0]: " << chunk_extent[0] << " : " << chunk_extent[1] << std::endl;

	  /* for (int l=0; l<10; l++) 
	    std::cout << "i: " << l 
		      << " px: " <<  part_kine_p[0].get()[l]
		      << " py: " << part_kine_p[1].get()[l]
		      << " pz: " << part_kine_p[2].get()[l]
		      << std::endl;
	  */
	  
	  }	

	// Kinematics Store 
	m_part_kine.len = dn_part;	  
	m_part_kine.charge = *charge.get();  
        m_part_kine.mass = *mass.get();
		
	// Do Mapping (x, p, w) for internal struct.
	for(size_t i=0; i < NDIMS ; i++){	  
	  for(size_t j=0; j < dn_part; j++){
	    m_part_kine.x[i].push_back((part_kine_x[i].get()[j]));
	    m_part_kine.p[i].push_back((part_kine_p[i].get()[j]));
	    if (i==0) m_part_kine.w.push_back((part_kine_w[i].get()[j]));
	  }	  
	}
	
      }//! try
      
      catch (std::exception &ex)
	{
	  if (0 == mpi_rank)
	    {
	      std::cerr << ex.what() << std::endl;
	    }
	}
      
}

void Opmd_Reducer::do_merging(const int n_part_pcell, const size_t n_bins[], const size_t p_bins[]){
  
  int mpi_size{-1};
  int mpi_rank{-1};
  
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  
  try{
        V_Merger pm(mpi_rank, mpi_size);
	pm.setVerbose(1);
	pm.setMinNpartPerCell(n_part_pcell);
	pm.merge(m_part_kine, n_bins, p_bins);
	//part_kine kine_reduced = pm.get_reduced_kine(m_part_kine);  
	    // Get mask indexes
	std::vector<size_t> vec_mask = pm.get_mask_indexes();
	std::vector<size_t> mask_array = pm.get_mask_array();


	// Copy full->reduced
	part_kine kine_reduced;
	kine_reduced.charge = m_part_kine.charge;  
	kine_reduced.mass   = m_part_kine.mass;

	size_t count{0};
	for (size_t part_index=0;part_index<m_part_kine.len; part_index++){
	  if (mask_array[part_index]== 0){
	    // Do Mapping (x, p, w) for internal struct.
	    for(size_t i=0; i < NDIMS ; i++){	  
	      kine_reduced.x[i].push_back(m_part_kine.x[i][part_index]);
	      kine_reduced.p[i].push_back(m_part_kine.p[i][part_index]);
	      if (i==0) kine_reduced.w.push_back(m_part_kine.w[part_index]);
	    }
	    count++;	  
	  }
	}//!part_index


	/*
	if (0 == mpi_rank){
          for (int l=0; l<10; l++) 
	    std::cout << "reduced i: " << l 
		      << " px: " <<  kine_reduced.p[0][l]
		      << " py: " <<  kine_reduced.p[1][l]
		      << " pz: " <<  kine_reduced.p[2][l]
		      << std::endl;
	  }	
	*/ 
	
	// Check the consitency count / vector(size)
	for (size_t i=0; i<NDIMS; i++){
	  assert(kine_reduced.x[i].size() == count);
	  if (i == 0) assert(kine_reduced.w.size() == count);
	}

	// Assign the tracks extension 
        kine_reduced.len=count;	
	
	// Get MPI know the reduction factor
	int ntracks_proc = (int) kine_reduced.len;
	int *ntracks = (int *)malloc(sizeof(int) * mpi_size);
	
	for (int i=0;i<mpi_size;i++) ntracks[i]=0;
	
	MPI_Allgather(&ntracks_proc, 1, MPI_INT,  ntracks, 1, MPI_INT,  MPI_COMM_WORLD);
	
	std::cout << "rank: " << mpi_rank
		  << " initial npart: " << m_part_kine.len
		  << " npart_reduced: " << kine_reduced.len 
		  << " final npart: " << ntracks[mpi_rank]
		  << " reduction level: "
		  <<  (1.-((double) ntracks[mpi_rank])/((double)m_part_kine.len)) * 100. << " %"
		  <<  std::endl;
	std::cout << "" << std::endl;  

	//
	// Open_pmd I/O
	//
	
	// Creating series
	Series o_series= Series(m_output_file.c_str(), Access::CREATE, MPI_COMM_WORLD);
	o_series.setAuthor("d.bertini@gsi.de");
	o_series.setMachine("Virgo3");
	o_series.setSoftwareDependencies("https://git.gsi.de/d.bertini/pp-containers/prod/rlx8_ompi_ucx.def");
	o_series.setParticlesPath("particles/");
	o_series.setIterationEncoding(IterationEncoding::fileBased);
	o_series.setIterationFormat("z_opmd_%06T.h5");
	
	// In parallel contexts, it's important to explicitly open iterations.
	o_series.iterations[m_istep].open();
       		
	// Re-compute total nb. of particles 
	size_t e_npart=0;
	for (int i=0; i<mpi_size; i++) {
	  e_npart+=ntracks[i];
	  if ( 0 == mpi_rank ){
	    std::cout << " rank: " << i << " ntracks/rank: "
		      << ntracks[i] << " tot: " << e_npart << std::endl;
	  }
	} 
		
	// Create Particle species
	ParticleSpecies e = o_series.iterations[m_istep].particles[m_species.c_str()];	    
	// Create Dataset
	Datatype datatype = determineDatatype<double>();
	Extent global_extent = {e_npart};
	Dataset dataset = Dataset(datatype, global_extent);
	
	if (0 == mpi_rank)
	  cout << "Prepared a Dataset of size " << dataset.extent[0] 
	       << " and Datatype " << dataset.dtype
	       << '\n';    
	
	// Recalculate particle distribution/proc
	size_t e_start=0;
	for (int i=0; i<mpi_rank; i++) e_start+=ntracks[i];
	
	Offset chunk_offset = {e_start};
	Extent chunk_extent = {ntracks[mpi_rank]};

       
	// Store reduced  kinematics
	for (size_t i = 0; i < NDIMS; i++){
	  Record part_record = e[m_part_attr[i].c_str()];
	  if (m_part_attr[i] != "weighting"){
	    for (size_t j = 0; j < NDIMS; j++){
	      std::string const &dim = m_dimensions[j];	      
	      RecordComponent rc = part_record[dim];
	      if(m_part_attr[i] == "position"){		
	       rc.resetDataset(dataset);
	       rc.storeChunkRaw( &(kine_reduced.x[j].data()[0]), chunk_offset, chunk_extent);
	      }else { 
	       rc.resetDataset(dataset);
	       rc.storeChunkRaw( &(kine_reduced.p[j].data()[0]), chunk_offset, chunk_extent);
	      }
	    }
	  } else
	    {
	      RecordComponent rc = part_record[openPMD::RecordComponent::SCALAR];
	      rc.resetDataset(dataset);
	      rc.storeChunkRaw( &(kine_reduced.w.data()[0]), chunk_offset, chunk_extent);
	    }
	}//!(records)

	Dataset s_dset = Dataset(datatype, Extent{1});	
	// Add charge and mass 
	RecordComponent rc_c = e["charge"][openPMD::RecordComponent::SCALAR];	
	rc_c.resetDataset(s_dset);
	rc_c.storeChunkRaw( &(kine_reduced.charge), Offset{0}, Extent{1});
	RecordComponent rc_m = e["mass"][openPMD::RecordComponent::SCALAR];	
	rc_m.resetDataset(s_dset);
	rc_m.storeChunkRaw( &(kine_reduced.mass), Offset{0}, Extent{1});	      			  	  
       
	// Free dyn. memory used by gather
	free(ntracks);
	// Sync to disk
	o_series.flush();	      	
	
  }// !try
  
  catch (std::exception &ex)
    {
      if (0 == mpi_rank)
	{
	  std::cerr << ex.what() << std::endl;
	}
    }
}

int Opmd_Reducer::reduce(int argc, char *argv[]){
  
  int mpi_rank{-1};
  int mpi_size{-1};
  
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
  std::string opmd_dir;
  std::string opmd_files;
  std::string species_name;
  std::string compression_name;      
  std::string output_format;
  
  cxxopts::Options optparse("reduce", "Particle Reduction");
  optparse.add_options()(
     "p,dir", "opendPMD dir to process",
     cxxopts::value<std::string>(opmd_dir))
    ("f,opmd_file", "opmd file to reduce",
     cxxopts::value<std::string>(opmd_files))    
    ("s,species_name", "species_name",
     cxxopts::value<std::string>(species_name))
    ("z,compression", "compression",
     cxxopts::value<std::string>(compression_name))	
    ("o,output_format", "output_format",
     cxxopts::value<std::string>(output_format));
  
  auto opts = optparse.parse(argc, argv);
  
  if (mpi_rank == 0 ){
    std::cout << std::endl;
    std::cout << "-I- opmd_reducer: directory to process: " << opmd_dir.c_str() << std::endl;
    std::cout << "-I- opmd_reducer: files to process: " << opmd_files.c_str() << std::endl;
    std::cout << "-I- opmd_reducer: species name list: " << species_name.c_str() << std::endl;
    std::cout << "-I- opmd_reducer: compression: " << compression_name.c_str() << std::endl;	  
    std::cout << "-I- opmd_reducer: output format: " << output_format.c_str() << std::endl;
    std::cout << std::endl;
  }

  // Get every files 
  std::vector<std::string> opmd_file_list;
  if ( !opmd_files.empty() ){
    opmd_file_list.push_back(opmd_files.c_str());
  }
 
  // Main loop over sdf files 
  for(std::string opmd_file : opmd_file_list)
    {
      // Input filename
      std::string opmd_file_full=opmd_dir+"/"+ opmd_file;
      // Output reduced filename
      std::string z_file = "z_opmd_%06T"; 
      if ( output_format == "h5" ){
	z_file+=".h5";
      }else if ( output_format == "bp"){
	z_file+=".bp";
      } else {
	// default to hdf5
	z_file+=".h5";	
      }
      m_output_file=z_file;
      
      if (0 == mpi_rank){
	std::cout << "-I- Reducing opmd file: " << opmd_file_full << std::endl;      
	std::cout << "-I- Output compressed opmd file: " << m_output_file << std::endl;            
      }
      // Get every species  
      std::vector<std::string> species_list;
      if ( !species_name.empty() )
	species_list = split(species_name.c_str());
      
      // Get compression parameters
      // Bining default definition for compression
      std::string comp_type;
      size_t  n_part_pcell{0};
      size_t  n_bins[3]  =  {4,4,4};
      size_t  p_bins[3]  =  {-1,-1,-1};
      
      std::vector<std::string> compression_list;
      if ( !compression_name.empty() ){
	compression_list = split(compression_name.c_str());	  
	
	comp_type = compression_list[0];
	n_part_pcell = std::atoi(compression_list[1].c_str());
	
	if (compression_list.size() >= 5){   
	  for (int i=0;i<3;i++){
	    n_bins[i]=std::atoi(compression_list[i+2].c_str());
	  }
	}
	if (compression_list.size() >= 8){   
	  for (int i=0;i<3;i++){
	    p_bins[i]=std::atoi(compression_list[i+5].c_str());
	  }
	}	    
      }
      
      if (0 == mpi_rank){
	std::cout << std::endl;
	std::cout << "-I- Compression method: " << comp_type << " npart/cell: " << n_part_pcell << std::endl;
	for (size_t i=0; i < NDIMS; i++ ) std::cout << " i: " 
					  << i << " n_bins: " << n_bins[i] 
					  << " p_bins: " << p_bins[i] << std::endl;
      }

      // Get particles from input
      m_species="electrons";

      // Load kinematics to internal struct.
      get_kinematics(opmd_file_full.c_str(),m_species.c_str());

      // Perform  Particle  Merging algorithm (Vranic et Al.)
      do_merging(n_part_pcell, n_bins, p_bins);
      

    }//! files to reduce
  
  return 0;      
}


int main(int argc, char *argv[])
{
  int mpi_s{-1};
  int mpi_r{-1};
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_s);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_r);
  
  auto mpi_size = static_cast<uint64_t>(mpi_s);
  auto mpi_rank = static_cast<uint64_t>(mpi_r);
  
  if ( 0 == mpi_rank ) {
    std::cout << std::endl;
    std::cout <<"-I- opmd_reducer:: MPI initialized with size: " << mpi_size << " : " << mpi_rank << std::endl;
    std::cout << std::endl;
  }
  
  Opmd_Reducer reducer;
  reducer.reduce(argc, argv);
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  
  return 0;
}

