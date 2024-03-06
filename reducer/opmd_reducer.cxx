#include <openPMD/openPMD.hpp>

#include <stdio.h> 
#include <unistd.h> 
#include <iostream>
#include <memory>
#include <cstddef>
#include <string>


#include "opmd_reducer.h"


using std::cout;
using namespace openPMD;



void Opmd_Reducer::getfullInfo(const std::string& fname){

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


void Opmd_Reducer::get_kinematics(const std::string& filename, int rank){

    try{

      // Open series
      std::string tag = "Processing: " + filename;    
      Timer kk(tag, rank);
      Series series = Series(filename, Access::READ_ONLY, MPI_COMM_WORLD);
      
      //Check one iteration in the file
      int n_iter = series.iterations.size();
      if (n_iter != 1) throw std::runtime_error("Only one iteration/file allowed !"); 

      std::cout << "Read iterations ";
      int i_step=0;
      for( auto const& val : series.iterations ){
	i_step=val.first;
        std::cout << '\t' << val.first;
      }

      Iteration i = series.iterations[i_step];
      
      if (0 == rank)
	{
	  std::cout << "  " << series.iterationEncoding() << std::endl;
	  std::cout << " Scanning  file: " <<  filename << std::endl;
	  std::cout << " found  n_iter: " << n_iter << std::endl;
	  cout << "Iteration 0 contains " << i.meshes.size() << " meshes:";
	  for (auto const &m : i.meshes)
	    cout << "\n\t" << m.first;
	  cout << '\n';
	  cout << "Iteration 0 contains " << i.particles.size()
	       << " particle species:";
	  for (auto const &ps : i.particles)
	    {
	      cout << "\n\t" << ps.first;
	      for (auto const &r : ps.second)
		{
		  cout << "\n\t" << r.first;
		  cout << '\n';
		}
	    }
	  std::cout << std::endl;
	}
    }//! try

    catch (std::exception &ex)
      {
	if (0 == rank)
	  {
	    std::cerr << ex.what() << std::endl;
	  }
      }
    
}

  
int Opmd_Reducer::reduce(int argc, char *argv[]){
  
  int mpi_rank, mpi_size;
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
    std::cout << "opmd_reducer: directory to process: " << opmd_dir.c_str() << std::endl;
    std::cout << "opmd_reducer: files to process: " << opmd_files.c_str() << std::endl;
    std::cout << "opmd_reducer: species name list: " << species_name.c_str() << std::endl;
    std::cout << "opmd_reducer: compression: " << compression_name.c_str() << std::endl;	  
    std::cout << "opmd_reducer: output format: " << output_format.c_str() << std::endl;
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
      // Output filename
      std::string reduced_opmd_file="/reduced_"+opmd_file;
	
      if (0 == mpi_rank){
	std::cout << "Reducing opmd file: " << opmd_file_full << std::endl;      
	std::cout << "OUtput compressed opmd file: " << reduced_opmd_file << std::endl;            
      }
      // Get every species  
      std::vector<std::string> species_list;
      if ( !species_name.empty() )
	species_list = split(species_name.c_str());
      
      // Get compression parameters
      // Bining default definition for compression
      std::string comp_type;
      int  n_part_pcell{0};
      int  n_bins[3]  =  {4,4,4};
      int  p_bins[3]  =  {-1,-1,-1};
      
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
	std::cout << " compression method: " << comp_type << " npart/cell: " << n_part_pcell << std::endl;
	for (int i=0;i<3; i++ ) std::cout << " i: " 
					  << i << " n_bins: " << n_bins[i] 
					  << " p_bins: " << p_bins[i] << std::endl;
      }

      // Get Info from input
      get_kinematics(opmd_file_full.c_str(), mpi_rank);

      
      
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
    std::cout <<"Opmd_Reducer:: MPI initialized with size: " << mpi_size << " : " << mpi_rank << std::endl;
    std::cout << std::endl;
  }
  
  Opmd_Reducer reducer;
  reducer.reduce(argc, argv);
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  
  return 0;
}















/*

int main(int argc, char *argv[])
{

    int opt;
    char* fname=NULL;
      
    while((opt = getopt(argc, argv, ":f:x")) != -1) 
    { 
        switch(opt) 
        { 
            case 'x': 
            case 'f': 
                printf("filename: %s\n", optarg);
		fname=optarg;
                break; 
        } 
    }   


    // Initiate the reading procedure

    getfullInfo(fname);
     

    std::cout << " Now create a serie " << std::endl;
    
    // Create a series and open the file
    Series series = Series( fname, Access::READ_ONLY);
    
    cout << "Read a Series with openPMD standard version "
         << series.openPMD() << '\n';
    cout << "The Series contains " << series.iterations.size() << " iterations:"; 

    // Loop over all iterations in the file
    int iter=0;
    for( auto const& i : series.iterations ){
      // Meshes  
      cout << "Iteration " << iter << " contains " << i.second.meshes.size() << " meshes:";
      for( auto const& m : i.second.meshes )
        cout << "\n\t" << m.first;
            
      cout << '\n';
      cout << "Iteration: "  <<  iter << "contains " << i.second.particles.size() << " particle species:";

      // Loop over species
      for( auto const& ps : i.second.particles ) {
        cout << "\n\t" << ps.first;
      }
      cout << '\n';	
      
      iter++;
      Iteration j = i.second;
    
      // Particles Species      
      openPMD::ParticleSpecies deuterons = j.particles["electron_l"];
      //std::shared_ptr<double> charge = electrons["charge"][openPMD::RecordComponent::SCALAR].loadChunk<double>();
      series.flush();
      
      // Access attribute within sub-group electron_gridx
      for( auto const& a : deuterons["momentum"].attributes() ){
	std::cout << '\t' << a << '\n';	
      }
      std::cout << '\n';
      
      auto p_x = deuterons["momentum"]["x"];
      Extent g_extent = p_x.getExtent();
      auto all_x  = p_x.loadChunk<double>();
      series.flush();

      cout << "Full Electron gridx starts with:\n\t{";
      for( size_t col = 0;  col < 5; ++col )
	cout << all_x.get()[col] << ", ";

      cout << "...}\n";
           
    }//!iteration++ 
    
    return 0;
}


*/


