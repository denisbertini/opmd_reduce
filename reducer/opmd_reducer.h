#pragma once

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <cstdlib>
#include <memory>
#include <numeric>

// Filesystem
#include <filesystem>
#include <sys/stat.h>
#include <algorithm>

// MPI
#include "mpi.h"
// Options
#include "cxxopts.h"
// Utils
#include "utils.h"
// openPMD
#include <openPMD/openPMD.hpp>

#include "vranic_merger.h"


using namespace openPMD;
using namespace std;

namespace fs = std::filesystem;

namespace si_units{
  enum si_units_t
    {
      length = 0, // L
      mass = 1, // M
      time = 2, // T
      electricCurrent = 3, // I
      thermodynamicTemperature = 4, // theta
      amountOfSubstance = 5, // N
      luminousIntensity = 6, // J
    };
}

inline
std::vector<std::string> split(const char *str, char c = ':')
{
  std::vector<std::string> result;
  do
    {
      const char *begin = str;
      while(*str != c && *str)
	str++;
      result.push_back(std::string(begin, str));
    } while (0 != *str++);
  
  return result;
}


class Opmd_Reducer{

  // Output file
  std::string m_output_file;

  // Iteration
  int m_istep{0};

  // Species
  std::string m_species;
  
  // Particle kinematics
  part_kine m_part_kine;

  // Store attributes
  std::array<std::string, NDIMS> const m_part_attr{{"position", "momentum", "weighting"}};
  std::array<std::string, NDIMS> const m_dimensions{{"x", "y", "z"}};


  public:
  void get_fullinfo(const std::string& fname);
  void get_kinematics(const std::string& fname, const std::string& species_name);
  void do_merging(const int n_part_cell, const size_t n_bins[], const size_t p_bins[]);
  int reduce(int argc, char *argv[]);  
  
  inline std::map< openPMD::UnitDimension, double >
  getOpmdUnits ( std::string const & record_name )
  {
    if( (record_name == "position") || (record_name == "positionOffset") ) return {
	{openPMD::UnitDimension::L,  1.}
      };
    else if( record_name == "momentum" ) return {
	{openPMD::UnitDimension::L,  1.},
	{openPMD::UnitDimension::M,  1.},
	{openPMD::UnitDimension::T, -1.}
      };
    else if( record_name == "charge" ) return {
	{openPMD::UnitDimension::T,  1.},
	{openPMD::UnitDimension::I,  1.}
      };
    else if( record_name == "mass" ) return {
	{openPMD::UnitDimension::M,  1.}
      };
    else if( record_name == "E" ) return {
	{openPMD::UnitDimension::L,  1.},
	{openPMD::UnitDimension::M,  1.},
	{openPMD::UnitDimension::T, -3.},
	{openPMD::UnitDimension::I, -1.},
      };
    else if( record_name == "B" ) return {
	{openPMD::UnitDimension::M,  1.},
	{openPMD::UnitDimension::I, -1.},
	{openPMD::UnitDimension::T, -2.}
      };	  
    else if( record_name == "J" ) return {
	{openPMD::UnitDimension::L, -2},
	{openPMD::UnitDimension::I,  1}
      };
    else if( record_name == "rho" ) return {
	{openPMD::UnitDimension::L, -3},
	{openPMD::UnitDimension::I,  1},
	{openPMD::UnitDimension::T,  1},
      };	    
    
    else return {}; 
  }
  
}; // !class Opmd_Reducer

