#pragma once
#include <cassert>
#include <omp.h>
#include <mpi.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <sstream>
#include <vector>
#include <chrono>


using std::cout;

    //Define some Physical constants
    constexpr double C_LIGHT=299792458.; // [m/s] 
    constexpr int  NDIMS=3; // 3 Dimensions 

    struct part{      
      int   len; 
      double charge, mass;
      double *x, *y, *z, *w,  *px, *py, *pz, *gm;
    };

    struct part_kine{
      size_t   len; 
      double charge, mass;
      std::vector<double> x[3];
      std::vector<double> p[3];
      std::vector<double> w;      
    };

         
    static inline bool sort_using_less_than(double u, double v) 
    {
      return u < v;
    }


    static inline double  vmax(const std::vector<double> &A) 
    {
      return  *std::max_element(A.begin(),A.end());
    }

    static inline double  vmin(const std::vector<double> &A) 
    {
      return  *std::min_element(A.begin(),A.end());
    }

    static inline double  max(const double* A, int len) 
    {
      int max_val = A[0];

      #pragma omp parallel for reduction(max:max_val) 
      for (int idx = 0; idx < len; idx++){
	max_val = max_val > A[idx] ? max_val : A[idx];
      }
      
      return max_val;
    }

    static inline double  min(double A[], int len) 
    {
      int min_val = A[0];  

      #pragma omp parallel for reduction(min:min_val) 
      for (int idx = 0; idx < len; idx++)
	min_val = min_val < A[idx] ? min_val : A[idx];

      return min_val;
    }

    // Value is within the range [low, high).
    template <typename T>      
      bool is_inside(const T& value, const T& low, const T& high) {
      if ((high==0) && (low==0)) return true;
      if (high==low) return true;
      return !(value < low) && (value < high);
    }


    double re_binning(const std::vector<double> &p){
      // Re-binning using Freedman-Diaconis' rule.
      size_t n = p.size();
      double prange = *std::max_element(p.begin(), p.end()) -
	*std::min_element(p.begin(), p.end());
      double bin_w = 1.0;
      if (n>1) {
	size_t q1  = static_cast<size_t>(n * 0.25);
	size_t q3 = n - static_cast<size_t>(n * 0.25);
	auto p_c = p;
	std::nth_element(p_c.begin(), p_c.begin() + q1, p_c.end());
	std::nth_element(p_c.begin(), p_c.begin() + q3, p_c.end());
	double iq_r = p_c[q3] - p_c[q1];
	double iq = std::max(iq_r, prange / 10.);
	bin_w = 2 * iq * pow(n, -1. / 3.);
      }
      return bin_w;
    }
  


/** The Memory profiler class for profiling purpose
 *
 *  Simple Memory usage report that works on linux system
 */

static std::chrono::time_point<std::chrono::system_clock> m_ProgStart =
    std::chrono::system_clock::now();

class MemoryProfiler
{
public:
    /** Simple Memory profiler for linux
     *
     * @param[in] rank     MPI rank
     * @param[in] tag      item name to measure
     */
    MemoryProfiler(int rank, const std::string &tag)
    {
        m_Rank = rank;
#if defined(__linux)
        // m_Name = "/proc/meminfo";
        m_Name = "/proc/self/status";
        Display(tag);
#else
        (void)tag;
        m_Name = "";
#endif
    }

    /**
     *
     * Read from /proc/self/status and display the Virtual Memory info at rank 0
     * on console
     *
     * A few worthy points can be noted here:
     *
     *   VmRSS in /proc/[pid]/statm is a useful data.
     *   =====
     *   It shows how much memory in RAM is occupied by the process.
     *   The rest extra memory has either been not used or has been swapped out.
     * 
     *   VmSize is how much virtual memory the process has in total.
     *   =====
     *	 This includes all types of memory, both in RAM and swapped out.
     *   These numbers can get skewed because they also include shared libraries.
     * 
     *
     * @param tag      item name to measure
     * @param rank     MPI rank
     */

    void Display(const std::string &tag)
    {
        if (0 == m_Name.size())
            return;

        if (m_Rank > 0)
            return;

        std::cout << "[Memory used: " << tag << "]" << std::endl;
        std::ifstream input(m_Name.c_str());

        if (input.is_open())
        {
            for (std::string line; getline(input, line);)
            {
                if (line.find("VmRSS") == 0)
                    std::cout << line << " ";
                if (line.find("VmSize") == 0)
                    std::cout << line << " ";
                if (line.find("VmSwap") == 0)
                    std::cout << line;
            }
            std::cout << std::endl;
            input.close();
        }
    }

private:
    int m_Rank;
    std::string m_Name;
};

/** The Timer class for profiling purpose
 *
 *  Simple Timer that measures time consumption btw constucture and destructor
 *  Reports at rank 0 at the console, for immediate convenience
 */
class Timer
{
public:
    /**
     *
     * Simple Timer
     *
     * @param tag      item name to measure
     * @param rank     MPI rank
     */
    Timer(const std::string &tag, int rank)
    {
        m_Tag = tag;
        m_Rank = rank;
        m_Start = std::chrono::system_clock::now();
        // MemoryProfiler (rank, tag);
    }
    ~Timer()
    {
        MPI_Barrier(MPI_COMM_WORLD);
        std::string tt = m_Tag;	
	if ( 0 == m_Rank) {
	  std::cout << std::endl;
	  std::cout <<"-I- Job statistics: " << std::endl;
	}
        MemoryProfiler (m_Rank, tt.c_str());
        m_End = std::chrono::system_clock::now();

        double millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                            m_End - m_Start)
                            .count();
        double secs = millis / 1000.0;
        if (m_Rank > 0)
            return;
	std::cout << std::endl;   
        std::cout << "[" << m_Tag << "] took:" << secs << " seconds.\n";
        std::cout << "   \t From Program Start in seconds "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         m_End - m_ProgStart)
                         .count() /
                1000.0
                  << std::endl;

        std::cout << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_Start;
    std::chrono::time_point<std::chrono::system_clock> m_End;

    std::string m_Tag;
    int m_Rank = 0;
};
