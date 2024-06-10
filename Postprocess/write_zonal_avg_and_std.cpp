#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <vector>

#include "../constants.hpp"
#include "../functions.hpp"
#include "../postprocess.hpp"
#include "../netcdf_io.hpp"

void write_zonal_avg_and_std(
        const std::vector< std::vector< double > > & zonal_averages,
        const std::vector< std::vector< double > > & zonal_std_devs,
        const std::vector< std::vector< double > > & zonal_medians,
        const std::vector<std::string> & vars_to_process,
        //const char * filename,
        const std::string & filename,
        const int Stime,
        const int Sdepth,
        const int Ntime,
        const int Ndepth,
        const int Nlat,
        const int num_fields
        ){

    // Dimension order: time - depth - latitude
    size_t start[3], count[3];
    start[0] = Stime;
    count[0] = Ntime;

    start[1] = Sdepth;
    count[1] = Ndepth;

    start[2] = 0;
    count[2] = Nlat;

    for ( int Ifield = 0; Ifield < num_fields; ++Ifield ) {
        write_field_to_output( 
                zonal_averages.at(Ifield), vars_to_process.at(Ifield) + "_zonal_average", 
                start, count, filename );
        write_field_to_output( 
                zonal_medians.at(Ifield), vars_to_process.at(Ifield) + "_zonal_median", 
                start, count, filename );
        // To turn standard deviation outputs back on, also need to turn back on the calculations in compute_region_avg_and_std
    }
}
