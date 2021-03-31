#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <vector>

#include "../constants.hpp"
#include "../functions.hpp"
#include "../postprocess.hpp"

void compute_region_avg_and_std(
        std::vector< std::vector< double > > & field_averages,
        std::vector< std::vector< double > > & field_std_devs,
        const dataset & source_data,
        const std::vector<const std::vector<double>*> & postprocess_fields
        ) {

    const int   Ntime   = source_data.Ntime,
                Ndepth  = source_data.Ndepth,
                Nlat    = source_data.Nlat,
                Nlon    = source_data.Nlon;

    const int   num_regions   = source_data.region_names.size(),
                num_fields    = postprocess_fields.size();

    const int chunk_size = get_omp_chunksize(Nlat, Nlon);

    double int_val, reg_area, dA;

    int Ifield, Iregion, Itime, Idepth, Ilat, Ilon;
    size_t int_index, area_index, index;

    for (Ifield = 0; Ifield < num_fields; ++Ifield) {
        for (Iregion = 0; Iregion < num_regions; ++Iregion) {
            for (Itime = 0; Itime < Ntime; ++Itime) {
                for (Idepth = 0; Idepth < Ndepth; ++Idepth) {

                    int_val = 0.;

                    #pragma omp parallel default(none)\
                    private(Ilat, Ilon, index, dA, area_index )\
                    shared( source_data, Ifield, Iregion, Itime, Idepth, int_index, postprocess_fields) \
                    reduction(+ : int_val)
                    { 
                        #pragma omp for collapse(2) schedule(guided, chunk_size)
                        for (Ilat = 0; Ilat < Nlat; ++Ilat) {
                            for (Ilon = 0; Ilon < Nlon; ++Ilon) {

                                index = Index(Itime, Idepth, Ilat, Ilon,
                                              Ntime, Ndepth, Nlat, Nlon);

                                if ( source_data.mask.at(index) ) { // Skip land areas

                                    area_index = Index(0, 0, Ilat, Ilon,
                                                       1, 1, Nlat, Nlon);

                                    dA = source_data.areas.at(area_index);

                                    if ( source_data.regions.at( source_data.region_names.at(Iregion) ).at(area_index) )
                                    {
                                        int_val += postprocess_fields.at(Ifield)->at(index) * dA;
                                    }
                                }
                            }
                        }
                    }
                    int_index = Index(0, Itime, Idepth, Iregion,
                                      1, Ntime, Ndepth, num_regions);
                    reg_area = source_data.region_areas.at(int_index);
                    field_averages.at(Ifield).at(int_index) = (reg_area == 0) ? 0. : int_val / reg_area;
                }
            }
        }
    }

    // Now that we have region averages, get region standard deviations
    for (Ifield = 0; Ifield < num_fields; ++Ifield) {
        for (Iregion = 0; Iregion < num_regions; ++Iregion) {
            for (Itime = 0; Itime < Ntime; ++Itime) {
                for (Idepth = 0; Idepth < Ndepth; ++Idepth) {

                    int_index = Index(0, Itime, Idepth, Iregion,
                                      1, Ntime, Ndepth, num_regions);
                    reg_area = source_data.region_areas.at(int_index);

                    int_val = 0.;

                    #pragma omp parallel default(none)\
                    private(Ilat, Ilon, index, dA, area_index )\
                    shared( source_data, Ifield, Iregion, Itime, Idepth, int_index,\
                            postprocess_fields, field_averages) \
                    reduction(+ : int_val)
                    { 
                        #pragma omp for collapse(2) schedule(guided, chunk_size)
                        for (Ilat = 0; Ilat < Nlat; ++Ilat) {
                            for (Ilon = 0; Ilon < Nlon; ++Ilon) {

                                index = Index(Itime, Idepth, Ilat, Ilon,
                                              Ntime, Ndepth, Nlat, Nlon);

                                if ( source_data.mask.at(index) ) { // Skip land areas

                                    area_index = Index(0, 0, Ilat, Ilon,
                                                       1, 1, Nlat, Nlon);

                                    dA = source_data.areas.at(area_index);

                                    if ( source_data.regions.at( source_data.region_names.at(Iregion) ).at(area_index) )
                                    {
                                        int_val +=
                                            pow(   field_averages.at(Ifield).at(int_index)
                                                 - postprocess_fields.at(Ifield)->at(index),
                                                2.) * dA;
                                    }
                                }
                            }
                        }
                    }
                    field_std_devs.at(Ifield).at(int_index) = (reg_area == 0) ? 0. : sqrt( int_val / reg_area );
                }
            }
        }
    }
}
