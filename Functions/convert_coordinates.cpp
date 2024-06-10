#include <math.h>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>

#include "../constants.hpp"
#include "../functions.hpp"

/*!
 * \brief Apply degree->radian conversion, if necessary. Applied in-place.
 *
 * This is just to provide a clean wrapper to improve some case file readability.
 * It applies a Cartesian check (via constants.hpp), so it is safe to include by default.
 *
 * @param[in,out]   longitude, latitude     Coordinate grids to be converted, if appropriate
 *
 */
void convert_coordinates(
        std::vector<double> & longitude,
        std::vector<double> & latitude
        ) {

    int wRank=-1;
    MPI_Comm_rank( MPI_COMM_WORLD, &wRank );

    // Convert coordinate to radians
    const double D2R = M_PI / 180.;
    size_t ii;
    if (not(constants::CARTESIAN)) {

        #if DEBUG >= 1
        if (wRank == 0) { fprintf(stdout, "Converting grid coordinates (if appropriate).\n"); }
        #endif

        #pragma omp parallel default(none) private(ii) shared(longitude, latitude) firstprivate( D2R )
        { 
            #pragma omp for collapse(1) schedule(static)
            for (ii = 0; ii < longitude.size(); ++ii) { longitude.at(ii) = longitude.at(ii) * D2R; }

            #pragma omp for collapse(1) schedule(static)
            for (ii = 0; ii < latitude.size(); ++ii)  { latitude.at(ii) = latitude.at(ii) * D2R; }
        }
    }

}
