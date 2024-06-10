#include <math.h>
#include <vector>
#include <mpi.h>
#include <cassert>
#include "../netcdf_io.hpp"
#include "../constants.hpp"

void initialize_output_file(
        const dataset & source_data,
        const std::vector<std::string> & vars,
        //const char * filename,
        const std::string filename,
        const double filter_scale,
        const MPI_Comm comm
        ) {

    int wRank=-1, wSize=-1;
    MPI_Comm_rank( MPI_COMM_WORLD, &wRank );
    MPI_Comm_size( MPI_COMM_WORLD, &wSize );

    #if DEBUG>=1
    if (wRank == 0) { fprintf(stdout, "\nPreparing to initialize the output file.\n"); }
    #endif

    // Create some tidy names for variables
    const std::vector<double>   &time       = source_data.time,
                                &depth      = source_data.depth,
                                &latitude   = source_data.latitude,
                                &longitude  = source_data.longitude,
                                &areas      = source_data.areas;

    // Open the NETCDF file
    int FLAG = NC_NETCDF4 | NC_CLOBBER | NC_MPIIO;
    int ncid=0, retval;
    //char buffer [50];
    //snprintf(buffer, 50, filename);
    //retval = nc_create_par(buffer, FLAG, comm, MPI_INFO_NULL, &ncid);
    retval = nc_create_par( filename.c_str(), FLAG, comm, MPI_INFO_NULL, &ncid);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }

    #if DEBUG>=2
    if (wRank == 0) { fprintf(stdout, "    Logging the filter scale\n"); }
    #endif
    if ( filter_scale >= 0 ) {
        retval = nc_put_att_double(ncid, NC_GLOBAL, "filter_scale", NC_DOUBLE, 1, &filter_scale);
        if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
    }

    retval = nc_put_att_double(ncid, NC_GLOBAL, "rho0", NC_DOUBLE, 1, &constants::rho0);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }

    // Record coordinate type
    #if DEBUG>=2
    if (wRank == 0) { fprintf(stdout, "    Logging the grid type\n"); }
    #endif
    if (constants::CARTESIAN) {
        retval = nc_put_att_text(ncid, NC_GLOBAL, "coord-type", 10, "cartesian");
    } else {
        retval = nc_put_att_text(ncid, NC_GLOBAL, "coord-type", 10, "spherical");
    }
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }

    // Extract dimension sizes
    const int Ntime   = time.size();
    const int Ndepth  = depth.size();
    const int Nlat    = latitude.size();
    const int Nlon    = longitude.size();

    // Define the dimensions
    #if DEBUG>=2
    if (wRank == 0) { fprintf(stdout, "    Defining the dimensions\n"); }
    #endif
    int time_dimid, depth_dimid, lat_dimid, lon_dimid;
    retval = nc_def_dim(ncid, "time",      Ntime,     &time_dimid);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
    retval = nc_def_dim(ncid, "depth",     Ndepth,    &depth_dimid);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
    
    if ( constants::GRID_TYPE == constants::GridType::MeshGrid ) {
        retval = nc_def_dim(ncid, "latitude",  Nlat,      &lat_dimid);
        if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
        retval = nc_def_dim(ncid, "longitude", Nlon,      &lon_dimid);
        if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
    } else if ( constants::GRID_TYPE == constants::GridType::LLC ) {
        retval = nc_def_dim(ncid, "latlon",  Nlat,      &lat_dimid);
        if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
        lon_dimid = lat_dimid;
    }

    // Define coordinate variables
    #if DEBUG>=2
    if (wRank == 0) { fprintf(stdout, "    Defining the dimension variables\n"); }
    #endif
    int time_varid, depth_varid, lat_varid, lon_varid;
    retval = nc_def_var(ncid, "time",      NC_DOUBLE, 1, &time_dimid,  &time_varid);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
    retval = nc_def_var(ncid, "depth",     NC_DOUBLE, 1, &depth_dimid, &depth_varid);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
    retval = nc_def_var(ncid, "latitude",  NC_DOUBLE, 1, &lat_dimid,   &lat_varid);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
    retval = nc_def_var(ncid, "longitude", NC_DOUBLE, 1, &lon_dimid,   &lon_varid);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }

    if (not(constants::CARTESIAN)) {
        #if DEBUG>=2
        if (wRank == 0) { fprintf(stdout, "    Add scale factors for Rad to Degrees\n"); }
        #endif
        const double rad_to_degree = 180. / M_PI;
        retval = nc_put_att_double(ncid, lon_varid, "scale_factor", 
                NC_DOUBLE, 1, &rad_to_degree);
        if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
        retval = nc_put_att_double(ncid, lat_varid, "scale_factor", 
                NC_DOUBLE, 1, &rad_to_degree);
        if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
    }

    // Write the coordinate variables
    #if DEBUG>=2
    if (wRank == 0) { fprintf(stdout, "    Write the dimensions\n"); }
    #endif
    size_t start[1], count[1];
    start[0] = 0;
    count[0] = Ntime;
    retval = nc_put_vara_double(ncid, time_varid,  start, count, &time[0]);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }

    count[0] = Ndepth;
    retval = nc_put_vara_double(ncid, depth_varid, start, count, &depth[0]);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }

    count[0] = Nlat;
    retval = nc_put_vara_double(ncid, lat_varid,   start, count, &latitude[0]);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }

    count[0] = Nlon;
    retval = nc_put_vara_double(ncid, lon_varid,   start, count, &longitude[0]);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }

    // Write the cell areas for convenience
    if ( constants::GRID_TYPE == constants::GridType::MeshGrid ) {
        #if DEBUG>=2
        if (wRank == 0) { fprintf(stdout, "    Write the cell areas\n"); }
        #endif
        int area_dimids[2];
        area_dimids[0] = lat_dimid;
        area_dimids[1] = lon_dimid;
        int area_varid;
        retval = nc_def_var(ncid, "cell_areas", NC_DOUBLE, 2, area_dimids, &area_varid);
        if (retval) { NC_ERR(retval, __LINE__, __FILE__); }

        size_t area_start[2], area_count[2];
        area_start[0] = 0;
        area_start[1] = 0;
        area_count[0] = Nlat;
        area_count[1] = Nlon;
        retval = nc_put_vara_double(ncid, area_varid, area_start, area_count, &areas[0]);
        if (retval) { NC_ERR(retval, __LINE__, __FILE__); }
    }

    // Close the file
    retval = nc_close(ncid);
    if (retval) { NC_ERR(retval, __LINE__, __FILE__); }

    #if DEBUG >= 2
    if (wRank == 0) { fprintf(stdout, "\nOutput file (%s) initialized.\n", filename.c_str() ); }
    #endif

    if (wRank == 0) {
        #if DEBUG>=2
        if (wRank == 0) { fprintf(stdout, "    Root rank will now add each variable.\n"); }
        #endif
        // Loop through and add the desired variables
        // Dimension names (in order!)
        const char* dim_names_MeshGrid[] = {"time", "depth", "latitude", "longitude"};
        const char* dim_names_LLC[]      = {"time", "depth", "latlon"};
        const int ndims_MeshGrid = 4;
        const int ndims_LLC      = 3;
        for (size_t varInd = 0; varInd < vars.size(); ++varInd) {
            if ( constants::GRID_TYPE == constants::GridType::MeshGrid ) {
                add_var_to_file( vars.at(varInd), dim_names_MeshGrid, ndims_MeshGrid, filename );
            } else if ( constants::GRID_TYPE == constants::GridType::LLC ) {
                add_var_to_file( vars.at(varInd), dim_names_LLC, ndims_LLC, filename );
            }
        }
    }

    // Add some global attributes from constants.hpp
    #if DEBUG>=2
    if (wRank == 0) { fprintf(stdout, "    Now add a series of computational details\n"); }
    #endif
    add_attr_to_file("R_earth",                                      constants::R_earth,    filename);
    add_attr_to_file("rho0",                                         constants::rho0,       filename);
    add_attr_to_file("g",                                            constants::g,          filename);
    add_attr_to_file("differentiation_convergence_order",   (double) constants::DiffOrd,    filename);
    add_attr_to_file("KERNEL_OPT",                          (double) constants::KERNEL_OPT, filename);
    if (constants::COMP_BC_TRANSFERS) {
        add_attr_to_file("KernPad",                             (double) constants::KernPad,    filename);
    }

    #if DEBUG >= 2
    if (wRank == 0) { fprintf(stdout, "\n"); }
    #endif
}
