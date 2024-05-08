#include "../constants.hpp"
#include "../functions.hpp"
#include "../netcdf_io.hpp"
#include "../preprocess.hpp"
#include "../differentiation_tools.hpp"
#include <algorithm>
#include <vector>
#include <omp.h>
#include <math.h>

/*
#include "../amgcl/amgcl/io/ios_saver.hpp"
#include "../amgcl/amgcl/util.hpp"
#include "../amgcl/amgcl/backend/builtin.hpp"
#include "../amgcl/amgcl/make_solver.hpp"
#include "../amgcl/amgcl/solver/bicgstab.hpp"
#include "../amgcl/amgcl/amg.hpp"
#include "../amgcl/amgcl/coarsening/smoothed_aggregation.hpp"
#include "../amgcl/amgcl/relaxation/spai0.hpp"
#include "../amgcl/amgcl/adapter/crs_tuple.hpp"
*/
#include "amgcl/io/ios_saver.hpp"
#include "amgcl/util.hpp"
#include "amgcl/backend/builtin.hpp"
#include "amgcl/make_solver.hpp"
#include "amgcl/solver/bicgstab.hpp"
#include "amgcl/amg.hpp"
#include "amgcl/coarsening/smoothed_aggregation.hpp"
#include "amgcl/relaxation/spai0.hpp"
#include "amgcl/adapter/crs_tuple.hpp"


void Apply_LLC_Helmholtz_Projection_AMGCL(
        const std::string output_fname,
        dataset & source_data,
        const std::vector<double> & seed_tor,
        const std::vector<double> & seed_pot,
        const bool single_seed,
        const double rel_tol,
        const int max_iters,
        const bool weight_err,
        const bool use_mask,
        const double Tikhov_Laplace,
        const MPI_Comm comm
        ) {

    int wRank, wSize;
    MPI_Comm_rank( comm, &wRank );
    MPI_Comm_size( comm, &wSize );

    // Create some tidy names for variables
    const std::vector<double>   &latitude   = source_data.latitude,
                                &longitude  = source_data.longitude,
                                &dAreas     = source_data.areas;

    const std::vector<bool> &mask = (constants::FILTER_OVER_LAND) ? source_data.reference_mask : source_data.mask;

    const std::vector<int>  &myCounts = source_data.myCounts,
                            &myStarts = source_data.myStarts;

    std::vector<double>   &u_lat = source_data.variables.at("u_lat"),
                          &u_lon = source_data.variables.at("u_lon");

    // Create a 'no mask' mask variable
    //   we'll treat land values as zero velocity
    //   We do this because including land seems
    //   to introduce strong numerical issues
    const std::vector<bool> unmask(mask.size(), true);

    const int   Ntime   = myCounts.at(0),
                Ndepth  = myCounts.at(1);

    const size_t Npts = latitude.size();

    int Itime=0, Idepth=0;
    size_t index, index_sub, iters_used = 0;

    // Fill in the land areas with zero velocity
    #pragma omp parallel default(none) shared( u_lon, u_lat, mask, stderr, wRank ) private( index )
    {
        #pragma omp for collapse(1) schedule(guided)
        for (index = 0; index < u_lon.size(); index++) {
            if (not(mask.at(index))) {
                u_lon.at(index) = 0.;
                u_lat.at(index) = 0.;
            } else if (    ( std::fabs( u_lon.at(index) ) > 30000.) 
                        or ( std::fabs( u_lat.at(index) ) > 30000.) 
                      ) {
                fprintf( stderr, "  Rank %d found a bad vel point at index %'zu! Setting to zero.\n", wRank, index );
                u_lon.at(index) = 0.;
                u_lat.at(index) = 0.;
            }
        }
    }

    size_t num_land_points = 0;
    if (constants::FILTER_OVER_LAND) {
        for (index = 0; index < u_lon.size(); index++) {
            if (not(mask.at(index))) { num_land_points++; }
        }
    }
    #if DEBUG>=0
    if (wRank == 0) { fprintf(stdout, " Identified %'zu land points.\n", num_land_points); }
    #endif

    // Storage vectors
    std::vector<double> 
        full_Psi(        u_lon.size(), 0. ),
        full_Phi(        u_lon.size(), 0. ),
        full_u_lon_tor(  u_lon.size(), 0. ),
        full_u_lat_tor(  u_lon.size(), 0. ),
        full_u_lon_pot(  u_lon.size(), 0. ),
        full_u_lat_pot(  u_lon.size(), 0. ),
        u_lon_tor_seed(  Npts, 0. ),
        u_lat_tor_seed(  Npts, 0. ),
        u_lon_pot_seed(  Npts, 0. ),
        u_lat_pot_seed(  Npts, 0. );

    // alglib variables
    const size_t Nboxrows = ( Tikhov_Laplace > 0 ) ? 4 : 2;
    num_land_points = 0; // need to have a square matrix
    std::vector<double> 
        RHS_vector( 2 * Npts, 0. ),
        Psi_seed(       Npts, 0. ),
        Phi_seed(       Npts, 0. ),
        work_arr(       Npts, 0. ),
        div_term(       Npts, 0. ),
        vort_term(      Npts, 0. ),
        u_lon_rem(      Npts, 0. ),
        u_lat_rem(      Npts, 0. );
    

    fprintf( stdout, "Copy seed\n" );
    // Copy the starting seed.
    if (single_seed) {
        #pragma omp parallel \
        default(none) \
        shared(Psi_seed, Phi_seed, seed_tor, seed_pot) \
        private( index ) \
        firstprivate( Npts )
        {
            #pragma omp for collapse(1) schedule(static)
            for (index = 0; index < Npts; ++index) {
                Psi_seed.at(index) = seed_tor.at(index);
                Phi_seed.at(index) = seed_pot.at(index);
            }
        }
    }

    const size_t num_neighbours = source_data.num_neighbours;
    // Get a magnitude for the derivatives, to help normalize the rows of the 
    //  Laplace entries to have similar magnitude to the others.
    long double deriv_ref_1 = 0, deriv_ref_2 = 0;
    size_t Ineighbour;
    #pragma omp parallel default(none) \
    private( Ineighbour, index ) \
    shared( source_data ) \
    firstprivate( num_neighbours, Npts ) \
    reduction( +:deriv_ref_1,deriv_ref_2 )
    {
        #pragma omp for collapse(1) schedule(static)
        for (index = 0; index < Npts; ++index) {
            for ( Ineighbour = 0; Ineighbour < num_neighbours + 1; Ineighbour++ ) {
                deriv_ref_1 += std::fabs( source_data.adjacency_ddlat_weights.at(index).at(Ineighbour) ) / (Npts*num_neighbours);
                //deriv_ref_2 += std::fabs( source_data.adjacency_d2dlat2_weights.at(index).at(Ineighbour) ) / (Npts*num_neighbours);
            }
        }
    }
    //const double deriv_scale_factor = deriv_ref_2 / deriv_ref_1;
    const double deriv_scale_factor = deriv_ref_1;
    fprintf( stdout, "deriv-scale-factor: %g\n", deriv_scale_factor );

    //
    //// Build the LHS part of the problem
    //      Ordering is: [  u_from_psi      u_from_phi   ] *  [ psi ]   =    [  u   ]
    //                   [  v_from_psi      v_from_phi   ]    [ phi ]        [  v   ]
    //
    //      Ordering is: [           - ddlat   sec(phi) * ddlon   ] *  [ psi ]   =    [     u     ]
    //                   [  sec(phi) * ddlon              ddlat   ]    [ phi ]        [     v     ]
    //                   [           Laplace                  0   ]                   [ vort(u,v) ]
    //                   [                 0            Laplace   ]                   [  div(u,v) ]
    
    #if DEBUG >= 1
    if (wRank == 0) {
        fprintf(stdout, "Building the LHS of the least squares problem.\n");
        fflush(stdout);
    }
    #endif

    double val;
    size_t column_skip, row_skip, land_counter = 0;

    std::vector<size_t> vals_upto_row( 2*Npts+1, 0 ), 
                        col_index( 2*Npts*(num_neighbours+1) );
    std::vector<double> LHS( 2*Npts*(num_neighbours+1), 0. );

    for ( size_t Ipt = 0; Ipt < Npts; Ipt++ ) {
        // Record how many values will occur before next row
        vals_per_row[Ipt+1] = (Ipt+1) * (num_neighbours + 1) * 2;

        double  weight_val = weight_err ? dAreas.at(Ipt) : 1.,
                cos_lat_inv = 1. / cos(latitude.at(Ipt)),
                R_inv = 1. / constants::R_earth;

        for ( size_t Ineighbour = 0; Ineighbour < num_neighbours + 1; Ineighbour++ ) {

            size_t neighbour_ind = (Ineighbour < num_neighbours) ? 
                                    source_data.adjacency_indices.at(Ipt).at(Ineighbour) :
                                    Ipt;

            size_t col_index_Psi = Ipt * (2*num_neighbours) + neighbour_ind;    // logical index for Psi
            size_t col_index_Phi = col_index_Psi + num_neighbours;              // logical index for Phi

            col_index[col_index_Psi] = neighbour_ind;           // the column index for the Psi coeff
            col_index[col_index_Phi] = neighbour_ind + Npts;    // the column index for the Phi coeff

            bool is_pole = std::fabs( std::fabs( latitude.at(Ipt) * 180.0 / M_PI ) - 90 ) < 1e-6;
            if ( is_pole ) { fprintf(stdout, "SKIPPING POLE POINT!\n"); continue; }

            //
            //// Second LON derivative
            //
            val  = source_data.adjacency_d2dlon2_weights.at(Ipt).at(Ineighbour);
            val *= weight_val * pow(cos_lat_inv * R_inv, 2.);

            LHS[col_index_Psi] += val;
            LHS[col_index_Phi] += val;


            //
            //// Second LAT derivative
            //
            val  = source_data.adjacency_d2dlat2_weights.at(Ipt).at(Ineighbour);
            val *= weight_val * pow(R_inv, 2.);

            LHS[col_index_Psi] += val;
            LHS[col_index_Phi] += val;

            //
            //// LAT first derivative
            //
            val  = source_data.adjacency_ddlat_weights.at(Ipt).at(Ineighbour);
            val *= weight_val * pow(R_inv, 2.) * tan( latitude.at(Ipt) );

            LHS[col_index_Psi] += -val;
            LHS[col_index_Phi] += -val;

        }
    }


    //
    //// Set up the backend
    //
    #if DEBUG >= 1
    if (wRank == 0) {
        fprintf(stdout, "Declaring the Backend\n");
        fflush(stdout);
    }
    #endif

    typedef amgcl::backend::builtin<double> Backend; // this should use OpenMP


    //
    //// Building the solver
    //
    #if DEBUG >= 1
    if (wRank == 0) {
        fprintf(stdout, "Building the solver\n");
        fflush(stdout);
    }
    #endif
    typedef amgcl::make_solver<
        // Use AMG as preconditioner:
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        // And BiCGStab as iterative solver:
        amgcl::solver::bicgstab<Backend>
    > Solver;

    Solver solve( std::tie(2*Npts, vals_upto_row, col_index, LHS) );


    // Build the RHS
    for (int Itime = 0; Itime < Ntime; ++Itime) {
        for (int Idepth = 0; Idepth < Ndepth; ++Idepth) {

            if (not(single_seed)) {
                #if DEBUG >= 2
                fprintf( stdout, "Extracting seed.\n" );
                fflush(stdout);
                #endif
                // If single_seed == false, then we were provided seed values, pull out the appropriate values here
                #pragma omp parallel \
                default(none) \
                shared( Psi_seed, Phi_seed, seed_tor, seed_pot, Itime, Idepth, stdout ) \
                private( index, index_sub ) \
                firstprivate( Ntime, Ndepth, Npts )
                {
                    #pragma omp for collapse(1) schedule(static)
                    for (index = 0; index < Npts; ++index) {
                        Psi_seed.at(index) = seed_tor.at(index + Npts*(Itime*Ndepth + Idepth));
                        Phi_seed.at(index) = seed_pot.at(index + Npts*(Itime*Ndepth + Idepth));
                    }
                }
            }

            // Get velocity from seed
            #if DEBUG >= 3
            fprintf( stdout, "Getting velocities from seed.\n" );
            fflush(stdout);
            #endif
            toroidal_vel_from_F(  u_lon_tor_seed, u_lat_tor_seed, Psi_seed, source_data, use_mask ? mask : unmask);
            potential_vel_from_F( u_lon_pot_seed, u_lat_pot_seed, Phi_seed, source_data, use_mask ? mask : unmask);

            #if DEBUG >= 3
            fprintf( stdout, "Subtracting seed velocity to get remaining.\n" );
            fflush(stdout);
            #endif
            #pragma omp parallel default(none) \
            shared( Itime, Idepth, RHS_vector, stdout, \
                    u_lon, u_lon_tor_seed, u_lon_pot_seed, u_lon_rem, \
                    u_lat, u_lat_tor_seed, u_lat_pot_seed, u_lat_rem ) \
            private( index, index_sub ) \
            firstprivate( Ntime, Ndepth, Npts )
            {
                #pragma omp for collapse(1) schedule(static)
                for (index_sub = 0; index_sub < Npts; ++index_sub) {
                    index = index_sub + Npts*(Itime*Ndepth + Idepth);
                    u_lon_rem.at( index_sub ) = u_lon.at(index) - u_lon_tor_seed.at(index_sub) - u_lon_pot_seed.at(index_sub);
                    u_lat_rem.at( index_sub ) = u_lat.at(index) - u_lat_tor_seed.at(index_sub) - u_lat_pot_seed.at(index_sub);
                }
            }

            #if DEBUG >= 3
            fprintf( stdout, "Getting divergence and vorticity from remaining velocity.\n" );
            fflush(stdout);
            #endif
            toroidal_vel_div(        div_term, u_lon_rem, u_lat_rem, source_data, use_mask ? mask : unmask );
            toroidal_curl_u_dot_er( vort_term, u_lon_rem, u_lat_rem, source_data, use_mask ? mask : unmask );

            #if DEBUG >= 2
            if ( wRank == 0 ) {
                fprintf(stdout, "Building the RHS of the least squares problem.\n");
                fflush(stdout);
            }
            #endif

            //
            //// Set up the RHS_vector
            //
            
            double is_pole;
            #pragma omp parallel default(none) \
            shared( dAreas, latitude, Itime, Idepth, RHS_vector, div_term, vort_term, u_lon_rem, u_lat_rem ) \
            private( index, index_sub, is_pole ) \
            firstprivate( Ndepth, Ntime, Npts, Tikhov_Laplace, weight_err, deriv_scale_factor, stdout )
            {
                #pragma omp for collapse(1) schedule(static)
                for (index_sub = 0; index_sub < Npts; ++index_sub) {
                    index = index_sub + Npts*(Itime*Ndepth + Idepth);

                    is_pole = std::fabs( std::fabs( latitude.at(index_sub) * 180.0 / M_PI ) - 90 ) < 1e-6;

                    RHS_vector.at( 0*Npts + index_sub) = vort_term.at(index_sub);
                    RHS_vector.at( 1*Npts + index_sub) = div_term.at( index_sub);

                    if ( weight_err ) {
                        RHS_vector.at( 0*Npts + index_sub) *= dAreas.at(index_sub);
                        RHS_vector.at( 1*Npts + index_sub) *= dAreas.at(index_sub);
                    }
                }
            }


            //
            //// Apply the AMG Solver
            //
            std::vector<double> AMG_soln(2*Npts, 0.0);
            int    iters;
            double error;
            std::tie(iters, error) = solve(RHS_vector, AMG_soln);

            fprintf( stdout, "Solver converged after %d iterations with an error of %g.\n", iters, error );
            fflush( stdout );

            }

            // Extract the solution and add the seed back in
            std::vector<double> Psi_vector( AMG_soln.data(),        AMG_soln.data() +     Npts),
                                Phi_vector( AMG_soln.data() + Npts, AMG_soln.data() + 2 * Npts);
            for (size_t ii = 0; ii < Npts; ++ii) {
                Psi_vector.at(ii) += Psi_seed.at(ii);
                Phi_vector.at(ii) += Phi_seed.at(ii);
            }

            // Get velocity associated to computed F field
            #if DEBUG >= 2
            if ( wRank == 0 ) {
                fprintf(stdout, " Extracting velocities and divergence from toroidal field.\n");
                fflush(stdout);
            }
            #endif

            std::vector<double> u_lon_tor(Npts, 0.), u_lat_tor(Npts, 0.), u_lon_pot(Npts, 0.), u_lat_pot(Npts, 0.);
            //alglib::sparsemv( LHS_matr, F_alglib, proj_vals );
            toroidal_vel_from_F(  u_lon_tor, u_lat_tor, Psi_vector, source_data, use_mask ? mask : unmask);
            potential_vel_from_F( u_lon_pot, u_lat_pot, Phi_vector, source_data, use_mask ? mask : unmask);

            //
            //// Store into the full arrays
            //
            #if DEBUG >= 2
            if ( wRank == 0 ) {
                fprintf(stdout, " Storing values into output arrays\n");
                fflush(stdout);
            }
            #endif
            #pragma omp parallel \
            default(none) \
            shared( full_u_lon_tor, u_lon_tor, full_u_lat_tor, u_lat_tor, \
                    full_u_lon_pot, u_lon_pot, full_u_lat_pot, u_lat_pot, \
                    full_Psi, full_Phi, Psi_vector, Phi_vector, \
                    Phi_seed, Psi_seed, \
                    Itime, Idepth ) \
            private( index, index_sub ) \
            firstprivate( Ndepth, Ntime, single_seed, Npts )
            {
                #pragma omp for collapse(1) schedule(static)
                for (index_sub = 0; index_sub < Npts; ++index_sub) {
                    index = index_sub + Npts*(Itime*Ndepth + Idepth);

                    full_u_lon_tor.at(index) = u_lon_tor.at(index_sub) ;
                    full_u_lat_tor.at(index) = u_lat_tor.at(index_sub) ;

                    full_u_lon_pot.at(index) = u_lon_pot.at(index_sub) ;
                    full_u_lat_pot.at(index) = u_lat_pot.at(index_sub) ;

                    full_Psi.at(index) = Psi_vector.at( index_sub );
                    full_Phi.at(index) = Phi_vector.at( index_sub );

                    // If we don't have a seed for the next iteration, use this solution as the seed
                    if (single_seed) {
                        Psi_seed.at(index_sub) = Psi_vector.at(index_sub);
                        Phi_seed.at(index_sub) = Phi_vector.at(index_sub);
                    }
                }
            }

            #if DEBUG >= 0
            if ( source_data.full_Ndepth > 1 ) {
                fprintf(stdout, "  --  --  Rank %d done depth %d after %'zu iterations\n", wRank, Idepth + myStarts.at(1), iters_used );
                fflush(stdout);
            }
            #endif

        }

        #if DEBUG >= 0
        if ( source_data.full_Ntime > 1 ) {
            fprintf(stdout, " -- Rank %d done time %d after %'zu iterations\n", wRank, Itime + myStarts.at(0), iters_used );
            fflush(stdout);
        }
        #endif

        #if DEBUG >= 0
        if ( ( source_data.full_Ntime = 1 ) and ( source_data.full_Ndepth = 1 ) ) {
            fprintf(stdout, " -- Rank done after %'zu iterations\n", iters_used );
            fflush(stdout);
        }
        #endif
    }

    //
    //// Write the output
    //

    const int ndims = 3;
    size_t starts[ndims] = {
        size_t(myStarts.at(0)), size_t(myStarts.at(1)), 0
    };
    size_t counts[ndims] = { size_t(Ntime), size_t(Ndepth), Npts };

    std::vector<std::string> vars_to_write;
    if (not(constants::MINIMAL_OUTPUT)) {
        vars_to_write.push_back("u_lon_tor");
        vars_to_write.push_back("u_lat_tor");

        vars_to_write.push_back("u_lon_pot");
        vars_to_write.push_back("u_lat_pot");

        vars_to_write.push_back("vorticity");
        vars_to_write.push_back("divergence");

        vars_to_write.push_back("proj_vorticity");
        vars_to_write.push_back("proj_divergence");
    }

    vars_to_write.push_back("Psi");
    vars_to_write.push_back("Phi");

    initialize_output_file( source_data, vars_to_write, output_fname.c_str(), -1);

    if (not(constants::MINIMAL_OUTPUT)) {
        write_field_to_output(full_u_lon_tor,  "u_lon_tor",  starts, counts, output_fname.c_str(), &unmask);
        write_field_to_output(full_u_lat_tor,  "u_lat_tor",  starts, counts, output_fname.c_str(), &unmask);

        write_field_to_output(full_u_lon_pot,  "u_lon_pot",  starts, counts, output_fname.c_str(), &unmask);
        write_field_to_output(full_u_lat_pot,  "u_lat_pot",  starts, counts, output_fname.c_str(), &unmask);

        write_field_to_output(vort_term,  "vorticity",  starts, counts, output_fname.c_str(), &unmask);
        write_field_to_output(div_term,   "divergence", starts, counts, output_fname.c_str(), &unmask);

        toroidal_vel_div(        div_term, full_u_lon_pot, full_u_lat_pot, source_data, use_mask ? mask : unmask );
        toroidal_curl_u_dot_er( vort_term, full_u_lon_tor, full_u_lat_tor, source_data, use_mask ? mask : unmask );

        write_field_to_output(vort_term,  "proj_vorticity",  starts, counts, output_fname.c_str(), &unmask);
        write_field_to_output(div_term,   "proj_divergence", starts, counts, output_fname.c_str(), &unmask);
    }

    write_field_to_output(full_Psi, "Psi", starts, counts, output_fname.c_str(), &unmask);
    write_field_to_output(full_Phi, "Phi", starts, counts, output_fname.c_str(), &unmask);

    // Store some solver information
    add_attr_to_file("rel_tol",         rel_tol,                        output_fname.c_str());
    add_attr_to_file("max_iters",       (double) max_iters,             output_fname.c_str());
    add_attr_to_file("diff_order",      (double) constants::DiffOrd,    output_fname.c_str());
    add_attr_to_file("use_mask",        (double) use_mask,              output_fname.c_str());
    add_attr_to_file("weight_err",      (double) weight_err,            output_fname.c_str());
    add_attr_to_file("Tikhov_Laplace",  Tikhov_Laplace,                 output_fname.c_str());


    //
    //// At the very end, compute the L2 and LInf error for each time/depth
    //

    #if DEBUG >= 1
    if (wRank == 0) {
        fprintf(stdout, "Computing the error of the projection.\n");
    }
    #endif

    std::vector<double> projection_2error(      Ntime * Ndepth, 0. ),
                        projection_Inferror(    Ntime * Ndepth, 0. ),
                        velocity_Infnorm(       Ntime * Ndepth, 0. ),
                        projection_KE(          Ntime * Ndepth, 0. ),
                        toroidal_KE(            Ntime * Ndepth, 0. ),
                        potential_KE(           Ntime * Ndepth, 0. ),
                        velocity_2norm(         Ntime * Ndepth, 0. ),
                        tot_areas(              Ntime * Ndepth, 0. );
    double total_area, error2, errorInf, velInf, tor_KE, pot_KE, proj_KE, orig_KE;
    for (int Itime = 0; Itime < Ntime; ++Itime) {
        for (int Idepth = 0; Idepth < Ndepth; ++Idepth) {

            total_area = 0.;
            error2 = 0.;
            tor_KE = 0.;
            pot_KE = 0.;
            proj_KE = 0.;
            orig_KE = 0.;
            errorInf = 0.;
            velInf = 0.;

            #pragma omp parallel \
            default(none) \
            shared( full_u_lon_tor, full_u_lat_tor, full_u_lon_pot, full_u_lat_pot, \
                    u_lon, u_lat, Itime, Idepth, dAreas, latitude ) \
            reduction(+ : total_area, error2, tor_KE, pot_KE, proj_KE, orig_KE) \
            reduction( max : errorInf, velInf )\
            private( index, index_sub ) \
            firstprivate( Ndepth, Ntime, Npts )
            {
                #pragma omp for collapse(1) schedule(static)
                for (index_sub = 0; index_sub < Npts; ++index_sub) {
                    index = index_sub + Npts*(Itime*Ndepth + Idepth);

                    total_area += dAreas.at(index_sub);

                    error2 += dAreas.at(index_sub) * (
                                    pow( u_lon.at(index) - full_u_lon_tor.at(index) - full_u_lon_pot.at(index) , 2.)
                                 +  pow( u_lat.at(index) - full_u_lat_tor.at(index) - full_u_lat_pot.at(index) , 2.)
                            );

                    errorInf = std::fmax( 
                                    errorInf,
                                    sqrt(     pow( u_lon.at(index) - full_u_lon_tor.at(index) - full_u_lon_pot.at(index) , 2.)
                                           +  pow( u_lat.at(index) - full_u_lat_tor.at(index) - full_u_lat_pot.at(index) , 2.)
                                         )
                                    );

                    velInf = std::fmax( velInf,  std::fabs( sqrt( pow( u_lon.at(index) , 2.) +  pow( u_lat.at(index) , 2.) ) )  );

                    tor_KE += dAreas.at(index_sub) * ( pow( full_u_lon_tor.at(index), 2.) + pow( full_u_lat_tor.at(index), 2.) );
                    pot_KE += dAreas.at(index_sub) * ( pow( full_u_lon_pot.at(index), 2.) + pow( full_u_lat_pot.at(index), 2.) );

                    proj_KE += dAreas.at(index_sub) * (
                                    pow( full_u_lon_tor.at(index) + full_u_lon_pot.at(index) , 2.)
                                 +  pow( full_u_lat_tor.at(index) + full_u_lat_pot.at(index) , 2.)
                            );

                    orig_KE += dAreas.at(index_sub) * ( pow( u_lon.at(index), 2.) + pow( u_lat.at(index), 2.) );
                }
            }
            size_t int_index = Index( Itime, Idepth, 0, 0, Ntime, Ndepth, 1, 1);

            tot_areas.at(int_index) = total_area;

            projection_2error.at(   int_index ) = sqrt( error2   / total_area );
            projection_Inferror.at( int_index ) = errorInf;

            velocity_2norm.at(   int_index ) = sqrt( orig_KE  / total_area );
            velocity_Infnorm.at( int_index ) = velInf;

            projection_KE.at( int_index ) = sqrt( proj_KE  / total_area );
            toroidal_KE.at(   int_index ) = sqrt( tor_KE   / total_area );
            potential_KE.at(  int_index ) = sqrt( pot_KE   / total_area );
        }
    }

    const char* dim_names[] = {"time", "depth"};
    const int ndims_error = 2;
    if (wRank == 0) {
        add_var_to_file( "total_area",    dim_names, ndims_error, output_fname.c_str() );

        add_var_to_file( "projection_2error",    dim_names, ndims_error, output_fname.c_str() );
        add_var_to_file( "projection_Inferror",  dim_names, ndims_error, output_fname.c_str() );

        add_var_to_file( "velocity_2norm",   dim_names, ndims_error, output_fname.c_str() );
        add_var_to_file( "velocity_Infnorm", dim_names, ndims_error, output_fname.c_str() );

        add_var_to_file( "projection_KE",  dim_names, ndims_error, output_fname.c_str() );
        add_var_to_file( "toroidal_KE",    dim_names, ndims_error, output_fname.c_str() );
        add_var_to_file( "potential_KE",   dim_names, ndims_error, output_fname.c_str() );
    }
    MPI_Barrier(MPI_COMM_WORLD);

    size_t starts_error[ndims_error] = { size_t(myStarts.at(0)), size_t(myStarts.at(1)) };
    size_t counts_error[ndims_error] = { size_t(Ntime), size_t(Ndepth) };

    write_field_to_output( tot_areas,   "total_area",   starts_error, counts_error, output_fname.c_str() );

    write_field_to_output( projection_2error,   "projection_2error",   starts_error, counts_error, output_fname.c_str() );
    write_field_to_output( projection_Inferror, "projection_Inferror", starts_error, counts_error, output_fname.c_str() );

    write_field_to_output( velocity_2norm,   "velocity_2norm",   starts_error, counts_error, output_fname.c_str() );
    write_field_to_output( velocity_Infnorm, "velocity_Infnorm", starts_error, counts_error, output_fname.c_str() );

    write_field_to_output( projection_KE, "projection_KE", starts_error, counts_error, output_fname.c_str() );
    write_field_to_output( toroidal_KE,   "toroidal_KE",   starts_error, counts_error, output_fname.c_str() );
    write_field_to_output( potential_KE,  "potential_KE",  starts_error, counts_error, output_fname.c_str() );

}
