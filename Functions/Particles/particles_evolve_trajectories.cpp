#include <math.h>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include "../../constants.hpp"
#include "../../functions.hpp"
#include "../../particles.hpp"
#include "../../netcdf_io.hpp"

bool check_if_recycling( 
        const double dt, 
        const double time, 
        const double next_recycle_time, 
        const double particle_lifespan ) {
    // Check if this particle is going to recycle
    //      i.e. if it gets reset to a new random location
    // A non-positive lifespan means no recycling
    if ( particle_lifespan <= 0 ) { return false; }

    if (constants::PARTICLE_RECYCLE_TYPE == constants::ParticleRecycleType::Stochastic) {
        // On average (ish) all of the particles will have recycled
        // within a period of particle_lifespan
        double test_val = ((double) rand() / (RAND_MAX));
        if ( test_val * particle_lifespan < dt ) {
            return true;
        }
    } else if (constants::PARTICLE_RECYCLE_TYPE == constants::ParticleRecycleType::FixedInterval) {
        // If it's a fixed-interval recycling, just check if we've 
        // passed the next recycle time
        if ( time >= next_recycle_time ) {
            return true;
        }
    }
    return false; // back-up catch-all
}

void move_on_sphere(
        double & x_dest,
        double & y_dest,
        const double x_init,
        const double y_init,
        const double u,
        const double v
        ) {

    x_dest = x_init + u / ( constants::R_earth * cos(y_init) );
    y_dest = y_init + v / constants::R_earth;

    // Account for moving across the poles
    if (y_dest >  M_PI/2) { y_dest = ( M_PI/2) - y_dest; x_dest += M_PI; }
    if (y_dest < -M_PI/2) { y_dest = (-M_PI/2) - y_dest; x_dest += M_PI; }

    // Adjust lon for periodicity
    if (x_dest >  M_PI) { x_dest -= 2 * M_PI; }
    if (x_dest < -M_PI) { x_dest += 2 * M_PI; }

}

double get_at_point( 
        const double t, 
        const double lon0, 
        const double lat0,
        const std::vector<double> & var0,
        const std::vector<double> & var1,
        const double t0,
        const double t1,
        const std::vector<double> & lat,
        const std::vector<double> & lon,
        const std::vector<bool> & mask
        ) {

    int left, right, bottom, top;
    particles_get_edges(left, right, bottom, top, lat0, lon0, lat, lon);

    double f0, f1;
    f0 = particles_interp_from_edges( lat0, lon0, lat, lon, &var0, mask, left, right, bottom, top, 0, 0, 1);
    f1 = particles_interp_from_edges( lat0, lon0, lat, lon, &var1, mask, left, right, bottom, top, 0, 0, 1);

    double t_p = ( t - t0 ) / (t1 - t0);

    double var_val = f0 * (1 - t_p) + f1 * t_p;

    return var_val;
}

void RKF45_update(
        double & x_new,
        double & y_new,
        double & dt_new,
        const double t0,
        const double x0,
        const double y0,
        const double dt_seed,
        const double dt_max,
        const std::vector<double> & u_lon_0,
        const std::vector<double> & u_lon_1,
        const std::vector<double> & u_lat_0,
        const std::vector<double> & u_lat_1,
        const double tb0,
        const double tb1,
        const std::vector<double> & lat,
        const std::vector<double> & lon,
        const std::vector<bool> & mask
        ) {

    // Pile of RKF45 coefficients
    const double A0 = 0,    CH0 = 47./450,  CT0 = 1./150,
                 A1 = 2./9, CH1 = 0,        CT1 = 0,
                 A2 = 1./3, CH2 = 12./25,   CT2 = -3./100,
                 A3 = 3./4, CH3 = 32./225,  CT3 = 16./75,
                 A4 = 1.,   CH4 = 1./30,    CT4 = 1./20,
                 A5 = 5./6, CH5 = 6./25,    CT5 = -6./25,
                 B10 = 2./9,
                 B20 = 1./12,   B21 = 1./4,
                 B30 = 69./128, B31 = -243./128, B32 = 135./64,
                 B40 = -17./12, B41 = 27./4,     B42 = -27./5,   B43 = 16./15,
                 B50 = 65./432, B51 = -5./16,    B52 = 13./16,   B53 = 4./27,    B54 = 5./144;

    // and a pile of working variables
    double t, u0, u1, u2, u3, u4, u5, 
              v0, v1, v2, v3, v4, v5, 
                  x1, x2, x3, x4, x5, x_final, 
                  y1, y2, y3, y4, y5, y_final;
    double dt = dt_seed;

    const double eps = 1e-6; // in m/s
    double TE = 10 * eps;

    while ( TE > eps ) {
    
        // First increment
        t = t0 + A0 * dt;
        u0 = dt * get_at_point( t, x0, y0, u_lon_0, u_lon_1, tb0, tb1, lat, lon, mask );
        v0 = dt * get_at_point( t, x0, y0, u_lat_0, u_lat_1, tb0, tb1, lat, lon, mask );

        // Second increment
        t = t0 + A1 * dt;
        move_on_sphere( x1, y1, x0, y0, B10 * u0, B10 * v0 );
        u1 = dt * get_at_point( t, x1, y1, u_lon_0, u_lon_1, tb0, tb1, lat, lon, mask );
        v1 = dt * get_at_point( t, x1, y1, u_lat_0, u_lat_1, tb0, tb1, lat, lon, mask );

        // Third increment
        t = t0 + A2 * dt;
        move_on_sphere( x2, y2, x0, y0, B20 * u0 + B21 * u1, B20 * v0 + B21 * v1 );
        u2 = dt * get_at_point( t, x2, y2, u_lon_0, u_lon_1, tb0, tb1, lat, lon, mask );
        v2 = dt * get_at_point( t, x2, y2, u_lat_0, u_lat_1, tb0, tb1, lat, lon, mask );

        // Fourth increment
        t = t0 + A3 * dt;
        move_on_sphere( x3, y3, x0, y0, B30 * u0 + B31 * u1 + B32 * u2, 
                                        B30 * v0 + B31 * v1 + B32 * v2 );
        u3 = dt * get_at_point( t, x3, y3, u_lon_0, u_lon_1, tb0, tb1, lat, lon, mask );
        v3 = dt * get_at_point( t, x3, y3, u_lat_0, u_lat_1, tb0, tb1, lat, lon, mask );

        // Fifth increment
        t = t0 + A4 * dt;
        move_on_sphere( x4, y4, x0, y0, B40 * u0 + B41 * u1 + B42 * u2 + B43 * u3, 
                                        B40 * v0 + B41 * v1 + B42 * v2 + B43 * v3);
        u4 = dt * get_at_point( t, x4, y4, u_lon_0, u_lon_1, tb0, tb1, lat, lon, mask );
        v4 = dt * get_at_point( t, x4, y4, u_lat_0, u_lat_1, tb0, tb1, lat, lon, mask );

        // Sixth increment
        t = t0 + A5 * dt;
        move_on_sphere( x5, y5, x0, y0, B50 * u0 + B51 * u1 + B52 * u2 + B53 * u3 + B54 * u4, 
                                        B50 * v0 + B51 * v1 + B52 * v2 + B53 * v3 + B54 * v4);
        u5 = dt * get_at_point( t, x5, y5, u_lon_0, u_lon_1, tb0, tb1, lat, lon, mask );
        v5 = dt * get_at_point( t, x5, y5, u_lat_0, u_lat_1, tb0, tb1, lat, lon, mask );


        // Net displacement vector
        double  u_final = CH0 * u0 + CH1 * u1 + CH2 * u2 + CH3 * u3 + CH4 * u4 * CH5 * u5,
                v_final = CH0 * v0 + CH1 * v1 + CH2 * v2 + CH3 * v3 + CH4 * v4 * CH5 * v5;
        double x_final, y_final;
        move_on_sphere( x_final, y_final, x0, y0, u_final, v_final );

        // Estimate the truncation error
        double  TE_u = CT0 * u0 + CT1 * u1 + CT2 * u2 + CT3 * u3 + CT4 * u4 * CT5 * u5,
                TE_v = CT0 * v0 + CT1 * v1 + CT2 * v2 + CT3 * v3 + CT4 * v4 * CT5 * v5;
        TE = sqrt( TE_u*TE_u + TE_v*TE_v );

        // If the error was too large, then we'll need to redo the step with a smaller dt.
        // If the error was smaller than our tolerance, than the step is adaptively increased
        dt = std::fmin( 0.9 * dt * pow( eps / TE, 1./5 ), dt_max );
    }

    // After success, store the desired values
    dt_new = dt;
    x_new = x_final;
    y_new = y_final;
}

void particles_evolve_trajectories(
        std::vector<double> & part_lon_hist,
        std::vector<double> & part_lat_hist,
        std::vector< std::vector<double> > & field_trajectories,
        const std::vector<double> & starting_lat,
        const std::vector<double> & starting_lon,
        const std::vector<double> & target_times,
        const double particle_lifespan,
        std::vector<double> & vel_lon_0,
        std::vector<double> & vel_lon_1,
        std::vector<double> & vel_lat_0,
        std::vector<double> & vel_lat_1,
        const std::string zonal_vel_name,
        const std::string merid_vel_name,
        const std::string input_fname,
        const std::vector<const std::vector<double>*> & fields_to_track,
        const std::vector<std::string> & names_of_tracked_fields,
        const std::vector<double> & time,
        const std::vector<double> & lat,
        const std::vector<double> & lon,
        const std::vector<bool> & mask,
        const MPI_Comm comm
        ) {

    int wRank, wSize;
    MPI_Comm_rank( comm, &wRank );
    MPI_Comm_size( comm, &wSize );

    double t_part, lon0, lat0, lon_new, lat_new,
           dx_loc, dy_loc, dt, dt_max, dt_new,
           up_0, up_1, vp_0, vp_1,
           vel_lon_part, vel_lat_part, field_val,
           test_val, lon_rng, lat_rng, lon_mid, lat_mid,
           time_p, next_recycle_time, time_block_0, time_block_1,
           data_load_time;

    const double dlon = lon.at(1) - lon.at(0),
                 dlat = lat.at(1) - lat.at(0),
                 cfl  = 1e-1,
                 U0   = 2.,
                 dt_target = target_times.at(1) - target_times.at(0);

    const unsigned int  Nlat   = lat.size(),
                        Nlon   = lon.size(),
                        Ntime  = time.size(),
                        Ndepth = 1,
                        Nparts = starting_lat.size(),
                        Nouts  = target_times.size();

    int left, right, bottom, top;
    bool do_recycle;


    unsigned int out_ind, prev_out_ind, step_iter, Ip;
    size_t index, next_load_index = 1;

    // Step through time 'blocks' i.e. get to the time of the next velocity data
    //  each particle is using adaptive stepping, so just loop through them getting there
    // We've already loaded in the first two times, so just flag the target time and continue.
    prev_out_ind = 0;
    for ( next_load_index = 1; next_load_index < Ntime; next_load_index++ ) {
        data_load_time = time.at(next_load_index);

        if (next_load_index > 1) {
            // Swap time 1 to time 0
            vel_lon_1.swap(vel_lon_0);
            vel_lat_1.swap(vel_lat_0);

            // Load in next time-block as time 1
            read_var_from_file_at_time( vel_lon_1, next_load_index, zonal_vel_name, input_fname );
            read_var_from_file_at_time( vel_lat_1, next_load_index, merid_vel_name, input_fname );
        }
        #pragma omp parallel \
        default(none) \
        shared( lat, lon, vel_lon_0, vel_lon_1, vel_lat_0, vel_lat_1, mask,\
                target_times, time, part_lon_hist, part_lat_hist,\
                field_trajectories, fields_to_track,\
                wRank, wSize)\
        private(Ip, index, t_part, step_iter, lon0, lat0, \
                dt_max, dt_new, lon_new, lat_new, \
                up_0, up_1, vp_0, vp_1, data_load_time, \
                dx_loc, dy_loc, dt, time_p, vel_lon_part, vel_lat_part, field_val,\
                do_recycle, next_recycle_time, time_block_0, time_block_1, \
                left, right, bottom, top) \
        firstprivate( Nparts, Nouts, Ntime, dlon, dlat, dt_target, \
                      particle_lifespan, next_load_index, prev_out_ind ) \
        reduction(max : out_ind)
        {

            #pragma omp for collapse(1) schedule(static)
            for (Ip = 0; Ip < Nparts; ++Ip) {

                // Give each particle a different random seed
                srand( (Ip + wRank * Nparts) * next_load_index );
                do_recycle = false;
                dt = 0.;

                //
                time_block_0 = time.at(next_load_index-1);
                time_block_1 = std::fmin( data_load_time, target_times.back() );

                // Particle time
                t_part    = time_block_0;   //
                out_ind   = prev_out_ind+1; // Index for the next output
                step_iter = 0;              //

                // Particle position
                index = Index(0,       0,      prev_out_ind, Ip,
                              Ntime,   Ndepth, Nouts,        Nparts);
                lon0 = part_lon_hist.at(index);
                lat0 = part_lat_hist.at(index);

                // Check if initial positions are fill_value, and recycle if they are
                // RECYCLE!!

                /*
                // Get initial values for tracked fields
                index = Index(0,       0,      out_ind, Ip,
                              Ntime,   Ndepth, Nouts,   Nparts);

                   time_p = ( t_part - time_block_0 ) / ( time_block_1 - time_block_0 );

                   particles_get_edges(left, right, bottom, top, lat0, lon0, lat, lon);
                   for (size_t Ifield = 0; Ifield < fields_to_track.size(); ++Ifield) {
                   field_val = particles_interp_from_edges(lat0, lon0, lat, lon, 
                   fields_to_track.at(Ifield), mask,
                   left, right, bottom, top, time_p, ref_ind, Ntime);
                   field_trajectories.at(Ifield).at(index) = field_val;
                   }
                out_ind++;
                */

                // Seed values for velocities (only used for dt)
                particles_get_edges(left, right, bottom, top, lat0, lon0, lat, lon);
                if ( (bottom < 0) or (top < 0) ) { continue; }
                vel_lon_part = particles_interp_from_edges( lat0, lon0, lat, lon, &vel_lon_0, 
                        mask, left, right, bottom, top, 0, 0, 1);

                vel_lat_part = particles_interp_from_edges( lat0, lon0, lat, lon, &vel_lat_0, 
                        mask, left, right, bottom, top, 0, 0, 1);

                while (t_part < time_block_1) {

                    // Get local dt
                    //   we'll use the previous velocities, which should
                    //   be fine, since it doesn't change very quickly
                    dx_loc = dlon * constants::R_earth * cos(lat0);
                    dy_loc = dlat * constants::R_earth;

                    dt_max = cfl * std::min( dx_loc / std::max(vel_lon_part, 1e-3), 
                                             dy_loc / std::max(vel_lat_part, 1e-3) );
                    if (dt == 0) { dt = 0.01 * dt_max; }

                    // If we're close to an output time, adjust the dt to land on it exactly
                    if ( ( (size_t)out_ind < target_times.size() )
                            and ( (t_part + dt) - target_times.at(out_ind) > (dt_target / 50.) ) 
                       )
                    {
                        dt_max = target_times.at(out_ind) - t_part;
                        dt = dt_max;
                    }

                    //
                    //// RKF45 Time-stepping
                    //
                    RKF45_update( lon_new, lat_new, dt_new,
                            t_part, lon0, lat0, dt, dt_max,
                            vel_lon_0, vel_lon_1, vel_lat_0, vel_lat_1,
                            time_block_0, time_block_1, lat, lon, mask
                            );
                    lon0 = lon_new;
                    lat0 = lat_new;
                    dt = dt_new;

                    //
                    //// Time-stepping is a simple first-order symplectic scheme
                    //
                    /*
                    time_p = ( t_part - time_block_0 ) / ( time_block_1 - time_block_0 );

                    //
                    //// Get u_lon at position and advance lon position
                    //
                    particles_get_edges(left, right, bottom, top, lat0, lon0, lat, lon);
                    if ( (bottom < 0) or (top < 0) ) { break; }
                    up_0 = particles_interp_from_edges( lat0, lon0, lat, lon, &vel_lon_0, 
                            mask, left, right, bottom, top, 0, 0, 1);
                    up_1 = particles_interp_from_edges (lat0, lon0, lat, lon, &vel_lon_1, 
                            mask, left, right, bottom, top, 0, 0, 1);
                    vel_lon_part = up_0 * (1 - time_p) + up_1 * time_p;
                    if ( fabs(vel_lon_part) > 100. ) { break; }

                    // convert to radial velocity and step in space
                    lon0 += dt * vel_lon_part / (constants::R_earth * cos(lat0));
                    if (lon0 >  M_PI) { lon0 -= 2 * M_PI; }
                    if (lon0 < -M_PI) { lon0 += 2 * M_PI; }

                    //
                    //// Get u_lat at position and advance lat position
                    //
                    particles_get_edges(left, right, bottom, top, lat0, lon0, lat, lon);
                    if ( (bottom < 0) or (top < 0) ) { break; }
                    vp_0 = particles_interp_from_edges( lat0, lon0, lat, lon, &vel_lat_0, 
                            mask, left, right, bottom, top, 0, 0, 1);
                    vp_1 = particles_interp_from_edges (lat0, lon0, lat, lon, &vel_lat_1, 
                            mask, left, right, bottom, top, 0, 0, 1);
                    vel_lat_part = vp_0 * (1 - time_p) + vp_1 * time_p;
                    if ( fabs(vel_lat_part) > 100. ) { break; }

                    // convert to radial velocity and step in space
                    lat0 += dt * vel_lat_part / constants::R_earth;

                    // Update time
                    t_part += dt;
                    */

                    // Keep track of if we're going to recycle at the next output
                    //  we're syncing recycling with outputs so that we can flag
                    //  them as singleton nan values
                    do_recycle = do_recycle or check_if_recycling( dt, t_part, next_recycle_time, particle_lifespan);


                    // Track, if at right time
                    if (      (t_part < target_times.back()) 
                            and (t_part >= target_times.at(out_ind)) 
                       ){

                        index = Index(0,       0,      out_ind, Ip,
                                      Ntime,   Ndepth, Nouts,   Nparts);

                        if ( do_recycle ) {
                            part_lon_hist.at(index) = constants::fill_value;
                            part_lat_hist.at(index) = constants::fill_value;
                            next_recycle_time += particle_lifespan; 
                            // UPDATE PARTICLE POSITION!


                            do_recycle = false;
                        } else {
                            part_lon_hist.at(index) = lon0;
                            part_lat_hist.at(index) = lat0;
                        }


                        /*
                        particles_get_edges(left, right, bottom, top, lat0, lon0, lat, lon);

                        for (size_t Ifield = 0; Ifield < fields_to_track.size(); ++Ifield) {
                           field_val = particles_interp_from_edges(lat0, lon0, lat, lon, 
                           fields_to_track.at(Ifield), mask,
                           left, right, bottom, top, time_p, ref_ind, Ntime);
                           field_trajectories.at(Ifield).at(index) = field_val;
                        }

                        */
                        out_ind++;
                    }

                    // If we've somehow gone too far in time, stop.
                    if ( (t_part >= target_times.back()) or (out_ind >= Nouts) ) { break; }

                    step_iter++;
                } // close inner time loop
            } // close particle loop
        } // close pragma 
        prev_out_ind = out_ind;
        #if DEBUG >= 1
        if (wRank == 0) {
            fprintf( stdout, "Finished time-block %zu. Previous out index is %d.\n", 
                    next_load_index, prev_out_ind );
            fflush( stdout );
        }
        #endif
    } // close time-block loop
}
