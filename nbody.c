/* Title: N-body Problem PR1 PHYM004
Author: Ewan-James Thomas
File_name: nbody.c
License: Public Domain
*/

/****************** PHYM004 PR1: The N-body Problem ******************/
//
// Purpose
// -------
//  This program serves to solve the coupled differential equations that govern the
// dynamics of N massive bodies in a gravitational field. Once solved, it then plots 
// the motion of these bodies using Gnuplot.
// 
//  The user has the ability to choose from the "velocity-verlet" and "The 4th order
// Runge-Kutta" methods in to solve the equations. The step-size used in these methods
// is also defined by the user in runtime. 
//
// Usage
// -----
// This program must be executed with a data file. The data file must have the following format (values are tab separated).
// Name     x   y   z   vx  vy  vz
// Error thrown if input file has incorrect format.
// Other files that must be in the same directory during execturion are - "gnu_vv.script", "gnu_rk4.script". Gnu must also be installed and linked.
// Output will be a gnuplot window displaying orbits, a data file containing positions "nbody_gnuplot.dat", a data file containing energies "energy.dat" & "e_over_t.dat"
// 
// Example
// --------
// Solar system data file: orbit_data_ss.txt
/*
# Name	Mass	x	y	z	Vx	Vy	Vz	                Comments can be written in data file by starting a line with '#
Sun 1.9885e30   0.0 0.0   0.0 0   0.0 0.0
Mercury 0.330e24    57.9e9  0.0 0.0 0.0 47.4e3  0.0
Venus   4.87e24 108.2e9 0.0 0.0 0.0 35.0e3  0.0
Earth	5.9726e24	1.5e11	0.0	0.0	0.0	2.929e4	0.0
Mars    6.39e23 222.35e9    0.0 0.0 0.0 24000    0.0
Jupiter 1898e24 778.6e9 0.0 0.0 0.0 13000   0.0
Saturn 568e24   1433.5e9    0.0 0.0 0.0 9700    0.0
Uranus  86.8e24 2872.5e9    0.0 0.0 0.0 6.8e3   0.0
Neptune 102e24  4495.1e9    0.0 0.0 0.0 5.4e3   0.0

1.) compile with -lgsl to link gsl manually: gcc nbody.c -o nbody -lgsl -Wall
2.) execute with data file: ./nbody orbit_data_ss.txt

-----Welcome to the Gravitational N-Body solver----
Your input file is orbit_data_ss.txt

-----Please choose a method of analysis-----
Velocity verlet method | -> Enter 1
RK4 method employed by GSL | -> Enter 2

3.) Choose method by entering 1 or 2

Enter desired step size:

4.) Enter desired step size

Enter number of iterations

5.) Enter number of iteration

Program will now run and prodce all output files/figures.
*/

/*      UPDATES
 Date         Version  Comments
 ----         -------  --------
 11/11/20       0.0.1  Build started -> Taking input form txt file
 12/11/20       0.0.2  Input taken and correctly formated
 13/11/20       0.0.3  Working on Velocity Verlet method
 17/11/20       0.0.3  Correctly obtained initial accelerations
 21/11/20       0.0.4  NOW accelerations are correct! (r^3 not r^2) || Method of coutning bodies doesn't work!!
 1/12/20        0.0.5  Iterative improvemenets since last updated, now plots to gnu using VV method and is correct
 1/12/20        0.0.6  RK4 method using gsl now gives numbers by they dont make sense, need to fix this.
 6/12/20        0.0.7  RK4 Method now works. Need to generalise this file and script for n bodies
 8/12/20        1.0.0  Both methods work for n bodies so long as n is less than max bodies.
 9/12/20        1.0.1  Trying to calculate total energy, but it seems to oscillate for VV. Investigate why.
 9/12/20        1.1.0  Energies for vv and gsl method make sense :)
 10/12/20       1.1.1  Energies for vv and gsl output to dat file for plotting
 12/12/20       1.1.2  Review of energy - wasn't updating velocities for GSL RK4 method
 15/12/20       1.5.0  All methods now working, analysis begins now
 17/12/20       1.6.0  Final stable release


*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <gsl/gsl_const_mksa.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_matrix.h>
#include <sys/time.h>


#define VERSION = "1.6.0";
#define REV_DATE = "17-Decmber-2020";

#define GNU_CALL "gnuplot"
#define GNU_SCRIPT_1 "gnu_vv.script"
#define GNU_SCRIPT_2 "gnu_rk4.script"
#define GNU_DATA "nbody_gnuplot.dat"

#define MAX_NAME_SIZE 24
#define MAX_ITERS_SIZE 20
#define MAX_BODIES 10
#define MAX_FILE_LINE_SIZE 150
#define ITEMS_PER_LINE 8
#define G GSL_CONST_MKSA_GRAVITATIONAL_CONSTANT
#define POLE_PREVENTOR 1.0
#define EQN_PER_BODY 6
#define EPSREL 10e-8
#define EPSABS 10e-8
#define HSTART 1e-3
#define USEC_IN_SEC 1000000

static struct timeval stop, start;
// 
typedef enum coords {x, y, z, N_coords} Coords;

// Body object used to store information about celestial bodies.
typedef struct{
    char name[MAX_NAME_SIZE];
    double T;
    double V;
    double H;
    long double mass;
    double r_init[N_coords];
    double r[N_coords];
    double v[N_coords];
    double a[N_coords];
} body;
// energy config dtype for the entire system.
typedef struct{
    double T;
    double V;
    double H;
} sys_energy_config;

//xfile is a function used to check if fopen succeeds.
void xfile(FILE* fp, const char* filename){
    if(!fp){
        fprintf(stderr, "Error: Could not open the file '%s'.\n",  filename);
        exit(1);
    }
}
// xmalloc is a function used to check if a pointer has been successfully declared.
void* xmalloc(size_t bytes){
    void* retval = malloc(bytes);
    if(retval){ //"succeed or die"
        return retval;
    }
    else{
        fprintf(stderr, "Fatal error: Memory exhausted (xmalloc of %zu bytes)", bytes);
        exit(-1);
    }
}
// number_of_bodies reads the input file to determine the number of bodies 
int number_of_bodies(const char* filename){
    FILE* fp = fopen(filename, "r");
    xfile(fp, filename);

    char line[MAX_FILE_LINE_SIZE];
    int bodies = 0;
    while(fgets(line, MAX_FILE_LINE_SIZE,fp) && bodies < MAX_BODIES){
        if(line[0]!='#'){
            bodies++;
        }
    }
    return bodies;
}
// read_from_file returns an array of "bodies"
body* read_from_file(const char* filename, int n){ /* Based on CDHW code from "ReadOrbits.c"*/
    char line[MAX_FILE_LINE_SIZE];
    char namebuf[MAX_FILE_LINE_SIZE];
    
    FILE* file = fopen(filename,"r");
    xfile(file, filename);
    // Creating an array of bodies
    body* bodies = xmalloc(MAX_BODIES * sizeof(body));

    int bodyN = 0;
    int items_in_line = 0;
    // So long as the line doesn't exceed MA_FILE_LINE_SIZE and we're on a line less than the number of bodies in the file
    while(fgets(line, MAX_FILE_LINE_SIZE, file) && bodyN < n){
        // Comments in txt file are followed by a '#'. Lines starting with # will be ignored
        if(line[0] != '#'){
            // Assigns values from file to the array of bodies
            items_in_line = sscanf(line, "%s %Lf %lg %lg %lg %lg %lg %lg", namebuf, &bodies[bodyN].mass,
            &bodies[bodyN].r[x], &bodies[bodyN].r[y], &bodies[bodyN].r[z], 
            &bodies[bodyN].v[x], &bodies[bodyN].v[y], &bodies[bodyN].v[z]);

            bodies[bodyN].r_init[x] = bodies[bodyN].r[x];
            bodies[bodyN].r_init[y] = bodies[bodyN].r[y];
            bodies[bodyN].r_init[z] = bodies[bodyN].r[z];

            if(items_in_line == ITEMS_PER_LINE){
                // Transfer of name from namebuf to body.name
                strcpy(bodies[bodyN].name, namebuf);
                bodyN++;
            }else{
                fprintf(stderr, "Incorrect file format: %s\n", line);
                free(bodies);
                return NULL;
            }
        }
    }
    return bodies;  
}
// function used to print data (used in early testing e.g. version 0.0.2)
void print_data(body* bodies, int n){
    printf("Name\tMass\t\tx\ty\tz\tvx\tvy\tvz\n");
    for(int body = 0; body < n; body++){
        printf("%s\t%.4Le\t%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n", bodies[body].name, bodies[body].mass, bodies[body].r[x], bodies[body].r[y],
        bodies[body].r[z], bodies[body].v[x], bodies[body].v[y], bodies[body].v[z]);
    }
}
//mod_distance returns the modulus of the distance between any two bodies -> used in vv method
double mod_distance(body bodyA, body bodyB){ 
    // POLE_PREVENTOR does what it says on the tin. No collisions, so it stops the force from diverging when two planets are "on top" of each other.
    double mod_dist = sqrt(((bodyA.r[x] - bodyB.r[x])*(bodyA.r[x] - bodyB.r[x])+
    (bodyA.r[y] - bodyB.r[y])*(bodyA.r[y] - bodyB.r[y]) + (bodyA.r[z] - bodyB.r[z])*(bodyA.r[z] - bodyB.r[z])+POLE_PREVENTOR));

    return mod_dist;
}
// Vec_distance returns the vector that describes the distance between any two bodies -> used in vv method
double* vec_distance(body bodyA, body bodyB){
    // Want to return an array of size 3, each element contains the x,y,z compenent of r_j-r_j
    double* vec_dist = (double *)xmalloc(N_coords*sizeof(double));
    for(int dimension = 0; dimension < N_coords; dimension++){
        vec_dist[dimension] = bodyA.r[dimension] - bodyB.r[dimension];
    }
    return vec_dist;
}
// Calculates initial acceleration and assigns these values to the bodies -> used in vv method
void initial_accelerations(body *bodies, int n){

    for(int dim = 0; dim < N_coords; dim++){
        for(int body = 0; body < n; body++){
            for(int subbody = 0; subbody < n; subbody++){
                if(body != subbody){
                        bodies[body].a[dim] += G*bodies[subbody].mass * vec_distance(bodies[body], bodies[subbody])[dim]/pow(mod_distance(bodies[subbody], bodies[body]), 3.0);
                } 
            }
        }
    }
    return;
}
// Ualculates the interim values of acceleration for a gievn body in a given dimension (very general) -> used in vv method
double acceleration_of_body(body main_body, body *bodies, int dim, int n){

    double acc = 0.0;

    for(int bod = 0; bod < n; bod++){  
        if(strcmp(main_body.name, bodies[bod].name) != 0){
            acc += G*bodies[bod].mass * (vec_distance(bodies[bod], main_body)[dim]) / pow(mod_distance(main_body, bodies[bod]), 3.0);
        }
    }
    return acc;
}
// Updates positions, velocities and accelerations of bodies using equations of vv method
void update(body *bodies, double step, int n){

    // Main part of the velocity verlet method
    for(int j = 0; j < n; j++){
        double new_acc[N_coords] = {0.0, 0.0, 0.0};

        // Update position based on velocity, accn, and time-step.
        bodies[j].r[x]  += bodies[j].v[x] * step + 0.5*bodies[j].a[x]*step*step;
        bodies[j].r[y]  += bodies[j].v[y] * step + 0.5*bodies[j].a[y]*step*step;
        bodies[j].r[z]  += bodies[j].v[z] * step + 0.5*bodies[j].a[z]*step*step;

        // Store the new acceleration in an array as we still need previous accn to update the velocites.
        new_acc[0] = acceleration_of_body(bodies[j], bodies, 0, n);
        new_acc[1] = acceleration_of_body(bodies[j], bodies, 1, n);
        new_acc[2] = acceleration_of_body(bodies[j], bodies, 2, n);
        
        // New velocities depend on average of old accn and new accn.
        bodies[j].v[x] = bodies[j].v[x] + (bodies[j].a[x]+new_acc[0])*0.5*step;
        bodies[j].v[y] = bodies[j].v[y] + (bodies[j].a[y]+new_acc[1])*0.5*step;
        bodies[j].v[z] = bodies[j].v[z] + (bodies[j].a[z]+new_acc[2])*0.5*step;

        // Can overwrite the old velocities with the new ones.
        bodies[j].a[x] = new_acc[0];
        bodies[j].a[y] = new_acc[1];
        bodies[j].a[z] = new_acc[2];
        // Then repeat for every body for every step.
    }
    return;
}
// func is used to define the system of ordinary differential equations that are solved by rk4_exe
int func(double t, const double y[], double f[], void* params){
    (void) t; // Avoids unused parameter warning
    //Casting the void array params to double
    double *masses = (double*)params;
    //First element of params also stores number of bodies
    int n = (int) masses[0];
    for(int i = 0; i < n; i++){
        // Need 6 equations per body (3 for position, 3 for accn)
        // y[] contains inital positions and velocities in the form [x1,y1,z1,vx1,vy1,vz1,x2,y2,z2,...,vxn,vyn,vzn]
        f[EQN_PER_BODY*i] = y[EQN_PER_BODY*i+3]; // \dot{x} = vx
        f[EQN_PER_BODY*i+1] = y[EQN_PER_BODY*i+4]; // \dot{y} = vy
        f[EQN_PER_BODY*i+2] = y[EQN_PER_BODY*i+5]; // \dot{z} = vz
        
        // accn in each dimension for each body
        double accn1 = 0.0;
        double accn2 = 0.0;
        double accn3 = 0.0;

        for(int j = 0; j < n; j++){
            if(i!=j){
                double delta_x = y[EQN_PER_BODY*i] - y[EQN_PER_BODY*j];
                double delta_y = y[EQN_PER_BODY*i+1] - y[EQN_PER_BODY*j+1];
                double delta_z = y[EQN_PER_BODY*i+2] - y[EQN_PER_BODY*j+2];

                double mod_dist = sqrt( delta_x*delta_x + delta_y*delta_y + delta_z*delta_z + POLE_PREVENTOR);

                // index for masses must go to j+1 because the 0th element of params is n
                accn1 -= (G*masses[j+1] * (delta_x) / pow(mod_dist, 3.0)); //x
                accn2 -= (G*masses[j+1] * (delta_y) / pow(mod_dist, 3.0)); //y
                accn3 -= (G*masses[j+1] * (delta_z) / pow(mod_dist, 3.0)); //z
            }

        f[EQN_PER_BODY*i+3] = accn1; // \ddot{x} = \sum{forces_in_x/masses}
        f[EQN_PER_BODY*i+4] = accn2; // \ddot{y} = \sum{forces_in_y/masses}
        f[EQN_PER_BODY*i+5] = accn3; // \ddot{z} = \sum{forces_in_z/masses}
        }
    }
    return GSL_SUCCESS;
}
// finds total energy of system and returns these energies in the sys_energy_config dtype
sys_energy_config find_total_energy(body* bodies, int n){
    sys_energy_config system;
    //initialsing all energies to zero
    system.T = 0.0;
    system.H = 0.0;
    system.V = 0.0;

    //calculating kinetic energy of each equation
    for(int body1 = 0; body1 < n; body1++){
        bodies[body1].T = 0.5 * bodies[body1].mass * (bodies[body1].v[x]*bodies[body1].v[x] + bodies[body1].v[y]*bodies[body1].v[y] + bodies[body1].v[z]*bodies[body1].v[z]);
        
        //calculating potential energy for each body. Must multiply by a half so don't overcount. i.e body1!=body2 and body2!=body1
        double ep = 0.0;
        for(int body2 = 0; body2 < n; body2++){ 
            if(body1 != body2){ 
                ep += -G*bodies[body1].mass*bodies[body2].mass / mod_distance(bodies[body1], bodies[body2]); 
            }
        }
        bodies[body1].V = 0.5*ep;
        bodies[body1].H = (bodies[body1].T + bodies[body1].V);

        system.H += bodies[body1].H;
        system.T += bodies[body1].T;
        system.V += bodies[body1].V;
    }
    // One function returning 3 values in form of a structure of type system_energy_config.
    return system;
}
// Brings together all functions defined for the vv method to evolve the system
void velocity_verlet_method(body *bodies, const char* filename, int n){
    double step;
    printf("Enter desired step size:\n");
    scanf("%lg", &step);
    if(step < 0){
        printf("Step size must be positive\n");
        exit(0);
    }

    char iterations_string[MAX_ITERS_SIZE]; 
    int iterations;
    printf("Enter number of iterations\n");
    scanf("%s", iterations_string);

    //Need integer number of steps
    int val = atoi(iterations_string);
    if(val!=0){
        iterations = val;
    }
    else{
        printf("Please input an integer number of iterations\n");
        exit(0);
    }
    // Allows for timing of vv to be investigated
    gettimeofday(&start, NULL);
    //initial energy so the change in energy overtime can be investigated
    double H_init = find_total_energy(bodies, n).H;

    // Opening required data files and checking if opened successfully.
    FILE *fdat = fopen (filename, "w");
    xfile(fdat, filename);
    FILE *fenergy = fopen("energy.dat", "w");
    xfile(fenergy, "energy.dat");
    FILE *e_over_t = fopen("e_over_t.dat", "w");
    xfile(e_over_t, "e_over_t.dat");
    
    // The Velocity Verlet method below
    initial_accelerations(bodies, n);
    for(int body = 0; body < n; body++){
        fprintf(fdat, "%s\n", bodies[body].name);
        int i = 0;
        // After every iteration we update the positions and energies and write them to their respective files
        while(i < iterations){
            update(bodies, step, n);
            fprintf(fdat, "%.4lg\t%.4lg\t%.4lg\t\n", bodies[body].r[x], bodies[body].r[y], bodies[body].r[z]);
            fprintf(fenergy, "%lg\t%lg\t%lg\n", find_total_energy(bodies, n).H, find_total_energy(bodies, n).T, find_total_energy(bodies, n).V);
            fprintf(e_over_t, "%lg\n", find_total_energy(bodies, n).H/H_init);
            i++;
            // Uncomment section below when using single planet orbiting sun to determine the period
            /*if(bodies[body].r[x] < bodies[body].r_init[x] + bodies[body].r_init[x]/100.0 && bodies[body].r[x] > bodies[body].r_init[x] - bodies[body].r_init[x]/100.0 && i*step > iterations*step/2){
                printf("Period of %s is %lg\n", bodies[body].name, i*step);
            }*/
        }fprintf(fdat, "\n\n");
    }
    fclose(fdat);
    fclose(fenergy);
    fclose(e_over_t);
    gettimeofday(&stop, NULL);
    //printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * USEC_IN_SEC + stop.tv_usec - start.tv_usec); Uncomment and compile to print run-time
    return;
}
// Uses func to define system and utilises a gsl_odeive2 driver to evolve the system. The rk4 stepping function is used as per the project requirements.
void rk4_exe(body* bodies, int n){

    double step;
    printf("Enter desired step size:\n");
    scanf("%lg", &step);
    if(step < 0){
        printf("Step size must be positive\n");
        exit(0);
    }

    char iterations_string[MAX_ITERS_SIZE]; 
    int iterations;
    printf("Enter number of iterations\n");
    scanf("%s", iterations_string);

    int val = atoi(iterations_string);
    if(val!=0){
        iterations = val;
    }
    else{
        printf("Please input an integer number of iterations\n");
        exit(0);
    }
    gettimeofday(&start, NULL);
    double masses[n+1]; //masses also stores the number of bodies at index 0. This is so we can pass n to func in params.
    masses[0] = n;
    for(int l = 1; l < n+1; l++){
        masses[l] = bodies[l-1].mass;
    }
    double t = 0.0, t1 = iterations*step;
    double init[EQN_PER_BODY*n];
    double H_init = find_total_energy(bodies, n).H;

    FILE *fdat = fopen("nbody_gnuplot.dat", "w");
    xfile(fdat, "nbody_gnuplot.dat");
    FILE *fenergy = fopen("energy.dat", "w");
    xfile(fenergy, "energy.dat");
    FILE *e_over_t_2 = fopen("e_over_t_2.dat", "w");
    xfile(e_over_t_2, "e_over_t_2.dat");

    for(int name = 0; name < n; name++){
        fprintf(fdat, "%s\t\t\t", bodies[name].name);
    }fprintf(fdat, "\n");

    //assigning initial values to array that will be passed to the driver
    for(int k = 0; k < n; k++){
        init[EQN_PER_BODY*k+0] = bodies[k].r[x];
        init[EQN_PER_BODY*k+1] = bodies[k].r[y];
        init[EQN_PER_BODY*k+2] = bodies[k].r[z];
        init[EQN_PER_BODY*k+3] = bodies[k].v[x];
        init[EQN_PER_BODY*k+4] = bodies[k].v[y];
        init[EQN_PER_BODY*k+5] = bodies[k].v[z]; 
    }
    gsl_odeiv2_system sys = {func, NULL, EQN_PER_BODY*n, masses};
    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk4, HSTART, EPSABS, EPSREL);
    
    //time values defined by step size and number of steps
    for(int i = 0; i <=iterations ; i++){ 
        double ti = i * t1/iterations;
        int status = gsl_odeiv2_driver_apply(d, &t, ti, init);
        
        if(status != GSL_SUCCESS){
            printf("Error, return value=%d\n", status);
            break;
        }
        //Updating the bodies positions and velocities
        for(int bod = 0; bod < n; bod++){
            bodies[bod].r[x] = init[EQN_PER_BODY*bod];
            bodies[bod].r[y] = init[EQN_PER_BODY*bod+1];
            bodies[bod].r[z] = init[EQN_PER_BODY*bod+2];
            bodies[bod].v[x] = init[EQN_PER_BODY*bod+3];
            bodies[bod].v[y] = init[EQN_PER_BODY*bod+4];
            bodies[bod].v[z] = init[EQN_PER_BODY*bod+5];
            fprintf(fdat, "%.4lg\t%.4lg\t%.4lg\t", bodies[bod].r[x], bodies[bod].r[y], bodies[bod].r[z]);
            /*if(bodies[bod].r[x] < bodies[bod].r_init[x] + bodies[bod].r_init[x]/1000.0 && bodies[bod].r[x] > bodies[bod].r_init[x] - bodies[bod].r_init[x]/1000.0 && ti > iterations*step/2){
                printf("Period of %s is %lg\n", bodies[bod].name, ti);
            }*/
        }fprintf(fenergy, "%lg\t%lg\t%lg\n", find_total_energy(bodies, n).H, find_total_energy(bodies, n).T, find_total_energy(bodies, n).V);
        fprintf(e_over_t_2, "%lg\n", find_total_energy(bodies, n).H/H_init);
        fprintf(fdat, "\n");
    }
    gsl_odeiv2_driver_free(d);
    fclose(fdat);
    fclose(fenergy);
    fclose(e_over_t_2);
    gettimeofday(&stop, NULL);
    //printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * USEC_IN_SEC + stop.tv_usec - start.tv_usec); Uncomment and compile to print run-time
}
// main
int main(int argc, char** argv){
    char command[PATH_MAX];
    //Checking only one file is input
    if(argc > 2){
        printf("Error: too many arguments at command line.\nOrbital data should be passed to program using a single text file.");
        return -1;
    }

    int n = number_of_bodies(argv[1]);
    body* bodies = read_from_file(argv[1], n);

    printf("\n\n\n-----Welcome to the Gravitational N-Body solver----\nYour input file is %s\n\n", argv[1]);
    printf("-----Please choose a method of analysis-----\nVelocity verlet method | -> Enter 1\nRK4 method employed by GSL | -> Enter 2\n");
    int choice;
    scanf("%i", &choice);

    if(choice == 1){
        velocity_verlet_method(bodies, GNU_DATA, n);
        snprintf(command, sizeof(command), "%s %s", GNU_CALL, GNU_SCRIPT_1);
        system(command);
    }
    if(choice == 2){
        rk4_exe(bodies, n);
        snprintf(command, sizeof(command), "%s %s", GNU_CALL, GNU_SCRIPT_2);
        system(command);
    }
    if(choice != 1 && choice != 2){
        printf("Error: Please choose a valid method\n");
        return 0;
    }
    free(bodies);
    return 0;
}