/*
** nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
**
**/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>
#include <mpi.h> 

#ifdef DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

#include "ui.h"
#include "nbody.h"
#include "nbody_tools.h"

FILE* f_out=NULL;

int nparticles=10;      /* number of particles */
float T_FINAL=1.0;     /* simulation end time */
particle_t*particles;

double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

void init() {
  /* Nothing to do */
}

#ifdef DISPLAY
Display *theDisplay;  /* These three variables are required to open the */
GC theGC;             /* particle plotting window.  They are externally */
Window theMain;       /* declared in ui.h but are also required here.   */
#endif

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force(particle_t*p, double x_pos, double y_pos, double mass) {
  double x_sep, y_sep, dist_sq, grav_base;

  x_sep = x_pos - p->x_pos;
  y_sep = y_pos - p->y_pos;
  dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  grav_base = GRAV_CONSTANT*(p->mass)*(mass)/dist_sq;

  p->x_force += grav_base*x_sep;
  p->y_force += grav_base*y_sep;
}

/* compute the new position/velocity */
void move_particle(particle_t*p, double step) {

  p->x_pos += (p->x_vel)*step;
  p->y_pos += (p->y_vel)*step;
  double x_acc = p->x_force/p->mass;
  double y_acc = p->y_force/p->mass;
  p->x_vel += x_acc*step;
  p->y_vel += y_acc*step;

  /* compute statistics */
  double cur_acc = (x_acc*x_acc + y_acc*y_acc);
  cur_acc = sqrt(cur_acc);
  double speed_sq = (p->x_vel)*(p->x_vel) + (p->y_vel)*(p->y_vel);
  double cur_speed = sqrt(speed_sq);

  sum_speed_sq += speed_sq;
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);
}


/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles(double step, int firstPart, int lastPart)
{
  /* First calculate force for particles. */
	int i;
  for(i=firstPart; i<lastPart; i++) {
    int j;
    particles[i].x_force = 0;
    particles[i].y_force = 0;
    for(j=0; j<nparticles; j++) {
      particle_t*p = &particles[j];
			/* compute the force of particle j on particle i */
      compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
			//printf("Pour la part : %d, force x : %lf, force y : %lf, part in range [%d,  %d]\n", i, particles[i].x_force, particles[i].y_force, firstPart, lastPart);
    }
  }

  /* then move all particles and return statistics */
  for(i=firstPart; i<lastPart; i++) {
    move_particle(&particles[i], step);
		//printf("Pour la part : %d, pos x : %lf, pos y : %lf, vel x %lf, vel y %lf\n", i, particles[i].x_pos, particles[i].y_pos, particles[i].x_vel, particles[i].y_vel);
  }
}

/* display all the particles */
void draw_all_particles() {
  int i;
  for(i=0; i<nparticles; i++) {
    int x = POS_TO_SCREEN(particles[i].x_pos);
    int y = POS_TO_SCREEN(particles[i].y_pos);
    draw_point (x,y);
  }
}

void print_all_particles(FILE* f) {
  int i;
  for(i=0; i<nparticles; i++) {
    particle_t*p = &particles[i];
    fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }
}

void debugPrint(int process, double time){
  int i;
  for(i = 0; i < nparticles; i++){
    particle_t *p = &particles[i];
    printf("P %d t %lf : particle %d ={pos=(%f,%f), vel=(%f,%f)}\n", process, time, i, p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }
}

void run_simulation(int rank, int nbTasks) {
  //int nbTasks;
  //MPI_Comm_size(MPI_COMM_WORLD, &nbTasks);
  //printf("nbTasks : %d\n\n", nbTasks);

  int firstPart = (nparticles / nbTasks) * rank;
  int lastPart = (nparticles / nbTasks) * (rank + 1);
  if(rank == nbTasks - 1)
    lastPart = nparticles;
  int maxNbParts = nparticles / nbTasks;
  if(nparticles - (nparticles / nbTasks) * (nbTasks - 1) > maxNbParts)
    maxNbParts = nparticles - (nparticles / nbTasks) * (nbTasks - 1);

  double t = 0.0, dt = 0.01;
  int idIter = 0;
  while (t < T_FINAL && nparticles>0) {
    //printf("\n\n\nTime : %lf and process %d\n", t, rank);
    /*if(idIter < 2){
        debugPrint(rank, t);
        //printf("IDiTER : % d, rank %d\n", idIter, rank);
    }*/
    //printf("in process %d, at time %lf, npart %d\n", rank, t, nparticles);
		/* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */
    all_move_particles(dt, firstPart, lastPart);

    /*if(idIter < 2){
        debugPrint(rank, t + 1);
        //printf("IDiTER : % d, rank %d\n", idIter, rank);
    }
    idIter++;*/
    //printf("At time %lf, Process %d finished moving particles from %d to %d \n", t, rank, firstPart, lastPart);
		//MPI_Barrier(MPI_COMM_WORLD);
    //Broadcast results -> new parts pos and speed -> envoyer un tableau de 4 * nbParts concernees qui contient pos et vel a la suite
    double* valsBroad = (double*)malloc(sizeof(double) * maxNbParts * 4);
    if(valsBroad == 0){
      printf("Malloc failed in process %d\n", rank);
      return;
    }
    //printf("At ti;e %lf, Process %d allocated %d space\n", t, rank, maxNbParts);
    int i;
    for(i = 0; i < nbTasks; i++){
      int nbSent = nparticles / nbTasks;
      if(i == nbTasks - 1){
        nbSent = nparticles - (nparticles / nbTasks) * (nbTasks - 1);
      	//printf("Changed val nbSent for process : %d -> %d \n", i, nbSent);	
			}
			//printf("At time %lf, process %d, nparticles %d, nbT %d, nbSent %d\n", t, rank, nparticles, nbTasks, nbSent);
			//printf("At time %lf, In process %d, broadcast with source %d, sending %d\n", t, rank, i, nbSent);
      if(i == rank){//fills data to send
        int j;
        for(j = firstPart; j < lastPart; j++){
          int id = j - firstPart;
          valsBroad[4 * id] = particles[j].x_pos;
          valsBroad[4 * id + 1] = particles[j].y_pos;
          valsBroad[4 * id + 2] = particles[j].x_vel;
          valsBroad[4 * id + 3] = particles[j].y_vel;
        }      
      }
      MPI_Bcast(valsBroad, nbSent * 4, MPI_DOUBLE, i, MPI_COMM_WORLD);
      //printf("At time %lf, rocess %d received/sent broadcast from process %d for parts [%d, %d]\n", t, rank, i, rank * (nparticles / nbT), rank * (nparticles / nbT) + nbSent);
      if(i != rank){//copy received data
        int j;
        int fP = i * (nparticles / nbTasks);
        for(j = fP; j < fP + nbSent; j++){
          int id = j - fP;
          particles[j].x_pos = valsBroad[4 * id];
          particles[j].y_pos = valsBroad[4 * id + 1];
          particles[j].x_vel = valsBroad[4 * id + 2];
          particles[j].y_vel = valsBroad[4 * id + 3];
					//printf("Process %d received from process %d pour la part %d at time %lf : x_pos : %lf, y_pos %lf\n", rank, i, j, t, particles[j].x_pos, particles[j].y_pos);
        }    
      }
    }
		//printf("At time %lf after broadcast, process %d, nparticles %d, nbT %d\n", t, rank, nparticles, nbTasks);
		
    double newMaxSpeed, newMaxAcc;
    //printf("At time %lf Process %d before reduce : max_acc -> %lf max_speed -> %lf\n", t, rank, max_acc, max_speed);	
    MPI_Allreduce(&max_speed, &newMaxSpeed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		//printf("At time %lf after first reduce, process %d, nparticles %d, nbT %d\n", t, rank, nparticles, nbTasks);
    MPI_Allreduce(&max_acc, &newMaxAcc, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    max_acc = newMaxAcc;
    max_speed = newMaxSpeed;
    //printf("At time %lf Process %d finished reduce : max_acc -> %lf max_speed -> %lf\n", t, rank, max_acc, max_speed);	
		//printf("At time %lf after all reduce, process %d, nparticles %d, nbT %d\n", t, rank, nparticles, nbTasks);
		
    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */

    dt = 0.1*max_speed/max_acc;

    if(rank == 0){
      /* Plot the movement of the particle */
      #if DISPLAY
          clear_display();
          draw_all_particles();
          flush_display();
      #endif
    }
  }
}

/*
  Simulate the movement of nparticles particles.
*/
int main(int argc, char**argv)
{
  MPI_Init(&argc, &argv); 
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int nbTasks;
	MPI_Comm_size(MPI_COMM_WORLD, &nbTasks);
  if(argc >= 2) {
    nparticles = atoi(argv[1]);
  }
  if(argc == 3) {
    T_FINAL = atof(argv[2]);
  }

  init();

  /* Allocate global shared arrays for the particles data set. */
  particles = malloc(sizeof(particle_t)*nparticles);
  all_init_particles(nparticles, particles);

  int i;
  for(i = 0; i < nparticles; i++){
    //printf("In process %d, part %i has x %lfm y %lf, vx %lf, vy %lf\n", rank, i, particles[i].x_pos, particles[i].y_pos, particles[i].x_vel, particles[i].y_vel);
  }

  if(rank == 0){
    /* Initialize thread data structures */
    #ifdef DISPLAY
      /* Open an X window to display the particles */
      simple_init (100,100,DISPLAY_SIZE, DISPLAY_SIZE);
    #endif

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    /* Main thread starts simulation ... */
    run_simulation(rank, nbTasks);

    gettimeofday(&t2, NULL);

    double duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

    #ifdef DUMP_RESULT
      FILE* f_out = fopen("particles.log", "w");
      assert(f_out);
      print_all_particles(f_out);
      fclose(f_out);
    #endif

      printf("-----------------------------\n");
      printf("nparticles: %d\n", nparticles);
      printf("T_FINAL: %f\n", T_FINAL);
      printf("-----------------------------\n");
      printf("Simulation took %lf s to complete\n", duration);

    #ifdef DISPLAY
      clear_display();
      draw_all_particles();
      flush_display();

      printf("Hit return to close the window.");

      getchar();
      /* Close the X window used to display the particles */
      XCloseDisplay(theDisplay);
    #endif
  }

  else
    run_simulation(rank, nbTasks);
	MPI_Finalize(); 
  return 0;
}
