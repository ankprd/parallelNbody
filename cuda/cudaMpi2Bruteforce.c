/*
** cudaMpi2Bruteforce.c - nbody simulation that implements bruteforce algorithm, using CUDA + MPI v2
**
**/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>

#ifdef DISPLAY
#include "ui.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

#include "../mpi/ui.h"
#include "../mpi/nbody_tools.h"
#include "../mpi/nbody.h"

void all_move_particles(double step, int fPart, int lPart);
void initCuda();
void finalizeCuda();

FILE* f_out=NULL;

int nparticles=10000;      /* number of particles */
float T_FINAL=1.0;     /* simulation end time */
particle_t*particles;

double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

void init() {
  initCuda();
}



#ifdef DISPLAY
Display *theDisplay;  /* These three variables are required to open the */
GC theGC;             /* particle plotting window.  They are externally */
Window theMain;       /* declared in ui.h but are also required here.   */
#endif

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */

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

void run_simulation(int rank, int nbT) {
  double t = 0.0, dt = 0.01;

  int nbPart = nparticles / nbT;
  if(rank < nparticles % nbT)
    nbPart++;
  int fPart = rank * nbPart;
  if(rank >= nparticles % nbT)
    fPart += (nparticles % nbT);
  int lPart = fPart + nbPart;
  //printf("pour rank : %d, fP : %d, lP : %d\n", rank, fPart, lPart);

  double* rcvVal = (double*)malloc(sizeof(double) * nparticles * 2);
  double* sndVal = (double*)malloc(sizeof(double) * nparticles * 2);

  if (rcvVal == 0 || sndVal == 0) {
		printf("Malloc failed in process %d\n", rank);
		return;
  }

  while (t < T_FINAL && nparticles>0) {
    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */ 
    int i, id; 
    all_move_particles(dt, fPart, lPart);
    
    /*on envoie juste la force*/
    for (i = 0; i < nparticles; i++) {
      sndVal[2 * i] = particles[i].x_force;
      sndVal[2 * i + 1] = particles[i].y_force;
    }
    MPI_Allreduce(sndVal, rcvVal, nparticles * 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (i = 0; i < nparticles; i++) {
      particles[i].x_force = rcvVal[2 * i];
      particles[i].y_force = rcvVal[2 * i + 1];
      //printf("Process %d updated particle %d\n", rank, i);
    }

      for(i=0; i< nparticles; i++) {
          move_particle(&particles[i], dt);
      }

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */
    dt = 0.1*max_speed/max_acc;

    /* Plot the movement of the particle */
#ifdef DISPLAY
    if(rank == 0){
        clear_display();
        draw_all_particles();
        flush_display();
    }
#endif
  }
}

/*
  Simulate the movement of nparticles particles.
*/
int main(int argc, char**argv)
{
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

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

  init(nparticles / nbTasks + 1);

  /* Allocate global shared arrays for the particles data set. */
  particles = (particle_t *)malloc(sizeof(particle_t)*nparticles);
  all_init_particles(nparticles, particles);

  if(rank == 0){
      /* Initialize thread data structures */
    #ifdef DISPLAY
      /* Open an X window to display the particles */
      simple_init (100,100,DISPLAY_SIZE, DISPLAY_SIZE);
    #endif

      

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
    finalizeCuda();
    MPI_Finalize(); 

    printf("Cuda + MPI v2\n");
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
      return 0;

  }

  else
    run_simulation(rank, nbTasks);

  finalizeCuda();
  MPI_Finalize(); 
  return 0;
}
