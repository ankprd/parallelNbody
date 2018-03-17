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

#include "ui.h"
#include "nbody_tools.h"
#include "nbody.h"

void all_move_particles(double step, int fPart, int lPart);
void initCuda();
void finalizeCuda();

FILE* f_out=NULL;

int nparticles=10;      /* number of particles */
float T_FINAL=1.0;     /* simulation end time */
particle_t*particles;

double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

void init() {
  initCuda();
}

#define DUMP_RESULT

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

  double* rcvVal = (double*)malloc(sizeof(double) * nparticles * 5);
  double* sndVal = (double*)malloc(sizeof(double) * nbPart * 5);
  int* nbPPerTask = (int*)malloc(sizeof(int) * nbT);
  int* offsetTask = (int*)malloc(sizeof(int) * nbT);
  offsetTask[0] = 0;

  if (rcvVal == 0 || sndVal == 0 || nbPPerTask == 0) {
		printf("Malloc failed in process %d\n", rank);
		return;
  }

  int curT;
  nbPPerTask[0] = 0;
  for(curT = 0; curT < nbT; curT++){
    nbPPerTask[curT] = nparticles / nbT;
    if(curT < nparticles % nbT)
      nbPPerTask[curT]++;
    nbPPerTask[curT] *= 5;//car on va s'en servir dans le allGatherV et 5 car on envoie la masse aussi mtn
    if(curT > 0)
      offsetTask[curT] = offsetTask[curT - 1] + nbPPerTask[curT - 1];
  }

  while (t < T_FINAL && nparticles>0) {
    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */ 
    int i, id; 
    for(i=0; i<nparticles; i++) {
      particle_t*p = &particles[i];
      printf("in rank %d particle={pos=(%f,%f), vel=(%f,%f)}\n", rank, p->x_pos, p->y_pos, p->x_vel, p->y_vel);
    }
    all_move_particles(dt, fPart, lPart);
    for(i=fPart; i<lPart; i++) {
        move_particle(&particles[i], dt);
    }

	for (i = fPart; i < lPart; i++) {
		id = i - fPart;
		sndVal[5 * id] = particles[i].x_pos;
		sndVal[5 * id + 1] = particles[i].y_pos;
		sndVal[5 * id + 2] = particles[i].x_vel;
		sndVal[5 * id + 3] = particles[i].y_vel;
        sndVal[5 * id + 4] = particles[i].mass;
	}
    MPI_Allgatherv(sndVal, nbPart * 5, MPI_DOUBLE, rcvVal, nbPPerTask, offsetTask, MPI_DOUBLE, MPI_COMM_WORLD);

    for (i = 0; i < nparticles; i++) {
		particles[i].x_pos = rcvVal[5 * i];
		particles[i].y_pos = rcvVal[5 * i + 1];
		particles[i].x_vel = rcvVal[5 * i + 2];
		particles[i].y_vel = rcvVal[5 * i + 3];
        particles[i].mass = rcvVal[5 * i + 4];
		//printf("Process %d updated particle %d\n", rank, i);
	}

    //printDebug(1, rank);
    //printf("\n\n\n");

    double newMaxSpeed, newMaxAcc;
    //printf("At time %lf Process %d before reduce : max_acc -> %lf max_speed -> %lf\n", t, rank, max_acc, max_speed);	
	MPI_Allreduce(&max_speed, &newMaxSpeed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	//printf("At time %lf after first reduce, process %d, nparticles %d, nbT %d\n", t, rank, nparticles, nbTasks);
	MPI_Allreduce(&max_acc, &newMaxAcc, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    max_acc = newMaxAcc;
    max_speed = newMaxSpeed;
    /*for(i=0; i<nparticles; i++) {
      particle_t*p = &particles[i];
      printf("in main.c after move particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
    }*/

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */
    //printf("max acc : %d\n", max_acc);
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

  finalizeCuda();
  MPI_Finalize(); 
  return 0;
}