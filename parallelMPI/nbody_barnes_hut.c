/*
** nbody_barnes_hut.c - nbody simulation that implements the Barnes-Hut algorithm (O(nlog(n)))
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
#include <string.h>

#ifdef DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

#include "ui.h"
#include "nbody.h"
#include "nbody_tools.h"
#include "nbody_alloc.h"

#define DUMP_RESULT

FILE* f_out=NULL;

int nparticles=10000;      /* number of particles */
float T_FINAL=1.0;     /* simulation end time */
int nbDeletedParts;

particle_t*particles;
particle_t* newParticles;

node_t *root;

extern struct memory_t mem_node;


double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

double timing[2];

void printDebug(int val, int rank){
  int i;
  for (i = 0; i<nparticles; i++) {
		particle_t*p;
    if(val)
      p = &particles[i];
    else
      p = &newParticles[i];
    if(val)
      printf("apres send ");
    else
      printf("avant send ");
		printf("Rank : %d : particle={pos=(%f,%f), vel=(%f,%f) et mass %f}\n", rank, p->x_pos, p->y_pos, p->x_vel, p->y_vel, p->mass);
	}
}

void init() {
  init_alloc(4*nparticles);//may cause segfault : if particles get really close, the tree may get bigger than that
  root = malloc(sizeof(node_t));
  init_node(root, NULL, XMIN, XMAX, YMIN, YMAX);
}

#ifdef DISPLAY
Display *theDisplay;  /* These three variables are required to open the */
GC theGC;             /* particle plotting window.  They are externally */
Window theMain;       /* declared in ui.h but are also required here.   */
#endif

void print_all_particles(FILE* f) {
	int i;
	for (i = 0; i<nparticles; i++) {
		particle_t*p = &particles[i];
		fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
	}
}

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

/* compute the force that node n acts on particle p  -> unchanged from sequential*/
void compute_force_on_particle(node_t* n, particle_t *p) {
  if(! n || n->n_particles==0) {
    return;
  }
  if(n->particle) {
    /* only one particle */
    assert(n->children == NULL);

    /*
      If the current node is an external node (and it is not body b),
      calculate the force exerted by the current node on b, and add
      this amount to b's net force.
    */
    compute_force(p, n->x_center, n->y_center, n->mass);
  } else {
    /* There are multiple particles */

    #define THRESHOLD 2
    double size = n->x_max - n->x_min; // width of n
    double diff_x = n->x_center - p->x_pos;
    double diff_y = n->y_center - p->y_pos;
    double distance = sqrt(diff_x*diff_x + diff_y*diff_y);

#if BRUTE_FORCE
    /*
      Run the procedure recursively on each of the current
      node's children.
      --> This result in a brute-force computation (complexity: O(n*n))
    */
    int i;
    for(i=0; i<4; i++) {
      compute_force_on_particle(&n->children[i], p);
    }
#else
    /* Use the Barnes-Hut algorithm to get an approximation */
    if(size / distance < THRESHOLD) {
      /*
	The particle is far away. Use an approximation of the force
      */
      compute_force(p, n->x_center, n->y_center, n->mass);
    } else {
      /*
        Otherwise, run the procedure recursively on each of the current
	node's children.
      */
      int i;
      for(i=0; i<4; i++) {
	compute_force_on_particle(&n->children[i], p);
      }
    }
#endif
  }
}

/* calculates force on particle iff it is in [fPart, lPart[ */
int compute_force_in_node(node_t *n, int fPart, int lPart, int idP) {
  if(!n) return 0;

  if(n->particle) {
    if(fPart <= idP && idP < lPart){
      particle_t*p = n->particle;
      p->x_force = 0;
      p->y_force = 0;
      compute_force_on_particle(root, p);
    }
    return 1;
  }

  if(idP >= lPart || idP + n->n_particles < fPart)
    return n->n_particles;

  if(n->children) {
    int i;
    int nbPasses = 0;
    for(i=0; i<4; i++) {
      int temp = compute_force_in_node(&n->children[i], fPart, lPart, idP + nbPasses);
      nbPasses += temp;
    }
  }
  return n->n_particles;
}

void move_and_save_particle(particle_t*p, double step, int idP) {
  if(p->mass >= 0.5){//otherwise particle was marked as deleted, and won't be printed or anything, so let's just not move it around
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
  p->node = NULL;
  if(p->x_pos < XMIN ||
     p->x_pos > XMAX ||
     p->y_pos < YMIN ||
     p->y_pos > YMAX) {
    p->mass = 0;//marks the particle as deleted
    nbDeletedParts++;
  } 
  memcpy(&newParticles[idP], p, sizeof(particle_t));
}

/* compute the new position of the particles in a node */
int move_and_save_particles_in_node(node_t*n, double step, int fPart, int lPart, int idP) {
  if(!n) return 0;

  if(n->particle) {
    if(fPart <= idP && idP < lPart){
      particle_t*p = n->particle;
      move_and_save_particle(p, step, idP);
    }
  }

  if(idP >= lPart || idP + n->n_particles < fPart)
    return n->n_particles;

  if(n->children) {
    int i;
    int nbPasses = 0;
    for(i=0; i<4; i++) {
      int temp = move_and_save_particles_in_node(&n->children[i], step, fPart, lPart, idP + nbPasses);
      nbPasses += temp;
    }
  }
  return n->n_particles;
}

/* create a quad-tree from an array of particles */
void insert_all_particles(int nparticles, particle_t*particles, node_t*Nroot, int rank) {
  int i;
  for(i=0; i<nparticles; i++) {
    if(particles[i].mass >= 0.5){
      insert_particle(&particles[i], Nroot);}
  }
}

int run_simulation(int rank, int nbT) {
  int idIter = 0;
  double t = 0.0, dt = 0.01;
  int nbpartChanged = 1;

  /*Tables used to send data between threads*/
  int* nbPPerTask = (int*)malloc(sizeof(int) * nbT);
  int* offsetTask = (int*)malloc(sizeof(int) * nbT);
  offsetTask[0] = 0;
  int nbPart, fPart, lPart;
  nbPart = nparticles / nbT + 1;

  double* rcvVal = (double*)malloc(sizeof(double) * nparticles * 5);
  double* sndVal = (double*)malloc(sizeof(double) * nbPart * 5);

  if (rcvVal == 0 || sndVal == 0 || nbPPerTask == 0) {
		printf("Malloc failed in process %d\n", rank);
		return -1;
  }
  double t1Calc, t2Calc; 
  t1Calc = MPI_Wtime(); 

  while (t < T_FINAL && nparticles>0) {
    if(nbpartChanged){
      /*calculates how many particles for all the threads + first particle of each thread*/
      int curT;
      nbPPerTask[0] = 0;
      for(curT = 0; curT < nbT; curT++){
        nbPPerTask[curT] = nparticles / nbT;
        if(curT < nparticles % nbT)
          nbPPerTask[curT]++;
        nbPPerTask[curT] *= 5;//to be used directly in allgatherv : 5 values for each particle : pos x pos y, vel x, vel y, mass (mass is set to zero if particles is out)
        if(curT > 0)
          offsetTask[curT] = offsetTask[curT - 1] + nbPPerTask[curT - 1];
      }

      fPart = offsetTask[rank] / 5;
      nbPart = nbPPerTask[rank] / 5;
      lPart = fPart + nbPart;

      nbpartChanged = 0;
    }
    //printf("Rank : %d\n", rank);
    nbDeletedParts = 0; //to share between threads how many particles were deleted

    /* Update time. */
    t += dt;

    /* Calculates forces applied on particles from fPart to lPart*/
    int unused = compute_force_in_node(root, fPart, lPart, 0);

    /* changes their pos/velocity accordingly and saves them in newParticles at pos [fpart, lpart[ */
    unused = move_and_save_particles_in_node(root, dt, fPart, lPart, 0);

    //printDebug(0, rank);
    t2Calc = MPI_Wtime(); 
    timing[0] += t2Calc - t1Calc;

    int i, id;
		for (i = fPart; i < lPart; i++) {
			id = i - fPart;
			sndVal[5 * id] = newParticles[i].x_pos;
			sndVal[5 * id + 1] = newParticles[i].y_pos;
			sndVal[5 * id + 2] = newParticles[i].x_vel;
			sndVal[5 * id + 3] = newParticles[i].y_vel;
      sndVal[5 * id + 4] = newParticles[i].mass;
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

    double newMaxSpeed, newMaxAcc;
    int totPartsDel = 0;
    //printf("At time %lf Process %d before reduce : max_acc -> %lf max_speed -> %lf\n", t, rank, max_acc, max_speed);	
		MPI_Allreduce(&max_speed, &newMaxSpeed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		//printf("At time %lf after first reduce, process %d, nparticles %d, nbT %d\n", t, rank, nparticles, nbTasks);
		MPI_Allreduce(&max_acc, &newMaxAcc, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(&nbDeletedParts, &totPartsDel, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    t1Calc = MPI_Wtime(); 
    //MPI_Barrier(MPI_COMM_WORLD);
    timing[1] += t1Calc - t2Calc;

    max_acc = newMaxAcc;
    max_speed = newMaxSpeed;

    node_t* new_root = malloc(sizeof(node_t));
    init_node(new_root, NULL, XMIN, XMAX, YMIN, YMAX);

    /* then move all particles and return statistics */
    insert_all_particles(nparticles, particles, new_root, rank);
    //printf("insertion of all parts worked in %d\n", rank);
    //fflush(stdout);

    //printf("Rank : %d Before free : %d\n", rank, mem_node.nb_free);
    free_node(root);
    free(root);
    //printf("Rank : %d after free : %d\n", rank, mem_node.nb_free);
    root = new_root;

    if(totPartsDel > 0){
      printf("Nb parts deleted : %d => newNbParts = %d\n", totPartsDel, nparticles);
      nbpartChanged = 1;
    }
    nparticles -= totPartsDel;

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */

    dt = 0.1*max_speed/max_acc;
    /* Plot the movement of the particle */
#if DISPLAY
    if(rank == 0){
    node_t *n = root;
    clear_display();
    draw_node(n);
    flush_display();
    }
#endif
  idIter++;
  }

  free_node(root);
  free(root);
  free(nbPPerTask);
  free(offsetTask);
  free(rcvVal);
  free(sndVal);

  return idIter;
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

  int nbIter;

  /* Allocate global shared arrays for the particles data set. */
  particles = malloc(sizeof(particle_t)*nparticles);
  newParticles = malloc(sizeof(particle_t)*nparticles);
  all_init_particles(nparticles, particles);
  insert_all_particles(nparticles, particles, root, rank);

if(rank == 0){
    printf("nbTasks : %d\n", nbTasks);
  /* Initialize thread data structures */
#ifdef DISPLAY
  /* Open an X window to display the particles */
  simple_init (100,100,DISPLAY_SIZE, DISPLAY_SIZE);
#endif

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  /* Main thread starts simulation ... */
  nbIter = run_simulation(rank, nbTasks);

  gettimeofday(&t2, NULL);

  double duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

  
#ifdef DUMP_RESULT
  FILE* f_out = fopen("particles.log", "w");
	assert(f_out);
	print_all_particles(f_out);
	fclose(f_out);
#endif
  printf("Barnes-hut MPI\n");
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
    nbIter = run_simulation(rank, nbTasks);

  /*
  //Used to calculate average calculation time per rank
  char nomF[4] = "i.t";
  nomF[0] = rank + '0';
  //printf("%c", rank - '0');
  FILE* f_out3 = fopen(nomF, "w");
  fprintf(f_out3, "%f %f\n", timing[0] / (double) nbIter, timing[1] / (double) nbIter);
	fclose(f_out3);
  */

  free(particles);
  free(newParticles);

  MPI_Finalize(); 
  return 0;
}
