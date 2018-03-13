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
particle_t* newParticles;

node_t *root;


double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

void init() {
  init_alloc(4*nparticles);
  root = malloc(sizeof(node_t));
  init_node(root, NULL, XMIN, XMAX, YMIN, YMAX);
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

/* compute the force that node n acts on particle p */ //AND MOVES IT
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
  } 
  newParticles[idP] = *p;
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
void insert_all_particles(int nparticles, particle_t*particles, node_t*root) {
  int i;
  for(i=0; i<nparticles; i++) {
    insert_particle(&particles[i], root);
  }
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

  double* rcvVal = (double*)malloc(sizeof(double) * nparticles * 5);
	double* sndVal = (double*)malloc(sizeof(double) * nbPart * 5);
  int* nbPPerTask = (int*)malloc(sizeof(int) * nbT + 1);

  if (rcvVal == 0 || sndVal == 0 || nbPPerTask == 0) {
		printf("Malloc failed in process %d\n", rank);
		return;
  }

  int curT;
  nbPPerTask[0] = 0;
  for(curT = 0; curT < nbT; curT++){
    nbPPerTask[curT + 1] = nparticles / nbT;
    if(curT < nparticles % nbT)
      nbPPerTask[curT + 1]++;
    nbPPerTask[curT + 1] *= 5;//car on va s'en servir dans le allGatherV et 5 car on envoie la masse aussi mtn
  }
  /*printf("Pour nbT = %d et nbparts = %d in process %d\n", nbT, nparticles, rank);
  for(curT = 0; curT <= nbT; curT++)
    printf("in rank %d curT : %d -> %d\n", rank, curT, nbPPerTask[curT + 1]);
  printf("\n");*/

  while (t < T_FINAL && nparticles>0) {
    /* Update time. */
    t += dt;

    /*constructs tree*/
    /*init();
    printf("init of node root worked in rank %d\n", rank);
    insert_all_particles(nparticles, particles, root);
    printf("insertion of all parts worked in %d\n", rank);
    fflush(stdout);*/

    /* Calculates forces applied on particles from fPart to lPart*/
    int unused = compute_force_in_node(root, fPart, lPart, 0);

    /* changes their pos/velocity accordingly and saves them in newParticles at pos [fpart, lpart[ */
    unused = move_and_save_particles_in_node(root, dt, fPart, lPart, 0);

    //TODO COMMUNICATION
    int i, id;
		for (i = fPart; i < lPart; i++) {
			id = i - fPart;
			sndVal[5 * id] = newParticles[i].x_pos;
			sndVal[5 * id + 1] = newParticles[i].y_pos;
			sndVal[5 * id + 2] = newParticles[i].x_vel;
			sndVal[5 * id + 3] = newParticles[i].y_vel;
      sndVal[5 * id + 4] = newParticles[i].mass;
		}
    MPI_Allgatherv(sndVal, nbPart * 5, MPI_DOUBLE, rcvVal, nbPPerTask + 1, nbPPerTask, MPI_DOUBLE, MPI_COMM_WORLD);

    for (i = 0; i < nparticles; i++) {
			particles[i].x_pos = rcvVal[5 * i];
			particles[i].y_pos = rcvVal[5 * i + 1];
			particles[i].x_vel = rcvVal[5 * i + 2];
			particles[i].y_vel = rcvVal[5 * i + 3];
      particles[i].mass = rcvVal[5 * i + 4];
			//printf("Process %d updated particle %d\n", rank, i);
		}

    double newMaxSpeed, newMaxAcc;
    //printf("At time %lf Process %d before reduce : max_acc -> %lf max_speed -> %lf\n", t, rank, max_acc, max_speed);	
		MPI_Allreduce(&max_speed, &newMaxSpeed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		//printf("At time %lf after first reduce, process %d, nparticles %d, nbT %d\n", t, rank, nparticles, nbTasks);
		MPI_Allreduce(&max_acc, &newMaxAcc, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    max_acc = newMaxAcc;
    max_speed = newMaxSpeed;

    node_t* new_root = malloc(sizeof(node_t));
    init_node(new_root, NULL, XMIN, XMAX, YMIN, YMAX);
    printf("init of node root worked in rank %d\n", rank);
    fflush(stdout);

    /* then move all particles and return statistics */
    insert_all_particles(nparticles, particles, new_root);
    printf("insertion of all parts worked in %d\n", rank);
    fflush(stdout);

    free_node(root);
    printf("freenod ok in rank %d\n", rank);
    fflush(stdout);
    free(root);
    printf("free root ok in rank %d\n", rank);
    fflush(stdout);
    root = new_root;

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
  }
  free(root);
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
  newParticles = malloc(sizeof(particle_t)*nparticles);
  all_init_particles(nparticles, particles);
  insert_all_particles(nparticles, particles, root);
  //printf("init of all parts worked\n");

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
  print_particles(f_out, root);
  fclose(f_out);
#endif

  printf("-----------------------------\n");
  printf("nparticles: %d\n", nparticles);
  printf("T_FINAL: %f\n", T_FINAL);
  printf("-----------------------------\n");
  printf("Simulation took %lf s to complete\n", duration);

#ifdef DISPLAY
  node_t *n = root;
  clear_display();
  draw_node(n);
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
