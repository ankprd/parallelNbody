#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "nbody.h"

// CUDA runtime
#include <cuda_runtime.h>

// includes
#include <helper_functions.h> // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>	  // helper functions for CUDA error checking and initialization

extern "C"
{
#include <cuda.h>
}
#define MEMSIZE 30

//int nparticles;
extern particle_t *particles;

particle_t  *d_particles, *d_nparticles;

__device__ void compute_force(particle_t*p, double x_pos, double y_pos, double mass) {
  double x_sep, y_sep, dist_sq, grav_base;

  x_sep = x_pos - p->x_pos;
  y_sep = y_pos - p->y_pos;
  dist_sq = MAX((x_sep*x_sep) + (y_sep*y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  grav_base = GRAV_CONSTANT*(p->mass)*(mass)/dist_sq;

  p->x_force += grav_base*x_sep;
  p->y_force += grav_base*y_sep;
}

__global__ void calcForce(particle_t *d_particles, particle_t *d_nparticles, int d_nbP)
{
  int i, j;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d_nbP){
    d_nparticles[i].x_force = 0;
    d_nparticles[i].y_force = 0;
    //printf("calculating for part %d \n", i);
    for(j = 0; j < d_nbP; j++)
        compute_force(&d_nparticles[i], d_particles[j].x_pos, d_particles[j].y_pos, d_particles[j].mass);//on modifie nparticle, et on prend les infos de d_particle
  }
}

extern "C" void initCuda(){
  cudaMalloc((void**)&d_particles, nparticles * sizeof(particle_t));
  cudaMalloc((void**)&d_nparticles, nparticles * sizeof(particle_t));
}

extern "C" void finalizeCuda(){
  cudaFree(d_particles);
  cudaFree(d_nparticles);
}

extern "C" void all_move_particles(double step)
{
  //nparticles = nbParts;
  //printf("Nb parts in gpu : %d\n", nparticles);
  int i;
  /*for(i=0; i<nparticles; i++) {
    particle_t*p = &particles[i];
    printf("in .cu particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }*/

  cudaMemcpy(d_particles, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nparticles, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice);

  calcForce<<<1000000, 10>>>(d_particles, d_nparticles, nparticles);

  /*for(i=0; i<nparticles; i++) {
    particle_t*p = &particles[i];
    printf("in .cu particle={pos=(%f,%f), vel=(%f,%f), force=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel, p->x_force, p->y_force);
  }*/

  cudaMemcpy(particles, d_nparticles, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost);
  /*for(i=0; i<nparticles; i++) {
    particle_t*p = &particles[i];
    printf("in .cu after calc particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
  }*/
}

