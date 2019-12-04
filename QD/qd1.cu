/*******************************************************************************
Quantum dynamics (QD) simulation of an electron in one dimension.

USAGE

%cc -o qd1 qd1.c -lm
%qd1 (input file qd1.in required; see qd1.h for the input format)
*******************************************************************************/
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <cuda.h>
#include "qd1.h"

int nprocs;
int myid;
double dbuf[NX],dbufr[NX];

#define BLOCKDIM 192
int nx;
int dim;

double psi_2[NX+2][2];
void host2device(double host[NX+2][2], double* dev, double* tmp, int offset) {
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j <= 1; ++j)
      tmp[2*i+j] = host[offset+i][j]; 
  //printf("offset%d\n", offset);
  cudaMemcpy(dev, tmp, 2*dim*sizeof(double), cudaMemcpyHostToDevice);
}

void device2host(double* dev, double host[NX+2][2], double* tmp, int offset) {
  cudaMemcpy(tmp, dev, 2*dim*sizeof(double), cudaMemcpyDeviceToHost);
  for (int i = 1; i <= nx; ++i)
    for (int j = 0; j <= 1; ++j)
      host[offset+i][j] = tmp[2*i+j]; 
}

int main(int argc, char **argv) {
  
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);


	init_param();  /* Read input parameters */
	init_prop();   /* Initialize the kinetic & potential propagators */
	init_wavefn(); /* Initialize the electron wave function */

  int nthreads = 2;
  int ndevices = 2;
  omp_set_num_threads(nthreads);
  #pragma omp parallel
  {
  int mytid = omp_get_thread_num();
  cudaSetDevice(mytid%ndevices);
  int dev_used;
  cudaGetDevice(&dev_used);
  //printf("myid = %d; mytid = %d: device used = %d\n", myid, mytid, dev_used);

  double* dev_psi;
  double* dev_wrk;
  double* dev_al0;
  double* dev_al1;
  double* dev_bux0;
  double* dev_bux1;
  double* dev_blx0;
  double* dev_blx1;
  double* dev_u;
  
  nx = NX/nthreads;
  dim = nx + 2;
  cudaMalloc(&dev_psi, 2*dim*sizeof(double));
  cudaMalloc(&dev_wrk, 2*dim*sizeof(double));
  cudaMalloc(&dev_al0, 2*sizeof(double));
  cudaMalloc(&dev_al1, 2*sizeof(double));
  cudaMalloc(&dev_bux0, 2*dim*sizeof(double));
  cudaMalloc(&dev_bux1, 2*dim*sizeof(double));
  cudaMalloc(&dev_blx0, 2*dim*sizeof(double));
  cudaMalloc(&dev_blx1, 2*dim*sizeof(double));
  cudaMalloc(&dev_u, 2*dim*sizeof(double));

  int offset = nx*mytid;
  double* tmp = (double*)malloc(2*dim*sizeof(double));
  host2device(psi, dev_psi, tmp, offset);
  //cudaMemcpy(psi + 2*nx*mytid, dev_psi, 2*dim*sizeof(double), cudaMemcpyHostToDevice);
  host2device(wrk, dev_wrk, tmp, offset);
  //cudaMemcpy(wrk + 2*nx*mytid, dev_wrk, 2*dim*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_al0, al[0], 2*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_al1, al[1], 2*sizeof(double), cudaMemcpyHostToDevice);
  host2device(bux[0], dev_bux0, tmp, offset);
  host2device(bux[1], dev_bux1, tmp, offset);
  host2device(blx[0], dev_blx0, tmp, offset);
  host2device(blx[1], dev_blx1, tmp, offset);
  //cudaMemcpy(bux + 2*nx*mytid, dev_bux, 2*dim*sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(bux + 2*nx*mytid + 2*(NX+2), dev_bux+2*dim, 2*dim*sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(blx + 2*nx*mytid, dev_blx, 2*dim*sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(blx + 2*nx*mytid + 2*(NX+2), dev_blx+2*dim, 2*dim*sizeof(double), cudaMemcpyHostToDevice);
  host2device(u, dev_u, tmp, offset);
  //cudaMemcpy(u   + 2*nx*mytid, dev_u,   2*dim*sizeof(double), cudaMemcpyHostToDevice);

	int step; /* Simulation loop iteration index */
	for (step=1; step<=NSTEP; step++) {
		single_step(dev_psi, dev_wrk, dev_al0, dev_al1, dev_bux0, dev_bux1, dev_blx0, dev_blx1, dev_u, tmp, offset); /* Time propagation for one step, DT */
    {
		if (step%NECAL==0) {
      #pragma omp master
			calc_energy();
      #pragma omp barrier
      if (myid == 0) {
			  printf("%le %le %le %le\n",DT*step,ekin,epot,etot);
      }
    }
    }
	}
  
  cudaFree(dev_psi);
  cudaFree(dev_wrk);
  cudaFree(dev_al0);
  cudaFree(dev_al1);
  cudaFree(dev_bux0);
  cudaFree(dev_bux1);
  cudaFree(dev_blx0);
  cudaFree(dev_blx1);
  cudaFree(dev_u);

  free(tmp);
  }

  MPI_Finalize();

	return 0;
}

/*----------------------------------------------------------------------------*/
void init_param() {
/*------------------------------------------------------------------------------
	Initializes parameters by reading them from input file.
------------------------------------------------------------------------------*/
	FILE *fp;

	/* Read control parameters */
	fp = fopen("qd1.in","r");
	fscanf(fp,"%le",&LX);
	fscanf(fp,"%le",&DT);
	fscanf(fp,"%d",&NSTEP);
	fscanf(fp,"%d",&NECAL);
	fscanf(fp,"%le%le%le",&X0,&S0,&E0);
	fscanf(fp,"%le%le",&BH,&BW);
	fscanf(fp,"%le",&EH);
	fclose(fp);

	/* Calculate the mesh size */
	dx = LX/NX;
}

/*----------------------------------------------------------------------------*/
void init_prop() {
/*------------------------------------------------------------------------------
	Initializes the kinetic & potential propagators.
------------------------------------------------------------------------------*/
	int stp,s,i,up,lw;
	double a,exp_p[2],ep[2],em[2];
	double x;

	/* Set up kinetic propagators */
	a = 0.5/(dx*dx);

	for (stp=0; stp<2; stp++) { /* Loop over half & full steps */
		exp_p[0] = cos(-(stp+1)*DT*a);
		exp_p[1] = sin(-(stp+1)*DT*a);
		ep[0] = 0.5*(1.0+exp_p[0]);
		ep[1] = 0.5*exp_p[1];
		em[0] = 0.5*(1.0-exp_p[0]);
		em[1] = -0.5*exp_p[1];

		/* Diagonal propagator */
		for (s=0; s<2; s++) al[stp][s] = ep[s];

		/* Upper & lower subdiagonal propagators */
		for (i=1; i<=NX; i++) { /* Loop over mesh points */
			if (stp==0) { /* Half-step */
				up = i%2;     /* Odd mesh point has upper off-diagonal */
				lw = (i+1)%2; /* Even               lower              */
			}
			else { /* Full step */
				up = (i+1)%2; /* Even mesh point has upper off-diagonal */
				lw = i%2;     /* Odd                 lower              */
			}
			for (s=0; s<2; s++) {
				bux[stp][i][s] = up*em[s];
				blx[stp][i][s] = lw*em[s];
			}
		} /* Endfor mesh points, i */
	} /* Endfor half & full steps, stp */

	/* Set up potential propagator */
	for (i=1; i<=NX; i++) {
		x = dx*i + myid*LX;;
		/* Construct the edge potential */
		if (i==1 || i==NX)
			v[i] = EH;
		/* Construct the barrier potential */
		else if (0.5*(LX*nprocs-BW)<x && x<0.5*(LX*nprocs+BW))
			v[i] = BH;
		else
			v[i] = 0.0;
		/* Half-step potential propagator */
		u[i][0] = cos(-0.5*DT*v[i]);
		u[i][1] = sin(-0.5*DT*v[i]);
	}
}

/*----------------------------------------------------------------------------*/
void init_wavefn() {
/*------------------------------------------------------------------------------
	Initializes the wave function as a traveling Gaussian wave packet.
------------------------------------------------------------------------------*/
	int sx,s;
	double x,gauss,psisq,norm_fac;

	/* Calculate the the wave function value mesh point-by-point */
	for (sx=1; sx<=NX; sx++) {
		x = dx*sx-X0 + myid*LX;
		gauss = exp(-0.25*x*x/(S0*S0));
		psi[sx][0] = gauss*cos(sqrt(2.0*E0)*x);
		psi[sx][1] = gauss*sin(sqrt(2.0*E0)*x);
	}

	/* Normalize the wave function */
	psisq=0.0;
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<2; s++)
			psisq += psi[sx][s]*psi[sx][s];
	psisq *= dx;
  MPI_Allreduce(&psisq, &psisq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	norm_fac = 1.0/sqrt(psisq);
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<2; s++)
			psi[sx][s] *= norm_fac;
}

/*----------------------------------------------------------------------------*/
void single_step(double* dev_psi, double* dev_wrk, double* dev_al0, double* dev_al1, double* dev_bux0, double* dev_bux1, double* dev_blx0, double* dev_blx1, double* dev_u, double* tmp, int offset) {
/*------------------------------------------------------------------------------
	Propagates the electron wave function for a unit time step, DT.
------------------------------------------------------------------------------*/
	pot_prop(dev_psi, dev_u, tmp, offset);  /* half step potential propagation */

  kin_prop(0, dev_psi, dev_wrk, dev_al0, dev_bux0, dev_blx0, tmp, offset); /* half step kinetic propagation   */
	kin_prop(1, dev_psi, dev_wrk, dev_al1, dev_bux1, dev_blx1, tmp, offset); /* full                            */
	kin_prop(0, dev_psi, dev_wrk, dev_al0, dev_bux0, dev_blx0, tmp, offset); /* half                            */

	pot_prop(dev_psi, dev_u, tmp, offset);  /* half step potential propagation */
}

void check(int i, double a, double b) {
  if (abs(a - b) > 1e-6) {
    fprintf(stderr, "[%d]: %lf %lf\n", i, a, b); 
  }
}

__global__ void gpu_pot_prop(double* psi, double* u, int nx) {
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx >= 1 && idx <= nx) {
    idx = 2*idx;
    double wr = u[idx] * psi[idx] - u[idx+1] * psi[idx+1];
    double wi = u[idx] * psi[idx+1] + u[idx+1] * psi[idx];
    psi[idx] = wr;
    psi[idx+1] = wi;
  }
}

/*----------------------------------------------------------------------------*/
void pot_prop(double* dev_psi, double* dev_u, double* tmp, int offset) {
/*------------------------------------------------------------------------------
	Potential propagator for a half time step, DT/2.
------------------------------------------------------------------------------*/
  host2device(psi, dev_psi, tmp, offset);
  //cudaMemcpy(psi + 2*nx*mytid, dev_psi, 2*dim*sizeof(double), cudaMemcpyHostToDevice);
  
  dim3 dimGrid((dim+BLOCKDIM-1)/BLOCKDIM,1,1); // Grid dimensions (only use 1D)
  dim3 dimBlock(BLOCKDIM,1,1); // Block dimensions (only use 1D)
  gpu_pot_prop<<<dimGrid,dimBlock>>>(dev_psi, dev_u, nx);

#if 1
  device2host(dev_psi, psi, tmp, offset);
  //cudaMemcpy(dev_psi, psi_2 + 2*(nx+1)*mytid, 2*(nx+1)*sizeof(double), cudaMemcpyDeviceToHost);
#else
  device2host(dev_psi, psi_2, tmp, offset);
#pragma omp barrier
#pragma omp master
{
  int sx;
	double wr,wi;
	for (sx=1; sx<=NX; sx++) {
		wr=u[sx][0]*psi[sx][0]-u[sx][1]*psi[sx][1];
		wi=u[sx][0]*psi[sx][1]+u[sx][1]*psi[sx][0];
		psi[sx][0]=wr;
		psi[sx][1]=wi;
    check(sx, psi[sx][0], psi_2[sx][0]);
    check(sx, psi[sx1[1], psi_2[sx][1]);
	}
}
#endif
#pragma omp barrier
}

__global__ void gpu_kin_prop(double* psi, double* wrk, double* al, double* bux, double* blx, int nx) {
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx >= 1 && idx <= nx) {
    idx = 2*idx;
    double wr = al[0] * psi[idx] - al[1] * psi[idx+1];
    double wi = al[0] * psi[idx+1] + al[1] * psi[idx];
    wr += blx[idx] * psi[idx-2] - blx[idx+1] * psi[idx-1];
    wi += blx[idx] * psi[idx-1] + blx[idx+1] * psi[idx-2];
    wr += bux[idx] * psi[idx+2] - bux[idx+1] * psi[idx+3];
    wi += bux[idx] * psi[idx+3] + bux[idx+1] * psi[idx+2];
    wrk[idx] = wr;
    wrk[idx+1] = wi;
  }
}


/*----------------------------------------------------------------------------*/
void kin_prop(int t, double* dev_psi, double* dev_wrk, double* dev_al, double* dev_bux, double* dev_blx, double* tmp, int offset) {
/*------------------------------------------------------------------------------
	Kinetic propagation for t (=0 for DT/2--half; 1 for DT--full) step.
-------------------------------------------------------------------------------*/
	/* Apply the periodic boundary condition */
  #pragma omp master
	periodic_bc();
  #pragma omp barrier

  host2device(psi, dev_psi, tmp, offset);
  //cudaMemcpy(psi + nx*mytid, dev_psi, 2*dim*sizeof(double), cudaMemcpyHostToDevice);
  
  dim3 dimGrid((dim+BLOCKDIM-1)/BLOCKDIM,1,1); // Grid dimensions (only use 1D)
  dim3 dimBlock(BLOCKDIM,1,1); // Block dimensions (only use 1D)
  gpu_kin_prop<<<dimGrid,dimBlock>>>(dev_psi, dev_wrk, dev_al, dev_bux, dev_blx, nx);

#if 1
  device2host(dev_wrk, psi, tmp, offset);
  device2host(dev_wrk, wrk, tmp, offset);
  //cudaMemcpy(dev_wrk, psi + nx*mytid, 2*dim*sizeof(double), cudaMemcpyDeviceToHost);
#else
  device2host(dev_wrk, psi_2, tmp, offset);
#pragma omp barrier
#pragma omp master
{
  int sx,s;
	double wr,wi;
	/* WRK|PSI holds the new|old wave function */
	for (sx=1; sx<=NX; sx++) {
		wr = al[t][0]*psi[sx][0]-al[t][1]*psi[sx][1];
		wi = al[t][0]*psi[sx][1]+al[t][1]*psi[sx][0];
		wr += (blx[t][sx][0]*psi[sx-1][0]-blx[t][sx][1]*psi[sx-1][1]);
		wi += (blx[t][sx][0]*psi[sx-1][1]+blx[t][sx][1]*psi[sx-1][0]);
		wr += (bux[t][sx][0]*psi[sx+1][0]-bux[t][sx][1]*psi[sx+1][1]);
		wi += (bux[t][sx][0]*psi[sx+1][1]+bux[t][sx][1]*psi[sx+1][0]);
		wrk[sx][0] = wr;
		wrk[sx][1] = wi;
    check(sx, wrk[sx][0], psi_2[sx][0]);
    check(sx, wrk[sx][1], psi_2[sx][1]);
	}
	/* Copy the new wave function back to PSI */	
  for (sx=1; sx<=NX; sx++)
		for (s=0; s<=1; s++)
			psi[sx][s] = wrk[sx][s];
}
#endif
#pragma omp barrier
}

/*----------------------------------------------------------------------------*/
void periodic_bc() {
/*------------------------------------------------------------------------------
	Applies the periodic boundary condition to wave function PSI, by copying
	the boundary values to the auxiliary array positions at the other ends.
------------------------------------------------------------------------------*/
	int s;
  
  MPI_Status status;
  MPI_Request r;
  int plw = (myid-1+nprocs)%nprocs; /* Lower partner process */
  int pup = (myid+1 )%nprocs; /* Upper partner process */

  /* Cache boundary wave function value at the lower end */
	for (s=0; s<=1; s++) {
    dbuf[s] = psi[NX][s];
  }

  MPI_Send(dbuf, 2, MPI_DOUBLE, pup, 0, MPI_COMM_WORLD);
  MPI_Irecv(dbufr, 2, MPI_DOUBLE, plw, 0, MPI_COMM_WORLD, &r); 
  MPI_Wait(&r, &status);
  
	for (s=0; s<=1; s++) {
    psi[0][s] = dbufr[s];
  }
  
  /* Cache boundary wave function value at the upper end */
	for (s=0; s<=1; s++) {
    dbuf[s] = psi[1][s];
  }
  MPI_Send(dbuf, 2, MPI_DOUBLE, plw, 1, MPI_COMM_WORLD);
  MPI_Irecv(dbufr, 2, MPI_DOUBLE, pup, 1, MPI_COMM_WORLD, &r); 
  MPI_Wait(&r, &status);

	for (s=0; s<=1; s++) {
    psi[NX+1][s] = dbufr[s];
  }

	/* Copy boundary wave function values */
  /*
	for (s=0; s<=1; s++) {
		psi[0][s] = psi[NX][s];
		psi[NX+1][s] = psi[1][s];
	}
  */
}

/*----------------------------------------------------------------------------*/
void calc_energy() {
/*------------------------------------------------------------------------------
	Calculates the kinetic, potential & total energies, EKIN, EPOT & ETOT.
------------------------------------------------------------------------------*/
	int sx,s;
	double a,bx;

	/* Apply the periodic boundary condition */
	periodic_bc();

	/* Tridiagonal kinetic-energy operators */
	a =   1.0/(dx*dx);
	bx = -0.5/(dx*dx);

	/* |WRK> = (-1/2)Laplacian|PSI> */
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<=1; s++)
			wrk[sx][s] = a*psi[sx][s]+bx*(psi[sx-1][s]+psi[sx+1][s]);

	/* Kinetic energy = <PSI|(-1/2)Laplacian|PSI> = <PSI|WRK> */
	ekin = 0.0;
	for (sx=1; sx<=NX; sx++)
		ekin += (psi[sx][0]*wrk[sx][0]+psi[sx][1]*wrk[sx][1]);
	ekin *= dx;
  MPI_Allreduce(&ekin, &ekin, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	/* Potential energy */
	epot = 0.0;
	for (sx=1; sx<=NX; sx++)
		epot += v[sx]*(psi[sx][0]*psi[sx][0]+psi[sx][1]*psi[sx][1]);
	epot *= dx;
  MPI_Allreduce(&epot, &epot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	/* Total energy */
	etot = ekin+epot;
}

