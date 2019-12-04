/* Hypercube quicksort using MPI **********************************************/
#include <stdio.h>
#include <math.h>
#include "mpi.h"

#define N 1024 /* Maximum list size */
#define MAX 99 /* Maximum value of a list element */
#define MAXPROC 8
#define MAXD 3
#define MAXP 3

int nprocs,dim,myid; /* Cube size, dimension, & my rank */

int partition(int pivot, int list[], int left, int right) {
  int i, j;
  int temp;
  i = left; j = right + 1;
  do {
    while (list[++i] < pivot && i <= right);
    while (list[--j] > pivot);
    if (i < j) {
    temp = list[i]; list[i] = list[j]; list[j] = temp;
    }
  } while (i < j);
  return j;
}

/* Sequential quicksort */
void quicksort(int list[],int left,int right) {
  int pivot,j;
  if (left < right) {
    pivot = list[left];
    j = partition(pivot,list,left,right);
    temp = list[left]; list[left] = list[j]; list[j] = temp;
    quicksort(list,left,j-1);
    quicksort(list,j+1,right);
  }
}

MPI_Comm cube[MAXD][MAXP];
MPI_Group cube_group[MAXD][MAXP];
int nprocs_cube, c;
int procs_cube[MAXPROC];

/* Parallel mergesort */
void parallel_quicksort(int myid, int list[], int recv_list[], int n)
{
  MPI_Status status;
  int i, j, k;
  int bitvalue = nprocs >> 1;
  int mask = nprocs - 1;
  int nelement = n/nprocs;
  for (L=dim; L>=1; L--) {
    // MPI Communicators
    // Recursive bisection of processor groups
    MPI_Comm_group(cube[L][c],&(cube_group[L][c]));
    nprocs_cube = nprocs_cube/2;
    for(p=0; p<nprocs_cube; p++) procs_cube[p] = p;
    MPI_Group_incl(cube_group[L][c],nprocs_cube,procs_cube,&(cube_group[L-1][2*c ]));
    MPI_Group_excl(cube_group[L][c],nprocs_cube,procs_cube,&(cube_group[L-1][2*c+1]));
    MPI_Comm_create(cube[L][c],cube_group[L-1][2*c ],&(cube[L-1][2*c ]));
    MPI_Comm_create(cube[L][c],cube_group[L-1][2*c+1],&(cube[L-1][2*c+1]));
    MPI_Group_free(&(cube_group[L ][c ]));
    MPI_Group_free(&(cube_group[L-1][2*c ]));
    MPI_Group_free(&(cube_group[L-1][2*c+1]));
    // Calculate the pivot as the average of the local list element values
    int sum = 0;
    if ((myid & mask) == 0) {
      for (i=0; i<nelement; i++) sum += list[i];
    }
    int pivot = sum/nelement;
    // broadcast the pivot from the master to the other members of the subcube;
    MPI_Bcast(&pivot,1,MPI_INT,0,cube[L][myid/nprocs_cube]);
    // partition list[0:nelement-1] into two sublists such that
    // list[0:j] ≤ pivot < list[j+1:nelement-1];
    j = partition(pivot,list,0,nelement-1);
    int partner = myid ^ bitvalue;
    if (myid & bitvalue = 0) { // junior partner
      // send the right sublist list[j+1:nelement-1] to partner;
      nsend = nelement - (j + 1);
      MPI_Send(nsend, 1, MPI_INT, partner, L*dim*2, MPI_COMM_WORLD);
      if (nsend) {
        MPI_Send(&list[j+1], nsend, MPI_INT, partner, L*dim*2+1, MPI_COMM_WORLD);
      }
      // receive the left sublist of partner;
      MPI_Recv(&nreceive, 1, MPI_INT, partner, L*dim*2, MPI_COMM_WORLD, &status);
      if (nreceive) {
        MPI_Recv(recv_list, nreceive, MPI_INT, partner, L*dim*2+1, LMPI_COMM_WORLD, &status);
      }
      // append the received list to my left list
      for (i = 0, k = nreceive; i < (j + 1); ++i,++k) recv_list[k] = list[i];
      for (i = 0; i < (j + 1) + nreceive; ++i) list[i] = recv_list[i];
    } else { // senior partner
      // receive the right sublist of partner;
      MPI_Recv(&nreceive, 1, MPI_INT, partner, L*dim*2, MPI_COMM_WORLD, &status);
      if (nreceive) {
        MPI_Recv(recv_list, nreceive, MPI_INT, partner, L*dim*2+1, MPI_COMM_WORLD, &status);
      }
      // send (& erase) the left sublist list[0:j] to partner;
      nsend = j + 1;
      MPI_Send(&nsend, 1, MPI_INT, partner, L*dim*2, MPI_COMM_WORLD);
      if (nsend) {
        MPI_Send(&list[0], nsend, MPI_INT, partner, L*dim*2+1, MPI_COMM_WORLD);
      }
      // append the received list to my right list
      for (i = nsend, k = 0; i < nelement; ++i,++k) list[k] = list[i];
      for (i = 0; i < nreceive; ++i,++k) list[k] = recv_list[i];
    }
    nelement = nelement - nsend + nreceive;
    quicksort(list,0,nelement-1);
    mask = mask ^ bitvalue; /* Flip the current bit to 0 */
    bitvalue = bitvalue >> 1; /* Next significant bit */
  }
}

int main(int argc, char *argv[])
{
  int list[N],recv_list[N],n=16,i;

  MPI_Init(&argc,&argv); /* Initialize the MPI environment */

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  dim = log(nprocs+1e-10)/log(2.0);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  MPI_Comm_size(MPI_COMM_WORLD,&nprocs_cube);
  c = myid/nprocs_cube
  cube[dim][0] = MPI_COMM_WORLD;

  srand((unsigned) myid+1);
  for (i=0; i<n/nprocs; i++) list[i] = rand()%MAX;

  printf("Before: Rank %2d :",myid);
  for (i=0; i<n/nprocs; i++) printf("%3d ",list[i]);
  printf("\n");

  parallel_quicksort(myid,list,recv_list,n);

  printf("After:  Rank %2d :",myid);
  for (i=0; i<n/nprocs; i++) printf("%3d ",list[i]);
  printf("\n");

  MPI_Finalize();

  return 0;
}

