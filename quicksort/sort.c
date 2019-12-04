/* Hypercube quicksort using MPI **********************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"

#define N 1024 /* Maximum list size */
#define MAX 99 /* Maximum value of a list element */
#define MAXPROC 10
#define MAXD 3
#define MAXP 4

int nprocs,dim,myid; /* Cube size, dimension, & my rank */

int partition(int pivot, int list[], int left, int right) {
  int i, j;
  int temp;
  i = left-1; j = right+1;
    //if (myid==2) printf("j%d\n",j);
  do {
    while (i <= right && list[++i] <= pivot);
    while (j >= left && list[--j] > pivot);
    if (i < j) {
    temp = list[i]; list[i] = list[j]; list[j] = temp;
    }
  } while (i < j);
  return j;
}

/* Sequential quicksort */
void quicksort(int list[],int left,int right) {
  int pivot,j;
  int temp;
  if (left < right) {
    pivot = list[left];
    j = partition(pivot,list,left+1,right);
    temp = list[left]; list[left] = list[j]; list[j] = temp;
    quicksort(list,left,j-1);
    quicksort(list,j+1,right);
  }
}

int nprocs_cube;
int nelement;

/* Parallel mergesort */
void parallel_quicksort(int myid, int list[], int recv_list[], int n)
{
  MPI_Comm cube[MAXD][MAXP];
  cube[dim][0] = MPI_COMM_WORLD;
  MPI_Group cube_group[MAXD][MAXP];
  MPI_Status status;
  int i, j, k;
  int bitvalue = nprocs >> 1;
  int mask = nprocs - 1;
  int pivot;
  int nsend, nreceive;
  nelement = n/nprocs;
  for (int L=dim; L>=1; L--) {
    // MPI Communicators
    // Recursive bisection of processor groups
    int c = myid/nprocs_cube;
    // Calculate the pivot as the average of the local list element values
    int sum = 0;
    if ((myid & mask) == 0) {
      for (i=0; i<nelement; i++) sum += list[i];
      if (nelement) pivot = sum/nelement; // if nelement is 0, use last pivot
    }
    // broadcast the pivot from the master to the other members of the subcube;
    MPI_Bcast(&pivot,1,MPI_INT,0,cube[L][c]);
    // partition list[0:nelement-1] into two sublists such that
    // list[0:j] ≤ pivot < list[j+1:nelement-1];
    j = partition(pivot,list,0,nelement-1);
    int partner = myid ^ bitvalue;
    //fprintf(stderr, "[%d]%d->%d:pivot%dj%d%s\n",L,myid,partner,pivot,j,(myid&bitvalue)?"senior":"junior");
    if ((myid & bitvalue) == 0) { // junior partner
      // send the right sublist list[j+1:nelement-1] to partner;
      nsend = nelement - (j + 1);
      MPI_Send(&nsend, 1, MPI_INT, partner, L*dim*2, MPI_COMM_WORLD);
      //fprintf(stderr,"[%d]%d:nsend%dnreceive%dtag%d\n",L,myid,nsend,nreceive,dim);
      if (nsend) {
        MPI_Send(&list[j+1], nsend, MPI_INT, partner, L*dim*2+1, MPI_COMM_WORLD);
      }
      // receive the left sublist of partner;
      MPI_Recv(&nreceive, 1, MPI_INT, partner, L*dim*2, MPI_COMM_WORLD, &status);
      if (nreceive) {
        MPI_Recv(recv_list, nreceive, MPI_INT, partner, L*dim*2+1, MPI_COMM_WORLD, &status);
      }
      // append the received list to my left list
      for (i = 0, k = nreceive; i < (j + 1); ++i,++k) recv_list[k] = list[i];
      for (i = 0; i < (j + 1) + nreceive; ++i) list[i] = recv_list[i];
    } else { // senior partner
      // receive the right sublist of partner;
      MPI_Recv(&nreceive, 1, MPI_INT, partner, L*dim*2, MPI_COMM_WORLD, &status);
      //fprintf(stderr,"[%d]%d:nsend%dnreceive%d\n",L,myid,nsend,nreceive);
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
    //fprintf(stderr,"[%d]%d:nsend%dnreceive%d\n",L,myid,nsend,nreceive);
    nelement = nelement - nsend + nreceive;
    quicksort(list,0,nelement-1);
    mask = mask ^ bitvalue; /* Flip the current bit to 0 */
    bitvalue = bitvalue >> 1; /* Next significant bit */
    if (L > 1) 
    {
    MPI_Comm_group(cube[L][c],&(cube_group[L][c]));
    nprocs_cube = nprocs_cube/2;
    int procs_cube[MAXPROC];
    for(int p=0; p<nprocs_cube; p++) procs_cube[p] = p;
    MPI_Group_incl(cube_group[L][c],nprocs_cube,procs_cube,&(cube_group[L-1][2*c ]));
    MPI_Group_excl(cube_group[L][c],nprocs_cube,procs_cube,&(cube_group[L-1][2*c+1]));
    MPI_Comm_create(cube[L][c],cube_group[L-1][2*c ],&(cube[L-1][2*c ]));
    MPI_Comm_create(cube[L][c],cube_group[L-1][2*c+1],&(cube[L-1][2*c+1]));
    MPI_Group_free(&(cube_group[L ][c ]));
    MPI_Group_free(&(cube_group[L-1][2*c ]));
    MPI_Group_free(&(cube_group[L-1][2*c+1]));
    }
    //for (i=0; i<nelement; i++) fprintf(stderr, "%d:%d ",myid,list[i]);
    //fprintf(stderr, "\n");
    //fprintf(stderr, "%d:%d\n",myid,nelement);
    //MPI_Barrier(MPI_COMM_WORLD);
  }
}

int main(int argc, char *argv[])
{
  int list[N],recv_list[N],n=32,i;

  MPI_Init(&argc,&argv); /* Initialize the MPI environment */

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  dim = log(nprocs+1e-10)/log(2.0);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  MPI_Comm_size(MPI_COMM_WORLD,&nprocs_cube);

  srand((unsigned) myid+1);
  for (i=0; i<n/nprocs; i++) list[i] = rand()%MAX;

  printf("Before: Rank %2d :",myid);
  for (i=0; i<n/nprocs; i++) printf("%3d ",list[i]);
  printf("\n");

  parallel_quicksort(myid,list,recv_list,n);

  printf("After:  Rank %2d :",myid);
  for (i=0; i<nelement; i++) printf("%3d ",list[i]);
  printf("\n");

  MPI_Finalize();

  return 0;
}

