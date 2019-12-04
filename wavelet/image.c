#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#define NMAX 512
#define LMAX 9
#define NMIN 64
#define LMIN 6
#define MAX 255
#define MAXLINE 1024
#define NTHREADS 4

#define C0 (0.4829629131445341)
#define C1 (0.8365163037378079)
#define C2 (0.2241438680420134)
#define C3 (-0.1294095225512604)

int main(int argc, char *argv[]) {
  MPI_Init(&argc,&argv);
  int np, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  int row_np = 2;
  int col_np = 2;
  int row_id = myid / row_np;
  int col_id = myid % row_np;
  assert(np ==  row_np * col_np);
  assert(1<<LMAX==NMAX);
  assert(1<<LMIN==NMIN);

  // read image
  double image[NMAX*NMAX];
  int nr,nc;
  if (myid == 0) {
  int n;
  FILE *f;
  char line[MAXLINE];

  f=fopen("Lenna512x512.pgm","r");

  fgets(line,MAXLINE,f);
  fgets(line,MAXLINE,f);
  fgets(line,MAXLINE,f);
  sscanf(line,"%d %d",&nc,&nr);
  fgets(line,MAXLINE,f);
  sscanf(line,"%d",&n);

  for (int r=0; r<nr; r++) {
    for (int c=0; c<nc; c++) {
      image[r*nc+c] = (double)fgetc(f);
    }
  }

  fclose(f);
  }
  MPI_Bcast(&nr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(image, nr*nc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  nr /= row_np;
  nc /= col_np;
  double outimage[NMAX*NMAX]; // shadow buffer
  for (int r=0; r<nr; r++) {
    for (int c=0; c<nc; c++) {
      image[r*nc+c] = image[(row_id*nr+r)*nc*col_np+col_id*nc+c];
    }
  }
  
  // wavelet
  double send_buf[NMAX];
  double recv_buf[NMAX];
  MPI_Status status;
  MPI_Request request;
  for (int l = LMAX; l > LMIN; --l) {
    // columndirection
    int left  = row_id * row_np + (col_id + row_np - 1) % row_np;
    int right = row_id * row_np + (col_id + 1) % row_np;
    //fprintf(stderr, "%d:left%d,right%d\n", myid, left, right);
    MPI_Irecv(recv_buf, 2*nr, MPI_DOUBLE, right, 2*l, MPI_COMM_WORLD, &request);
    for (int r=0; r<nr; r++) {
      for (int c=0; c<=1; c++) {
        send_buf[c*nr+r] = image[r*nc+c];
      }
    }
    MPI_Send(send_buf, 2*nr, MPI_DOUBLE, left, 2*l, MPI_COMM_WORLD);
    MPI_Wait(&request, &status);
    #pragma omp parallel for num_threads(NTHREADS)
    for (int r=0; r<nr; r++) {
      for (int c=0; c<nc/2; c++) {
        if (c == nc/2-1) {
          outimage[r*nc/2+c] = C0*image[r*nc+2*c]+C1*image[r*nc+2*c+1]+C2*recv_buf[r]+C3*recv_buf[nr+r];
          //printf("%lf,%lf,%lf,%lf\n", image[r*nc+2*c], image[r*nc+2*c+1], recv_buf[r], recv_buf[r+nr]);
        } else {
          outimage[r*nc/2+c] = C0*image[r*nc+2*c]+C1*image[r*nc+2*c+1]+C2*image[r*nc+2*c+2]+C3*image[r*nc+2*c+3];
        }
      }
    }
    nc /= 2;
    // rowdirection
    int up   = ((row_id + col_np - 1) % col_np) * row_np + col_id;
    int down = ((row_id + 1) % col_np)          * row_np + col_id;
    // fprintf(stderr, "%d:%d,%d\n", myid, up, down);
    MPI_Irecv(recv_buf, 2*nc, MPI_DOUBLE, down, 2*l+1, MPI_COMM_WORLD, &request);
    for (int r=0; r<=1; r++) {
      for (int c=0; c<nc; c++) {
        send_buf[r*nc+c] = outimage[r*nc+c];
      }
    }
    MPI_Send(send_buf, 2*nc, MPI_DOUBLE, up, 2*l+1, MPI_COMM_WORLD);
    MPI_Wait(&request, &status);
    #pragma omp parallel for num_threads(NTHREADS)
    for (int r=0; r<nr/2; r++) {
      for (int c=0; c<nc; c++) {
        if (r == nr/2-1) {
          image[r*nc+c] = C0*outimage[(2*r)*nc+c]+C1*outimage[(2*r+1)*nc+c]+C2*recv_buf[c]+C3*recv_buf[nc+c];
        } else {
          image[r*nc+c] = C0*outimage[(2*r)*nc+c]+C1*outimage[(2*r+1)*nc+c]+C2*outimage[(2*r+2)*nc+c]+C3*outimage[(2*r+3)*nc+c];
        }
      }
    }
    nr /= 2;
  }

  MPI_Gather(image, nr*nc, MPI_DOUBLE, outimage, nr*nc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (int rp = 0; rp < row_np; ++rp)
  for (int cp = 0; cp < col_np; ++cp)
  for (int r=0; r<nr; r++) {
    for (int c=0; c<nc; c++) {
      image[(rp*nr+r)*nc*col_np+cp*nc+c] = outimage[(rp*col_np+cp)*nr*nc+r*nc+c];
    }
  }
  nr *= row_np;
  nc *= col_np;

  // write image
  if (myid == 0) {
  FILE *f;
  f=fopen("Lenna64x64.pgm","w");

  fprintf(f,"P5\n");
  fprintf(f,"# Simple image test\n");
  fprintf(f,"%d %d\n",nc,nr);
  fprintf(f,"%d\n",MAX);
  fprintf(stderr,"nr%dnc%d\n",nr,nc);

  double imagemax = 0;
  for (int r=0; r<nr; r++) {
    for (int c=0; c<nc; c++) {
      if (image[r*nc+c]>imagemax)
        imagemax = image[r*nc+c];
    }
  }
  fprintf(stderr,"imagemax%lf\n",imagemax);
  for (int r=0; r<nr; r++) {
    for (int c=0; c<nc; c++) {
      //fprintf(stderr,"image%lf\n",image[r*nc+c]);
      fputc((char)(image[r*nc+c]/imagemax*255),f);
    }
  }
  fclose(f);
  }

  MPI_Finalize();

  return 0;
}

