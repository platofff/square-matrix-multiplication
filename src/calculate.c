#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <immintrin.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>


#define SIZE 4096
#define PARTSIZE 8
static double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE] = {{0}}, C_REFERENCE[SIZE][SIZE] = {{0}};
struct timeval start, end;

static inline bool approximately_equal(double n1, double n2)
{
  return (int)n1 == (int)n2;
}

static inline double dprod(double *a, double *b) {
  double s0 = a[0] * b[0],
         s1 = a[1] * b[1],
         s2 = a[2] * b[2],
         s3 = a[3] * b[3],
         s4 = a[4] * b[4],
         s5 = a[5] * b[5],
         s6 = a[6] * b[6],
         s7 = a[7] * b[7];
  return (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
}

void rec_mult(double *C, double *A, double *B, int n, int rowsize)
{
  if (n == PARTSIZE)
  {
    double BT[PARTSIZE][PARTSIZE], AA[PARTSIZE][PARTSIZE];
    for (uint_fast8_t i = 0; i < PARTSIZE; i++)
    {
      BT[0][i] = B[rowsize * i + 0];
      BT[1][i] = B[rowsize * i + 1];
      BT[2][i] = B[rowsize * i + 2];
      BT[3][i] = B[rowsize * i + 3];
      BT[4][i] = B[rowsize * i + 4];
      BT[5][i] = B[rowsize * i + 5];
      BT[6][i] = B[rowsize * i + 6];
      BT[7][i] = B[rowsize * i + 7];
      AA[i][0] = A[rowsize * i + 0];
      AA[i][1] = A[rowsize * i + 1];
      AA[i][2] = A[rowsize * i + 2];
      AA[i][3] = A[rowsize * i + 3];
      AA[i][4] = A[rowsize * i + 4];
      AA[i][5] = A[rowsize * i + 5];
      AA[i][6] = A[rowsize * i + 6];
      AA[i][7] = A[rowsize * i + 7];
    }

    C[0] += dprod(&AA[0][0], &BT[0][0]);
    C[1] += dprod(&AA[0][0], &BT[1][0]);
    C[2] += dprod(&AA[0][0], &BT[2][0]);
    C[3] += dprod(&AA[0][0], &BT[3][0]);
    C[4] += dprod(&AA[0][0], &BT[4][0]);
    C[5] += dprod(&AA[0][0], &BT[5][0]);
    C[6] += dprod(&AA[0][0], &BT[6][0]);
    C[7] += dprod(&AA[0][0], &BT[7][0]);

    C[rowsize]     += dprod(&AA[1][0], &BT[0][0]);
    C[rowsize + 1] += dprod(&AA[1][0], &BT[1][0]);
    C[rowsize + 2] += dprod(&AA[1][0], &BT[2][0]);
    C[rowsize + 3] += dprod(&AA[1][0], &BT[3][0]);
    C[rowsize + 4] += dprod(&AA[1][0], &BT[4][0]);
    C[rowsize + 5] += dprod(&AA[1][0], &BT[5][0]);
    C[rowsize + 6] += dprod(&AA[1][0], &BT[6][0]);
    C[rowsize + 7] += dprod(&AA[1][0], &BT[7][0]);

    C[2 * rowsize]     += dprod(&AA[2][0], &BT[0][0]);
    C[2 * rowsize + 1] += dprod(&AA[2][0], &BT[1][0]);
    C[2 * rowsize + 2] += dprod(&AA[2][0], &BT[2][0]);
    C[2 * rowsize + 3] += dprod(&AA[2][0], &BT[3][0]);
    C[2 * rowsize + 4] += dprod(&AA[2][0], &BT[4][0]);
    C[2 * rowsize + 5] += dprod(&AA[2][0], &BT[5][0]);
    C[2 * rowsize + 6] += dprod(&AA[2][0], &BT[6][0]);
    C[2 * rowsize + 7] += dprod(&AA[2][0], &BT[7][0]);

    C[3 * rowsize]     += dprod(&AA[3][0], &BT[0][0]);
    C[3 * rowsize + 1] += dprod(&AA[3][0], &BT[1][0]);
    C[3 * rowsize + 2] += dprod(&AA[3][0], &BT[2][0]);
    C[3 * rowsize + 3] += dprod(&AA[3][0], &BT[3][0]);
    C[3 * rowsize + 4] += dprod(&AA[3][0], &BT[4][0]);
    C[3 * rowsize + 5] += dprod(&AA[3][0], &BT[5][0]);
    C[3 * rowsize + 6] += dprod(&AA[3][0], &BT[6][0]);
    C[3 * rowsize + 7] += dprod(&AA[3][0], &BT[7][0]);

    C[4 * rowsize]     += dprod(&AA[4][0], &BT[0][0]);
    C[4 * rowsize + 1] += dprod(&AA[4][0], &BT[1][0]);
    C[4 * rowsize + 2] += dprod(&AA[4][0], &BT[2][0]);
    C[4 * rowsize + 3] += dprod(&AA[4][0], &BT[3][0]);
    C[4 * rowsize + 4] += dprod(&AA[4][0], &BT[4][0]);
    C[4 * rowsize + 5] += dprod(&AA[4][0], &BT[5][0]);
    C[4 * rowsize + 6] += dprod(&AA[4][0], &BT[6][0]);
    C[4 * rowsize + 7] += dprod(&AA[4][0], &BT[7][0]);

    C[5 * rowsize]     += dprod(&AA[5][0], &BT[0][0]);
    C[5 * rowsize + 1] += dprod(&AA[5][0], &BT[1][0]);
    C[5 * rowsize + 2] += dprod(&AA[5][0], &BT[2][0]);
    C[5 * rowsize + 3] += dprod(&AA[5][0], &BT[3][0]);
    C[5 * rowsize + 4] += dprod(&AA[5][0], &BT[4][0]);
    C[5 * rowsize + 5] += dprod(&AA[5][0], &BT[5][0]);
    C[5 * rowsize + 6] += dprod(&AA[5][0], &BT[6][0]);
    C[5 * rowsize + 7] += dprod(&AA[5][0], &BT[7][0]);

    C[6 * rowsize]     += dprod(&AA[6][0], &BT[0][0]);
    C[6 * rowsize + 1] += dprod(&AA[6][0], &BT[1][0]);
    C[6 * rowsize + 2] += dprod(&AA[6][0], &BT[2][0]);
    C[6 * rowsize + 3] += dprod(&AA[6][0], &BT[3][0]);
    C[6 * rowsize + 4] += dprod(&AA[6][0], &BT[4][0]);
    C[6 * rowsize + 5] += dprod(&AA[6][0], &BT[5][0]);
    C[6 * rowsize + 6] += dprod(&AA[6][0], &BT[6][0]);
    C[6 * rowsize + 7] += dprod(&AA[6][0], &BT[7][0]);

    C[7 * rowsize]     += dprod(&AA[7][0], &BT[0][0]);
    C[7 * rowsize + 1] += dprod(&AA[7][0], &BT[1][0]);
    C[7 * rowsize + 2] += dprod(&AA[7][0], &BT[2][0]);
    C[7 * rowsize + 3] += dprod(&AA[7][0], &BT[3][0]);
    C[7 * rowsize + 4] += dprod(&AA[7][0], &BT[4][0]);
    C[7 * rowsize + 5] += dprod(&AA[7][0], &BT[5][0]);
    C[7 * rowsize + 6] += dprod(&AA[7][0], &BT[6][0]);
    C[7 * rowsize + 7] += dprod(&AA[7][0], &BT[7][0]);
  }
  else
  {
    const int d11 = 0,
              d12 = n / 2,
              d21 = (n / 2) * rowsize,
              d22 = (n / 2) * (rowsize + 1);

    // C11 += A1B11
    rec_mult(C + d11, A + d11, B + d11, n / 2, rowsize);
    // C11 += A12 * B21
    rec_mult(C + d11, A + d12, B + d21, n / 2, rowsize);

    // C12 += A1B12
    rec_mult(C + d12, A + d11, B + d12, n / 2, rowsize);
    // C12 += A12 * B22
    rec_mult(C + d12, A + d12, B + d22, n / 2, rowsize);

    // C21 += A2B11
    rec_mult(C + d21, A + d21, B + d11, n / 2, rowsize);
    // C21 += A22 * B21
    rec_mult(C + d21, A + d22, B + d21, n / 2, rowsize);

    // C22 += A2B12
    rec_mult(C + d22, A + d21, B + d12, n / 2, rowsize);
    // C22 += A22 * B22
    rec_mult(C + d22, A + d22, B + d22, n / 2, rowsize);
  }
}

int main()
{
  srand(time(NULL)); // Текущее время как random seed

  // Привязка исполнения к 1 процессору
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(0, &mask);
  sched_setaffinity(0, sizeof(mask), &mask);

  // Заполнение матриц A и B случайными данными
  for (uint_fast32_t i = 0; i < SIZE; i++)
  {
    for (uint_fast32_t j = 0; j < SIZE; j++)
    {
      A[i][j] = (double)rand() / RAND_MAX * 100;
      B[i][j] = (double)rand() / RAND_MAX * 100;
    }
  }

  // Наивная реализация по формуле из линейной алгебры
  gettimeofday(&start, 0);
  
  for (uint_fast16_t i = 0; i < SIZE; i++) {
    for (uint_fast16_t j = 0; j < SIZE; j++) {
      for (uint_fast16_t k = 0; k < SIZE; k++) {
        C_REFERENCE[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  gettimeofday(&end, 0);
  printf("1) Формула из линейной алгебры. Прошло времени: %lfs\n", (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1e-6);

  gettimeofday(&start, 0);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, SIZE, SIZE, SIZE, 1.0L, &A[0][0], SIZE, &B[0][0], SIZE, 0.0L, &C_REFERENCE[0][0], SIZE);
  gettimeofday(&end, 0);
  printf("2) BLAS. Прошло времени: %lfс\n", (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1e-6);

  gettimeofday(&start, 0);
  rec_mult(&C[0][0], &A[0][0], &B[0][0], SIZE, SIZE);
  gettimeofday(&end, 0);

  printf("3) Моя реализация. Прошло времени: %lfs\n", (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1e-6);
  for (uint_fast32_t i = 0; i < SIZE; i++)
  {
    for (uint_fast32_t j = 0; j < SIZE; j++)
    {
      if (!approximately_equal(C[i][j], C_REFERENCE[i][j]))
      {
        fprintf(stderr, "i=%lu j=%lu Референс %lf Значение %lf\n", i, j, C_REFERENCE[i][j], C[i][j]);
        exit(-1);
      } // Проверка соответствия результатов моей реализации и BLAS
    }
  }
  puts("Результаты совпадают.");
}
