#include <cstdio>
#include <omp.h>

#define NX 102

int main(void)
{
    double vecA[NX], vecB[NX], vecC[NX];

    /* Initialization of the vectors */
    for (int i = 0; i < NX; i++) {
        vecA[i] = 1.0 / ((double)(NX - i));
        vecB[i] = vecA[i] * vecA[i];
    }

    /* TODO:
     *   Implement here a parallelized version of vector addition,
     *   vecC = vecA + vecB
     */
    #pragma omp parallel for 
    for (int i = 0; i < NX; ++i) {
        int tid = omp_get_thread_num();
        printf("thread %d solves for %d\n", tid, i);
        vecC[i] = vecA[i] + vecB[i];
    }

    double sum = 0.0;
    /* Compute the check value */
    for (int i = 0; i < NX; i++) {
        sum += vecC[i];
    }
    printf("Reduction sum: %18.16f\n", sum);

    return 0;
}
