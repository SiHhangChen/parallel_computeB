#include <omp.h>
#include <stdio.h>

int main() {
    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        printf("Thread %d executes loop iteration %d\n", omp_get_thread_num(), i);
    }
    return 0;
}