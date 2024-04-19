#include<stdio.h>
#include<omp.h>
#include<time.h>
#include<stdlib.h>

void partition(int* arr, int start, int end, int* pos){
    int temp = arr[start];
    while(start < end){
        while(start < end && arr[end] >= temp) end--;
        arr[start] = arr[end];
        while(start < end && arr[start] <= temp) start++;
        arr[end] = arr[start];
    }
    arr[start] = temp;
    *pos = start;
}

void quickSort(int* arr, int start, int end){
    if (start < end) {
        int pos = 0;
        partition(arr, start, end, &pos);
        #pragma omp parallel sections
        {
            #pragma omp section
            quickSort(arr, start, pos);
            #pragma omp section
            quickSort(arr, pos + 1, end);
        }
    }
}

int main(int argc, char* argv[]){
    int threadNum = atoi(argv[1]);
    int arrSize = atoi(argv[2]);
    int* arr = (int*)malloc(sizeof(int)*arrSize);
    int idx;
    double startTime = omp_get_wtime();
    srand(time(NULL) - rand());
    for(idx = 0; idx < arrSize; idx++) {
        arr[idx] = rand();
    }
    omp_set_num_threads(threadNum);
    quickSort(arr, 0, arrSize - 1);
    double endTime = omp_get_wtime();

    printf("数组大小：%d\n进程数量: %d\n并行花费时间: %lf\n", arrSize, threadNum, endTime - startTime);

}