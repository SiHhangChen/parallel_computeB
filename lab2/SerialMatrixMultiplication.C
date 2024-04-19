#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int** MultipleOfMatrix(int** MatA, int rowOfMatA, int columnOfMatA, 
    int** MatB, int rowOfMatB, int columnOfMatB){
    
    int** MatC = (int**)malloc(sizeof(int*)*rowOfMatA);
    for(int i = 0; i < rowOfMatA; i++)
        MatC[i] = (int*)malloc(sizeof(int)*columnOfMatB);

    for(int i = 0; i < rowOfMatA; i++){
        for(int j = 0; j < columnOfMatB; j++){
            MatC[i][j] = 0;
            for(int k = 0; k < columnOfMatA; k++){
                MatC[i][j] += MatA[i][k] * MatB[k][j];
            }
        }
    }
    return MatC;
}

int main(int argc, char* argv[]){
    int columnOfMatA = atoi(argv[1]);
    int rowOfMatA = atoi(argv[2]);
    int columnOfMatB = atoi(argv[3]);
    // int columnOfMatA = 8;
    // int rowOfMatA = 7;
    // int columnOfMatB = 9;
    int** MatA = (int**)malloc(sizeof(int*)*rowOfMatA);
    int** MatB = (int**)malloc(sizeof(int*)*columnOfMatA);
    for(int i = 0; i < rowOfMatA; i++)
        MatA[i] = (int*)malloc(sizeof(int)*columnOfMatA);
    for(int i = 0; i < columnOfMatA; i++)
        MatB[i] = (int*)malloc(sizeof(int)*columnOfMatB);

    double startTime = time(NULL);
    srand(time(NULL)+rand());
    for(int i = 0; i < rowOfMatA; i++){
        for(int j = 0; j < columnOfMatA; j++){
            MatA[i][j] = rand();
        }
    }

    for(int i = 0; i < columnOfMatA; i++){
        for(int j = 0; j < columnOfMatB; j++)
            MatB[i][j] = rand();
    }

    int** retMat = MultipleOfMatrix(MatA, rowOfMatA, columnOfMatA, MatB, columnOfMatA, columnOfMatB);
    double endTime = time(NULL);

    // for(int i = 0; i < rowOfMatA; i++){
    //     for(int j = 0; j < columnOfMatB; j++)
    //         printf("%d ", retMat[i][j]);
    //     printf("\n");
    // }
    printf("A矩阵大小: %d  %d\nB矩阵大小: %d  %d\n进程数量: 1\n并行花费时间: %lf\n",
        rowOfMatA, columnOfMatA, columnOfMatA, columnOfMatB, endTime - startTime);
}