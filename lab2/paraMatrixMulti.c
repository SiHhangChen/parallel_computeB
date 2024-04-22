#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<time.h>

int** MultipleOfMatrix(int** MatA, int rowOfMatA, int columnOfMatA, int** MatB, int rowOfMatB, int columnOfMatB){
    int** retMat = (int**)malloc(sizeof(int*)*rowOfMatA);
    for(int i = 0; i < rowOfMatA; i++)
        retMat[i] = (int*)malloc(sizeof(int)*columnOfMatB);
    
    #pragma omp parallel for
    for(int i = 0; i < rowOfMatA; i++){
        #pragma omp parallel for
        for(int j = 0; j < columnOfMatB; j++){
            retMat[i][j] = 0;
            int temp[4] = {0, 0, 0, 0};
            // printf("the num of i: %d, j: %d\n", i, j);
            #pragma omp parallel sections 
            {
                #pragma omp section
                for(int k = 0; k < columnOfMatA/4; k++){
                    temp[1] += MatA[i][k] * MatB[k][j];
                }
                #pragma omp section
                for(int k = columnOfMatA/4; k < columnOfMatA/2; k++){
                    temp[2] += MatA[i][k] * MatB[k][j];
                }
                #pragma omp section
                for(int k = columnOfMatA/2; k < columnOfMatA*3/4; k++){
                    temp[3] += MatA[i][k] * MatB[k][j];
                }
                #pragma omp section
                for(int k = columnOfMatA*3/4; k < columnOfMatA; k++){
                    temp[0] += MatA[i][k] * MatB[k][j];
                }
            }
            retMat[i][j] = temp[0] + temp[1] + temp[2] + temp[3];
            // #pragma omp parallel for
            // for(int k = 0; k < columnOfMatA; k++){
            //     retMat[i][j] += MatA[i][k] * MatB[k][j];
            // }
        }
    }
    return retMat;
}

int** SMultipleOfMatrix(int** MatA, int rowOfMatA, int columnOfMatA, 
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
    int threadNum = atoi(argv[4]);
    // int columnOfMatA = 8;
    // int rowOfMatA = 7;
    // int columnOfMatB = 9;
    int** MatA = (int**)malloc(sizeof(int*)*rowOfMatA);
    int** MatB = (int**)malloc(sizeof(int*)*columnOfMatA);
    for(int i = 0; i < rowOfMatA; i++)
        MatA[i] = (int*)malloc(sizeof(int)*columnOfMatA);
    for(int i = 0; i < columnOfMatA; i++)
        MatB[i] = (int*)malloc(sizeof(int)*columnOfMatB);

    int startTimeS = clock();
    int** retMatS = SMultipleOfMatrix(MatA, rowOfMatA, columnOfMatA, MatB, columnOfMatA, columnOfMatB);
    int endTimeS = clock();
    printf("串行花费时间: %lf\n", (double)(endTimeS - startTimeS) / CLOCKS_PER_SEC);

    double startTime = omp_get_wtime();

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
    printf("had been gen\n");
    omp_set_num_threads(threadNum);
    int** retMat = MultipleOfMatrix(MatA, rowOfMatA, columnOfMatA, MatB, columnOfMatA, columnOfMatB);
    double endTime = omp_get_wtime();

    // for(int i = 0; i < rowOfMatA; i++){
    //     for(int j = 0; j < columnOfMatB; j++)
    //         printf("%d ", retMat[i][j]);
    //     printf("\n");
    // }
    printf("A矩阵大小: %d  %d\nB矩阵大小: %d  %d\n线程数量: %d\n并行花费时间: %lf\n",
        rowOfMatA, columnOfMatA, columnOfMatA, columnOfMatB, threadNum, endTime - startTime);

   

    printf("加速比为：%lf\n", (double)((endTimeS - startTimeS)/CLOCKS_PER_SEC)/(endTime - startTime));

    for(int i = 0; i < rowOfMatA; i++){
        for(int j = 0; j < columnOfMatB; j++){
            if(retMat[i][j] != retMatS[i][j]){
                printf("error\n");
                return 0;
            }
        }
    }
    printf("right\n");
    return 0;
}