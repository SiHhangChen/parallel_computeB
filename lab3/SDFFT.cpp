#include <bits/stdc++.h>
using namespace std;

const int NR = 1 << 24;
const double eps = 1e-6, pi = acos(-1.0);
complex<double> a[NR]; //complex为C++自带虚数

void FFT(complex<double> *a, int n, int inv) //inv为虚部符号，inv为1时FFT，inv为-1时IFFT
{
    if (n == 1) //如果需要转换的只有一项就直接返回
        return;
    int mid = n / 2;
    complex<double> A1[mid + 1], A2[mid + 1];
    for (int i = 0; i <= n; i += 2) //拆分多项式
    {
        A1[i / 2] = a[i];
        A2[i / 2] = a[i + 1];
    }
    FFT(A1, mid, inv); //递归分治
    FFT(A2, mid, inv);
    complex<double> w0(1, 0), wn(cos(2 * pi / n), inv * sin(2 * pi / n)); //单位根
    for (int i = 0; i < mid; ++i, w0 *= wn)                               //合并多项式
    {
        a[i] = A1[i] + w0 * A2[i];
        a[i + n / 2] = A1[i] - w0 * A2[i];
    }
}


int main(int argc, char *argv[])
{
    int idxOfN = atoi(argv[1]);
    // int idxOfM = atoi(argv[2]);
    srand(time(0) + rand()); //随机数种子
    double startTime = clock(); //计时开始
    for (int i = 0; i <= idxOfN; i++) //输入第一个多项式
    {
        double x = rand() % 1000;
        a[i].real(x); //complex类型变量.real(x)意味着将实数部赋为x，real()返回实数部值
    }
    // for (int i = 0; i <= idxOfM; i++) //输入第二个多项式
    // {
    //     double x = rand() % 1000;
    //     b[i].real(x);
    // }
    int len = 1 << max((int)ceil(log2(idxOfN)), 1); //由于FFT需要项数为2的整数次方倍，所以多项式次数len为第一个大于n+m的二的正整数次方
    FFT(a, len, 1);                                //系数表达转点值表达
    // FFT(b, len, 1);
    // for (int i = 0; i <= len; ++i)
    //     a[i] = a[i] * b[i];                       //O(n)乘法
    // FFT(a, len, -1);                              //点值表达转系数表达
    double endTime = clock(); //计时结束
    // for (int i = 0; i <= idxOfN + idxOfM; ++i)              //输出
    //    cout << a[i].real() / len + eps << endl; //记得要除n，eps为解决掉精度问题

    cout << "Time: " << (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl; //输出时间
    return 0;
}