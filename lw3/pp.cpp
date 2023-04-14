#include <math.h>
#include <omp.h>
#include <iostream>


using namespace std;


float next_jacobi_approximation(float **b_mat, float *x, float *x_next, float *d, int n)
{
    float xi, max;
#pragma omp parallel for private(xi)
    for (int i = 0; i < n; i++)
    {
        xi = 0;
        for (int j = 0; j < n; j++)
        {
            xi += b_mat[i][j] * x[j];
        }
        xi += d[i];
#pragma omp critical
        if (i == 0 || fabs(x[i] - xi) > max)
        {
            max = fabs(x[i] - xi);
        }
        x_next[i] = xi;
    }
    return max;
}


int jacobi_parallel(float **a, float *b, float *x, int n, float eps)
{
    float **b_mat, *d, *x_next, max = 1;


    b_mat = new float *[n];
    for (int i = 0; i < n; i++)
        b_mat[i] = new float[n];


    d = new float[n];
    x_next = new float[n];


#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
#pragma omp parallel for
        for (int j = 0; j < n; j++)
            if (i == j)
                b_mat[i][j] = 0;
            else
                b_mat[i][j] = -a[i][j] / a[i][i];
        d[i] = b[i] / a[i][i];
    }
   
    for (int i = 0; i < n; i++)
        x_next[i] = d[i];




    while (max > eps)
    {
        for (int i = 0; i < n; i++)
            x[i] = x_next[i];


        max = next_jacobi_approximation(b_mat, x, x_next, d, n);
    }


    for (int i = 0; i < n; i++)
            x[i] = x_next[i];


    delete[] b_mat;
    delete[] d;
    delete[] x_next;
    return 0;
}


void generate_system(float** mat, float* vec, int size){
    for (int i = 0; i < size; mat[i][i] = 1, i++)
        for (int j = 0; j < size; j++)
            if (i != j)
                mat[i][j] = 0.1 / (i + j);


    for (int i = 0; i < size; i++)
        vec[i] = sin(i);
}


int main()
{
    int result;
    float **a;
    float *b;
    float *x;
    float eps = 1e-6;


    int size = 3;


    float mat[3][3] = {{1.16129, 0.971164, 0.405584}, {0.696859, 1.13715, 20.261}, {0.485486, 0.793196, 3.13421 }};
    a = new float *[size];


    for (int i = 0; i < size; i++)
        a[i] = mat[i];//new float[size];


    b = new float[size]{4.64871, 35.855, 7.84521}; // 0.703978 3.29319 1.56061
    x = new float[size];


    jacobi_parallel(a, b, x, size, eps);


    cout << "[" << x[0];
    for(int i = 1; i < size; i++){
        cout << ", " << x[i];
    }
    cout << "]" << endl;




    cout.flush();
    delete[] a;
    delete[] b;
    delete[] x;
   
    return 0;
}
