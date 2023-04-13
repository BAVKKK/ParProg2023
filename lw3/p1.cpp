#include <omp.h>
#include <iostream>
#include <cmath>

using namespace std;

int f(int& b)
{
    int result = 0;
    for (int i = 0; i < 10000000; i++)
    {
        result += sin(b) + cos(i);
    }
    return result;
}

int main()
{
    int a[100], b[100];


    for (int i = 0; i < 100; i++)
        b[i] = i;


#pragma omp parallel for
    for (int i = 0; i < 100; i++)
    {
        a[i] = f(b[i]);
        b[i] = 2 * a[i];
    }
    int result = 0;


#pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < 100; i++)
        result += (a[i] + b[i]);
    cout << "Result = " << result << endl;

    return 0;
}
