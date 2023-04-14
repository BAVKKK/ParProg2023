#include<iostream>
#include<omp.h>
#include<cmath>
#include<vector>
#include<chrono>

#define SIZE 3
#define MAX_TIMES 39

using namespace std;

float random_float();
void do_task();
void gen_matrix(vector <vector <double>>&);
void gen_answer(vector <double>&);
void count_answer(vector <vector <double>>&, vector <double>&, vector <double>&);
void input();
void Jacobi(vector <vector <double>>& A, vector <double>& B);
void output();

int times = 0;//Current iteration number
double init[SIZE];//Initial vector
double ans[SIZE];//answer

int main()
{
    srand(time(0));
    do_task();
    return 0;
}

void do_task()
{
    vector <vector <double>> A;
    vector <double> B;
    vector <double> result;

    // vector <vector <double>> A = {{10, 1, -1}, {1, 10, -1},{-1, 1, 10}};
    // vector <double> B = {11,10,10};

    gen_matrix(A);
    gen_answer(result);
    count_answer(A,B,result);
    cout << "MATRIX\n";
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            cout << A[i][j] << " ";
        }
        cout << '\n';
    }
    cout << "B is \n";
    for (int i = 0; i < SIZE; i++)
    {
        cout << B[i] << " ";
    }
    cout << endl;
    cout << "ANSWER\n";
    for (int i = 0; i < SIZE; i++)
    {
        cout << result[i] << " ";
    }
    cout << endl;
    auto begin = chrono::steady_clock::now();
    Jacobi(A,B);
    auto end = chrono::steady_clock::now();
        auto elapsed_time = chrono::duration_cast<std::chrono::microseconds>(end - begin);

    cout << "With " << SIZE << " size" << endl
            << "                    (work time)" << elapsed_time.count() << " mcrosec" << endl
            << "=====================================" << endl;
}

float random_float()
{
    return (float)(rand()) / (float)(rand());
}

void gen_matrix(vector <vector <double>>& A)
{
    A.resize(SIZE);
    for (int i = 0; i < SIZE; i++)
    {
        A[i].resize(SIZE);
        for (int j = 0; j < SIZE; j++)
        {
            A[i][j] = random_float();
        }
    }
}

void gen_answer(vector <double>& result)
{
    for (int i = 0; i < SIZE; i++)
    {
        result.push_back(random_float());
    }
}

void count_answer(vector <vector <double>>& A, vector <double>& B, vector <double>& result)
{
    double tmp = 0;
    for (int i = 0; i < SIZE; i++)
    {
        tmp = 0;
        for (int j = 0; j < SIZE; j++)
        {
            tmp += A[i][j]*result[j];
        }
        B.push_back(tmp);
    }
}

void Jacobi(vector <vector <double>>& A, vector <double>& B)
{

    for ( int i = 0; i < SIZE; i++ )
    {
        init[i] = 0.0;
        ans[i] = init[i];
    }

    while ( times++ < MAX_TIMES)
    {
    //  #pragma omp parallel for
        for ( int i = 0; i < SIZE; i++ )
        {
            double sum = 0;
            for ( int j = 0; j < SIZE; j++ )
            {
                if ( j != i )
                {
                    sum += init[j] * A[i][j];
                }
            }
            ans[i] = ( B[i] - sum  ) / A[i][i];
        }
    // #pragma omp critical
        for ( int i = 0; i < SIZE; i++ )
        {
            init[i] = ans[i];
        }
        output();
    }
    // cout << "Result\n";
    // for ( int i = 0; i < SIZE; i++ )
    // {
    //     printf ( "%.4f\t", init[i] );
    // }
}

void output()
{

    cout << "Iteration" << times << " Times, the answer is" << endl;
    for ( int i = 0; i < SIZE; i++ )
    {
        printf ( "%.4f\t", init[i] );
    }
    printf ( "\n" );
}

