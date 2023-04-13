#include<iostream>
#include<omp.h>
#include<math.h>
#define SIZE 3

using namespace std;

void input();
void Jacobi();
void GS();
void SOR();
void func();
void output();

int m; //m*n coefficient matrix
int max_times;// z maximum number of iterations
int times = 0;//Current iteration number
string choice;//Function selection
double mis;//tolerance scope
double mtrx[SIZE][SIZE];//Coefficient matrix
double b[SIZE];//Constant matrix
double init[SIZE];//Initial vector
double ans[SIZE];//answer

int main()
{
    input();
    Jacobi();
    return 0;
}

void  input()
{
    cout << "Please enter m" << endl;
    cin >> m;

    cout << "Please enter the coefficient matrix m*n(m)" << endl;
    for ( int i = 0; i < m; i++ )
    {
        for ( int j = 0; j < m; j++ )
        {
            cin >> mtrx[i][j];
        }
    }

    cout << "Please enter the constant matrix b:" << endl;
    for ( int i = 0; i < m; i++ )
    {
        cin >> b[i];
    }

    for ( int i = 0; i < m; i++ )
    {
        ans[i] = 0;
    }

    cout << "Please enter the maximum number of iterations:" << endl;
    cin >> max_times;

// cout << endl << "Please enter the maximum error:"<<endl;
//         cin>>mis;
}

void Jacobi()
{
    double sum = 0;
    while ( times++ < max_times )
    {
        // Vector multiplication based on the initial vector. Find the solution corresponding to each row
        //Put the result in ans
        #pragma omp parallel for
            for ( int i = 0; i < m; i++ )
            {
                sum = 0;
                for ( int j = 0; j < m; j++ )
                {
                    if ( j != i )
                    {
                        sum += init[j] * mtrx[i][j];
                    }
                }
                ans[i] = ( b[i] - sum  ) / mtrx[i][i];
            }
            //After the iteration is completed once, update the init array once for the next iteration
            #pragma omp critical
            {
                for ( int i = 0; i < m; i++ )
                {
                    init[i] = ans[i];
                }
                output();
            }
       
    }
}

void output()
{

    cout << "Iteration" << times << "Times, the answer is" << endl;
    for ( int i = 0; i < m; i++ )
    {
        printf ( "%.4f\t", init[i] );
    }
    printf ( "\n" );
}

