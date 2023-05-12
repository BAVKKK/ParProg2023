#include <omp.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <vector>

//#define MATRIX_MULTY

void crate_rand_matr(int* matr, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
       matr[i] = random() % 100;
    }
}

void mm(int const* mat_1, int const* mat_2, int* mat_3, size_t m, size_t n, size_t p)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < p; ++j)
        {
            int acc_sum = 0;
            for (size_t k = 0; k < n; ++k)
            {
                acc_sum += mat_1[i * n + k] * mat_2[k * p + j];
            }
            mat_3[i * p + j] = acc_sum;
        }
    }
}

void mmOMP(int const* mat_1, int const* mat_2, int* mat_3, size_t m, size_t n, size_t p)
{
    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < p; ++j)
        {
            int acc_sum = 0;
            for (size_t k = 0; k < n; ++k)
            {
                acc_sum += mat_1[i * n + k] * mat_2[k * p + j];
            }
            mat_3[i * p + j] = acc_sum;
        }
    }
}

double measure_latency_mmOMP(size_t m, size_t n, size_t p, size_t num_tests)
{
    double itog = 0.0f;
    int* mat_1 = new int[m*n];
    int* mat_2 = new int[n*p];
    int* mat_3 = new int[m*p];

    crate_rand_matr(mat_1, m*n);
    crate_rand_matr(mat_2, n*p);
#ifdef MATRIX_MULTY
    for (int i = 0; i < m*n; i++)
    {
        std::cout << mat_1[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < n*p; i++)
    {
        std::cout << mat_2[i] << " ";
    }
    std::cout << std::endl;
#endif
    for (int i_test = 0; i_test < 10; i_test++)
    {
        auto begin = std::chrono::steady_clock::now();
        for (size_t i = 0; i < num_tests; ++i)
        {
            mmOMP(mat_1, mat_2, mat_3, m, n, p);
        }
        auto end = std::chrono::steady_clock::now();
        auto f_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        itog += f_time.count() / 100.0f;
    }

#ifdef MATRIX_MULTY
    for (int i = 0; i < n*p; i++)
    {
        std::cout << mat_3[i] << " ";
    }
    std::cout << std::endl;
#endif
    return itog / 10.0f;
}

double measure_latency_mm(size_t m, size_t n, size_t p, size_t num_tests)
{

    double itog = 0.0f;
    int* mat_1 = new int[m*n];
    int* mat_2 = new int[n*p];
    int* mat_3 = new int[m*p];

    crate_rand_matr(mat_1, m*n);
    crate_rand_matr(mat_2, n*p);

#ifdef MATRIX_MULTY
    for (int i = 0; i < m*n; i++)
    {
        std::cout << mat_1[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < n*p; i++)
    {
        std::cout << mat_2[i] << " ";
    }
    std::cout << std::endl;
#endif

    for (int i_test = 0; i_test < 10; i_test++)
    {
        auto begin = std::chrono::steady_clock::now();
        for (size_t i = 0; i < num_tests; ++i)
        {
            mm(mat_1, mat_2, mat_3, m, n, p);
        }
        auto end = std::chrono::steady_clock::now();
        auto f_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        itog += f_time.count() / 100.0f;
    }
    
#ifdef MATRIX_MULTY
    for (int i = 0; i < n*p; i++)
    {
        std::cout << mat_3[i] << " ";
    }
    std::cout << std::endl;
#endif
    return itog / 10.0f;
}

int main()
{
    size_t size;
    std::cout << "Введите размерность матрицы\n";
    std::cin >> size;

    size_t m = size, n =  size, p =  size;

    size_t num_measurement_tests = 100;

    double mm_latency_OMP = measure_latency_mmOMP(m, n, p, num_measurement_tests);
    double mm_latency = measure_latency_mm(m, n, p, num_measurement_tests);

    std::cout << "Matrix Multiplication" << '\n';
    std::cout << "OMP " << std::fixed << std::setprecision(5) << mm_latency_OMP<< " ms" << '\n';
    std::cout << "CPU " << std::fixed << std::setprecision(5) << mm_latency << " ms" << '\n';
    return 0;
}