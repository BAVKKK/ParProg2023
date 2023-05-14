#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define BLOCK_DIM 32
#define CHECK_TIME
// #define MATRIX_MULTY

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Настройка вычислительного ядра
__global__ void mm_kernel(int const* mat_1, int const* mat_2, int* mat_3, size_t m,
                          size_t n, size_t p)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    // Чтобы не обрабатывать вне матрицы
    if ((i >= m) || (j >= p))
    {
        return;
    }

    int acc_sum = 0;
    for (size_t k = 0; k < n; ++k)
    {
        acc_sum += mat_1[i * n + k] * mat_2[k * p + j];
    }
    mat_3[i * p + j] = acc_sum;
}


// Умножение матриц с кудой 
void mm_cuda(int const* mat_1, int const* mat_2, int* mat_3, size_t m, size_t n,
             size_t p)
{
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(p) /
                                  static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(m) /
                                  static_cast<double>(threads_per_block.y));
    mm_kernel<<<blocks_per_grid, threads_per_block>>>(mat_1, mat_2, mat_3, m, n, p);
}

void crate_rand_matr(int* matr, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
       matr[i] = random() % 100;
    }
}

// Измерение времени умножения куда
float measure_latency_mm_cuda(size_t m, size_t n, size_t p, size_t num_tests)
{
    cudaEvent_t startEvent, stopEvent;
    float time = 0.0f;

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    int *d_mat_1, *d_mat_2, *d_mat_3;
    
    // Выделяем память под матрицы
    checkCuda(cudaMalloc(&d_mat_1, sizeof(int) * m * n));
    checkCuda(cudaMalloc(&d_mat_2, sizeof(int) * n * p));
    checkCuda(cudaMalloc(&d_mat_3, sizeof(int) * m * p));

    int* mat_1 = new int[m*n];
    int* mat_2 = new int[n*p];
    int* mat_3 = new int[m*p];

    crate_rand_matr(mat_1, m*n);
    crate_rand_matr(mat_2, n*p);

    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(d_mat_1, mat_1, m*n * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_mat_2, mat_2, n*p * sizeof(int), cudaMemcpyHostToDevice));
#ifdef CHECK_TIME
// for (int i_test = 0; i_test < 10; i_test++)
// {
    float tmp_time = 0.0f;
    for (size_t i = 0; i < num_tests; ++i)
    {
        mm_cuda(d_mat_1, d_mat_2, d_mat_3, m, n, p);
    }
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaMemcpy(mat_3, d_mat_3, m*p * sizeof(int), cudaMemcpyDeviceToHost));
    checkCuda(cudaEventElapsedTime(&tmp_time, startEvent, stopEvent));
    time+=tmp_time/num_tests;
// }
#endif

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
    mm_cuda(d_mat_1, d_mat_2, d_mat_3, m, n, p);
    checkCuda(cudaMemcpy(mat_3, d_mat_3, m*p * sizeof(int), cudaMemcpyDeviceToHost));
    int k = 1;
    for (int i = 0; i < m*p; i++)
    {
        std::cout << mat_3[i] << " ";
        if (k == p)
        {
            std::cout << '\n';
            k = 0;
        }
        k++;
    }
    std::cout << std::endl;
#endif

    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Free device buffer.
    checkCuda(cudaFree(d_mat_1));
    checkCuda(cudaFree(d_mat_2));
    checkCuda(cudaFree(d_mat_3));

    // float latency = time / 10;
    float latency = time;

    return latency;
}

int main()
{
    size_t size;
    std::cout << "Введите размерность матрицы\n";
    std::cin >> size;

    size_t m = size, n =  size, p =  size;

    size_t num_measurement_tests = 100;

    float mm_cuda_latency = measure_latency_mm_cuda(m, n, p, num_measurement_tests);

    std::cout << "Matrix Multiplication CUDA" << '\n';
    std::cout << "GPU: " << std::fixed << std::setprecision(5) << mm_cuda_latency << " ms" << '\n';
}