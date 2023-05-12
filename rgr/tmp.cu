#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define BLOCK_DIM 32

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// mat_1: m x n
// mat_2: n x p
// mat_3: m x p
template <typename T>
void mm(int const* mat_1, T const* mat_2, T* mat_3, size_t m, size_t n, size_t p)
{
    // Compute the cells in mat_3 sequentially.
    for (size_t i{0}; i < m; ++i)
    {
        for (size_t j{0}; j < p; ++j)
        {
            T acc_sum{0};
            for (size_t k{0}; k < n; ++k)
            {
                acc_sum += mat_1[i * n + k] * mat_2[k * p + j];
            }
            mat_3[i * p + j] = acc_sum;
        }
    }
}
template <typename T>
__global__ void mm_kernel(T const* mat_1, T const* mat_2, T* mat_3, size_t m,
                          size_t n, size_t p)
{
    // 2D block and 2D thread
    // Each thread computes one cell in mat_3.
    size_t i{blockIdx.y * blockDim.y + threadIdx.y};
    size_t j{blockIdx.x * blockDim.x + threadIdx.x};

    // Do not process outside the matrix.
    // Do not forget the equal sign!
    if ((i >= m) || (j >= p))
    {
        return;
    }

    T acc_sum{0};
    for (size_t k{0}; k < n; ++k)
    {
        acc_sum += mat_1[i * n + k] * mat_2[k * p + j];
    }
    mat_3[i * p + j] = acc_sum;
}

template <typename T>
void mm_cuda(T const* mat_1, T const* mat_2, T* mat_3, size_t m, size_t n,
             size_t p)
{
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(p) /
                                  static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(m) /
                                  static_cast<double>(threads_per_block.y));
    mm_kernel<<<blocks_per_grid, threads_per_block>>>(mat_1, mat_2, mat_3, m, n,
                                                      p);
}

int main()
{
    
    std::cout << "Becare, all right\n";
    return 0;
}