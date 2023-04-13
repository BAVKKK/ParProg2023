#include <omp.h>
#include <iostream>
using namespace std;
int main()
{
    int sum = 0;
    const int a_size = 100;
    int a[a_size], id, size;
    for (int i = 0; i < 100; i++)
        a[i] = i;


#pragma omp parallel private(id, size)
    {
        id = omp_get_thread_num();
        size = omp_get_num_threads();
       
        int integer_part = a_size / size;
        int remainder = a_size % size;
        int a_local_size = integer_part +
                           ((id < remainder) ? 1 : 0);
        int start = integer_part * id +
                    ((id < remainder) ? id : remainder);
        int end = start + a_local_size;
       
       #pragma omp critical (sum)
       {
            for (int i = start; i < end; i++)
            sum += a[i];
       }

        #pragma omp critical (sum)
        cout << "Thread " << id << ", partial sum = " << sum << endl;

    }
    cout << "Final sum = " << sum << endl;
    return 0;
}
