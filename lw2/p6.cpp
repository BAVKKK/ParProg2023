#include <cstdlib>
#include <iostream>
#include <cstring>
#include <vector>
#include <pthread.h>
#include <ctime>
#include <map>
#include <chrono>

using namespace std;

#define err_exit(code, str)                   \
    {                                         \
        cerr << str << ": " << strerror(code) \
             << endl;                         \
        exit(EXIT_FAILURE);                   \
    }

// Структура параметров потока для map-функции
struct thread_params
{
    int col;                       // Количество элементов, обрабатываемых одним потоком
    multimap<int, int> pContainer; // Контейнер для хранения обработанных значений
};

vector<int> arr;         // Массив значений
int current_task = 0;    // Указатель на последний необработанный элемент
int size = 0;            // Размер массива данных
pthread_spinlock_t lock; // Спинлок
int res[10];             // Сумма чисел, остаток которых при делении на 10 совпадает

void *map_job(void *arg)
{
    thread_params *params = (thread_params *)arg;

    // Считываем поданные параметры
    int col = params->col; // размер массива

    int err;
    int task_no;

    // Перебираем в цикле необработанные значения массива
    while (true)
    {
        // Захватываем спинлок
        err = pthread_spin_lock(&lock);

        if (err != 0)
            err_exit(err, "Cannot lock spinlock");

        // Запоминаем номер элемента массива, с которого начнём обработку
        task_no = current_task;

        // Сдвигаем указатель последнего необработанного элемента
        current_task += col;

        // Освобождаем спинлок
        err = pthread_spin_unlock(&lock);
        if (err != 0)
            err_exit(err, "Cannot unlock spinlock");

        if (task_no < size)
        {

            int i = task_no;

            while ((i < task_no + col) && (i < size))
            {
                params->pContainer.insert(pair<int, int>(arr[i] % 10, arr[i]));
                i++;
            }
        }
        else
            return NULL;
    }
}

void *reduce_job(void *arg)
{
    multimap<int, int> *pContainer = (multimap<int, int> *)arg;

    int err;

    for (auto item = pContainer->begin(); item != pContainer->end(); ++item)
    {
        // Захватываем спинлок
        err = pthread_spin_lock(&lock);
        if (err != 0)
            err_exit(err, "Cannot lock spinlock");
    #ifdef COUT
        cout << item->first << ": " << item->second << endl;
    #endif
        res[item->first] += item->second;

        // Освобождаем спинлок
        err = pthread_spin_unlock(&lock);
        if (err != 0)
            err_exit(err, "Cannot unlock spinlock");
    }

    return NULL;
}

// Функция для заполнения массива
void fillArr()
{
    // Генерируем случайное число в диапазоне от 1 до 1000
    for (int i = 0; i < size; i++)
    {
        arr.push_back(rand() % 1000 + 1);
    }
}

int main(int argc, char *argv[])
{
    srand(time(0));

    // Если задано меньше параметров, чем необходимо, то завершаем работу программы
    if (argc < 2)
    {
        cout << "Wrong number of arguments" << endl;
        exit(-1);
    }

    int err;

    // Инициализируем спинлок
    err = pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE);
    if (err != 0)
        err_exit(err, "Cannot initialize spinlock");

    int threads_number = atoi(argv[1]);                        // Количество потоков
    size = atoi(argv[2]);                                      // Размер массива данных
    pthread_t *threads = new pthread_t[threads_number];        // Идентификаторы потока
    thread_params *params = new thread_params[threads_number]; // Параметры потока

    if (threads_number > size)
    {
        threads_number = size;
    }

    // ---------------------------------
    // заполняем массив
    fillArr();

    // ---------------------------------
    // функция MAP (обрабатываем полученные данные)

    int col = size / threads_number;
    auto begin = chrono::steady_clock::now();
    for (int i = 0; i < threads_number; i++)
    {
        params[i].col = col;

        // Создаём i-тый поток
        err = pthread_create(&(threads[i]), NULL, map_job, &(params[i]));

        if (err != 0)
        {
            cout << "Cannot create a thread number " << i << ":" << strerror(err) << endl;
            exit(-1);
        }
    }

    // Ожидаем завершения созданных потоков
    for (int i = 0; i < threads_number; i++)
    {
        pthread_join(threads[i], NULL);
    }

    auto end = chrono::steady_clock::now();
    auto f_time = chrono::duration_cast<std::chrono::microseconds>(end - begin);
    cout << "Map Time : " << f_time.count() << "ms" << endl;


    
    // ---------------------------------
    // функция REDUCE (финальная обработка данных)

    auto begin_r = chrono::steady_clock::now();
    for (int i = 0; i < threads_number; i++)
    {
        // Создаём i-тый поток
        err = pthread_create(&(threads[i]), NULL, reduce_job, &(params[i].pContainer));

        if (err != 0)
        {
            cout << "Cannot create a thread number " << i << ":" << strerror(err) << endl;
            exit(-1);
        }
    }


    // Ожидаем завершения созданных потоков
    for (int i = 0; i < threads_number; i++)
    {
        pthread_join(threads[i], NULL);
    }

    auto end_r = chrono::steady_clock::now();
    auto f_r_time = chrono::duration_cast<std::chrono::microseconds>(end_r - begin_r);
    cout << "Reduce Time : " << f_r_time.count() << "ms" << endl;

    #ifdef COUT
    cout << "\nPeзyльтaт: " << endl;

    for (int i = 0; i < 10; i++)
    {
        cout << i << ": " << res[i] << endl;
    }
    #endif
    pthread_spin_destroy(&lock);
    pthread_exit(NULL);
}