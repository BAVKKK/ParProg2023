// Команда для компиляции программы
// g++ -pthread -Dsyn_type=0 p4.cpp -o p4 && ./p4 10
// А то потом опять забуду как дефайны менять во время компиляции

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <pthread.h>
#include <cmath>
#include <chrono>

using namespace std;

#define syn_type 0
#define err_exit(code, str)                   \
    {                                         \
        cerr << str << ": " << strerror(code) \
             << endl;                         \
        exit(EXIT_FAILURE);                   \
    }

const int TASKS_COUNT = 1000;

int task_list[TASKS_COUNT]; // Массив заданий
int current_task = 0;       // Указатель на текущее задание
pthread_mutex_t mutex;      // Мьютекс
pthread_spinlock_t lock;    // Спинлок
// int syn_type = 1;           // Тип синхронизации

void do_task()
{
    int n = 0;

    for (int i = 0; i < 100000; i++)
    {
        n += sin(i);
    }
}

void *thread_job(void *arg)
{
    int task_no;
    int err;

    // Перебираем в цикле доступные задания
    while (true)
    {

        if (syn_type == 1)
        {
            // Захватываем спинлок
            err = pthread_spin_lock(&lock);
            if (err != 0)
                err_exit(err, "Cannot lock spinlock");
        }
        else if (syn_type == 2)
        {
            // Захвтываем мьютекс
            err = pthread_mutex_lock(&mutex); // Закоментил для П3
            if (err != 0)
                err_exit(err, "Cannot lock mutex");
        }

        // Запоминаем номер текущего задания, которое будем исполнять
        task_no = current_task;

        // Сдвигаем указатель текущего задания на следующее
        current_task++;

        if (syn_type == 1)
        {
            // Освобождаем спинлок
            err = pthread_spin_unlock(&lock);
            if (err != 0)
                err_exit(err, "Cannot unlock spinlock");
        }
        else if (syn_type == 2)
        {
            // Освобождаем мьютекс
            err = pthread_mutex_unlock(&mutex); // Закоментил для П3
            if (err != 0)
                err_exit(err, "Cannot unlock mutex");
        }

        // Все ли задания выполнены
        if (task_no < TASKS_COUNT)
            do_task();
        else
            return NULL;
    }
}

int main(int argc, char *argv[])
{
    // Если задано меньше параметров, чем необходимо, то завершаем работу программы
    if (argc < 2)
    {
        cout << "Wrong number of arguments" << endl;
        exit(-1);
    }

    // Считываем переданные параметры
    int threads_number = atoi(argv[1]); // Количество потоков

    pthread_t *threads = new pthread_t[threads_number]; // Идентификаторы потока
    int err;                                            // Код ошибки

    // Инициализируем массив заданий случайными числами
    for (int i = 0; i < TASKS_COUNT; ++i)
        task_list[i] = rand() % TASKS_COUNT;

    if (syn_type == 1)
    {
        // Инициализируем спинлок
        err = pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE);
        if (err != 0)
            err_exit(err, "Cannot initialize spinlock");
    }
    else if (syn_type == 2)
    {
        // Инициализируем мьютекс
        err = pthread_mutex_init(&mutex, NULL);
        if (err != 0)
            err_exit(err, "Cannot initialize mutex");
    }

    // Начинаем замер времени
    auto begin = chrono::steady_clock::now();

    // Создаём потоки
    for (int i = 0; i < threads_number; i++)
    {
        // Создаём i-тый поток
        err = pthread_create(&(threads[i]), NULL, thread_job, NULL);

        // Если при создании потока произошла ошибка, выводим
        // сообщение об ошибке и прекращаем работу программы
        if (err != 0)
        {
            cout << "Cannot create a thread number " << i << ":" << strerror(err) << endl;
            exit(-1);
        }
    }

    // Ожидаем завершения созданных потоков перед завершением
    // работы программы
    for (int i = 0; i < threads_number; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // Завершаем замер времени
    auto end = chrono::steady_clock::now();
    auto f_time = chrono::duration_cast<std::chrono::microseconds>(end - begin);

    double itog = f_time.count();

    cout << "Thread Creation Time : " << itog / 1000000 << endl;

    // Освобождаем ресурсы, связанные с мьютексом
    pthread_mutex_destroy(&mutex);
    // Освобождаем ресурсы, связанные со спинлоком
    pthread_spin_destroy(&lock);
}