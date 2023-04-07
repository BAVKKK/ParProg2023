#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <pthread.h>

using namespace std;

#define err_exit(code, str)                   \
    {                                         \
        cerr << str << ": " << strerror(code) \
             << endl;                         \
        exit(EXIT_FAILURE);                   \
    }

enum task_state
{
    EMPTY,
    FULL
};
task_state state = EMPTY;

const int TASKS_COUNT = 10;
int current_task = 0;

pthread_mutex_t mutex;
pthread_cond_t cond;

void *producer(void *arg)
{
    int err;
    
    while (current_task < TASKS_COUNT)
    {
        // Захватываем мьютекс и ожидаем решения задачи
        err = pthread_mutex_lock(&mutex);
        if (err != 0)
            err_exit(err, "Cannot lock mutex");

        while (state == FULL)
        {
            err = pthread_cond_wait(&cond, &mutex);
            if (err != 0)
                err_exit(err, "Cannot wait on condition variable");
        }

        // Получен сигнал, что задачи решены
        // Добавляем новую задачу
        current_task++;
        cout << "Add task " << current_task << endl;
        state = FULL;

        // Посылаем сигнал, что появилась новая задача
        err = pthread_cond_signal(&cond);
        if (err != 0)
            err_exit(err, "Cannot send signal");

        err = pthread_mutex_unlock(&mutex);
        if (err != 0)
            err_exit(err, "Cannot unlock mutex");
    }

    return NULL;
}
void *solving(void *arg)
{
    int err;
    
    while (current_task < TASKS_COUNT)
    {
        // Захватываем мьютекс и ожидаем появления нерешённых задач

        err = pthread_mutex_lock(&mutex);
        if (err != 0)
            err_exit(err, "Cannot lock mutex");

        while (state == EMPTY)
        {
            err = pthread_cond_wait(&cond, &mutex);
            if (err != 0)
                err_exit(err, "Cannot wait on condition variable");
        }

        // Получен сигнал, что появилась новая задача
        // Решаем её
        cout << "Task " << current_task << " ... ";
        sleep(1);
        cout << "done" << endl;
        state = EMPTY;

        err = pthread_cond_signal(&cond);
        if (err != 0)
            err_exit(err, "Cannot send signal");

        err = pthread_mutex_unlock(&mutex);
        if (err != 0)
            err_exit(err, "Cannot unlock mutex");
    }

    return NULL;
}
int main()
{
    int err;

    pthread_t thread1, thread2; // Идентификаторы потоков

    // Инициализируем мьютекс и условную переменную

    err = pthread_cond_init(&cond, NULL);
    if (err != 0)
        err_exit(err, "Cannot initialize condition variable");

    err = pthread_mutex_init(&mutex, NULL);
    if (err != 0)
        err_exit(err, "Cannot initialize mutex");

    // Создаём потоки

    err = pthread_create(&thread1, NULL, producer, NULL);
    if (err != 0)
        err_exit(err, "Cannot create thread 1");

    err = pthread_create(&thread2, NULL, solving, NULL);
    if (err != 0)
        err_exit(err, "Cannot create thread 2");

    // Дожидаемся завершения потоков
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    // Освобождаем ресурсы, связанные с мьютексом
    // и условной переменной
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
}