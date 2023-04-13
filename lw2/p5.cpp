#include <cstdlib>
#include <iostream>
#include <cstring>
#include <pthread.h>
#include <unistd.h>


using namespace std;

#define err_exit(code, str)                   \
    {                                         \
        cerr << str << ": " << strerror(code) \
             << endl;                         \
        exit(EXIT_FAILURE);                   \
    }


bool isSignal = false;
const int TASKS_COUNT = 10;
int task_list[TASKS_COUNT];
int current_task = 0;
pthread_mutex_t mutex, cond_mutex;


void signal()
{
    int err;
    err = pthread_mutex_lock(&cond_mutex);
    if (err != 0)
    {
        err_exit(err, "Cannot lock mutex\n"); 
    }

    isSignal = true;
    err = pthread_mutex_unlock(&cond_mutex);
    if (err != 0)
    {
        err_exit(err, "Cannot unlock mutex\n"); 
    }
}


void wait()
{
    int err;
    bool blocked = true;
    while(blocked)
    {
        err = pthread_mutex_lock(&cond_mutex);
        if (err != 0)
        {
            err_exit(err, "Cannot lock mutex\n"); 
        }
        if(isSignal == true){
            blocked = isSignal = false;
        }
        err = pthread_mutex_unlock(&cond_mutex);
        if (err != 0)
        {
            err_exit(err, "Cannot unlock mutex\n"); 
        }
    }
}


void check_error(int err){
    if (err != 0)
    {
        cout << strerror(err) << endl;
        exit(-1);
    }
}


void *cond_job(void *arg)
{
    int err;
    cout << "mutex lock" << endl;
    err = pthread_mutex_lock(&mutex);
    if (err != 0)
    {
        err_exit(err, "Cannot lock mutex\n"); 
    }
   
    cout << "waiting..." << endl;
    wait();
    cout << "signal recieved" << endl;


    cout << "mutex unlock" << endl;
    err = pthread_mutex_unlock(&mutex);
    if (err != 0)
    {
        err_exit(err, "Cannot unlock mutex\n"); 
    }
}


void *thread_job(void *arg)
{
    cout << "sleep 2 sec" << endl;
    sleep(2);
    cout << "send signal" << endl;
    signal();  
}


int main()
{
    int err;
    pthread_t thread1, thread2;

    for (int i = 0; i < TASKS_COUNT; ++i)
        task_list[i] = rand() % TASKS_COUNT;
    
    err = pthread_mutex_init(&mutex, NULL) ;
    if (err != 0)
    {
        err_exit(err, "Cannot initialize mutex\n"); 
    }

    err = pthread_create(&thread1, NULL, thread_job, NULL) ;
    if (err != 0)
    {
        err_exit(err, "Cannot create thread 1\n"); 
    }

    err = pthread_create(&thread2, NULL, cond_job, NULL)  ;
    if (err != 0)
    {
        err_exit(err, "Cannot create thread 2\n"); 
    }

    err =  pthread_join(thread1, NULL);
    if (err != 0)
    {
        err_exit(err, "Cannot join thread 1\n"); 
    }

    err =  pthread_join(thread2, NULL);
    if (err != 0)
    {
        err_exit(err, "Cannot join thread 2\n"); 
    }
    
    
    err =  pthread_mutex_destroy(&mutex);
    if (err != 0)
    {
        err_exit(err, "Cannot join thread 2\n"); 
    }
}
