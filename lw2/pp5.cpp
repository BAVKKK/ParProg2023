#include <cstdlib>
#include <iostream>
#include <cstring>
#include <pthread.h>
#include <unistd.h>


using namespace std;


void check_error(int);

#define check() check_error(err);

#define CHECK(command) err = command; check_error(err);

int err;
bool isSignal = false;

const int TASKS_COUNT = 10;
int task_list[TASKS_COUNT];
int current_task = 0;
pthread_mutex_t mutex, cond_mutex;


void signal(){
    CHECK( pthread_mutex_lock(&cond_mutex) );
    isSignal = true;
    CHECK( pthread_mutex_unlock(&cond_mutex) );
}


void wait(){
    bool blocked = true;
    while(blocked){
        CHECK( pthread_mutex_lock(&cond_mutex) );
        if(isSignal == true){
            blocked = isSignal = false;
        }
        CHECK( pthread_mutex_unlock(&cond_mutex) );
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
    cout << "mutex lock" << endl;
    CHECK( pthread_mutex_lock(&mutex) );
   
    cout << "waiting..." << endl;
    wait();
    cout << "signal recieved" << endl;


    cout << "mutex unlock" << endl;
    CHECK( pthread_mutex_unlock(&mutex) );
}


void *thread_job(void *arg)
{
    cout << "sleep 2 sec" << endl;
    CHECK( sleep(2) );
    cout << "send signal" << endl;
    signal();
}


int main()
{
    pthread_t thread1, thread2;
    int err;


    for (int i = 0; i < TASKS_COUNT; ++i)
        task_list[i] = rand() % TASKS_COUNT;
   
    CHECK( pthread_mutex_init(&mutex, NULL) );
   
    CHECK( pthread_create(&thread1, NULL, thread_job, NULL) );


    CHECK( pthread_create(&thread2, NULL, cond_job, NULL) );


    CHECK( pthread_join(thread1, NULL) );
    CHECK( pthread_join(thread2, NULL) );
   
    CHECK( pthread_mutex_destroy(&mutex) );
}
