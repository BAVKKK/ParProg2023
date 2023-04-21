#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <err.h>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <pthread.h>
#include <queue>
#include <chrono>


#define err_exit(code, str)                   \
    {                                         \
        cerr << str << ": " << strerror(code) \
             << endl;                         \
        exit(EXIT_FAILURE);                   \
    }

using namespace std;
char recieve[500];

// Структура параметров потока
struct thread_params
{
    int clientFd;
    int number;
};

enum store_state
{
    EMPTY,
    FULL
};
store_state state = EMPTY;

queue<thread_params> client_q; // очередь запросов
pthread_mutex_t mutex;         // мьютекс
pthread_cond_t cond;           // условная переменная

void *thread_job(void *arg)
{
    double sum = 0;
    while (true)
    {
        int err = pthread_mutex_lock(&mutex);

        while (state == EMPTY)
        {
            err = pthread_cond_wait(&cond, &mutex);
            if (err != 0)
                err_exit(err, "Cannot wait on condition variable");
        }
        auto begin = chrono::steady_clock::now();
        int client_fd = client_q.front().clientFd; //  индекс потока
        int number = client_q.front().number;      //  индекс потока

        client_q.pop();
        state = EMPTY;

        err = pthread_mutex_unlock(&mutex);

        string response = "HTTP/1.1 200 OK\r\n"
                          "Content-Type: text/html; charset=UTF-8\r\n\r\n"
                          "<!DOCTYPE html><html><head><title></title>"
                          "<body>Request number " +
                          to_string(number) +
                          " has been processed</body></html>\r\n";

        read(client_fd, &recieve, 500);
        write(client_fd, response.c_str(), 256); /*-1:'\0'*/

        auto end = chrono::steady_clock::now();
        auto f_time = chrono::duration_cast<std::chrono::microseconds>(end - begin);
        cout << "Time for request : " << sum << "ms" << '\n';
        close(client_fd);
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

    int threads_number = atoi(argv[1]); // количество потоков в пуле потоков
    pthread_t *threads = new pthread_t[threads_number];   
    int er;

    int one = 1, client_fd;
    struct sockaddr_in svr_addr, cli_addr;
    socklen_t sin_len = sizeof(cli_addr);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
        err(1, "can't open socket");

    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(int));

    int port = 8080;
    svr_addr.sin_family = AF_INET;
    svr_addr.sin_addr.s_addr = INADDR_ANY;
    svr_addr.sin_port = htons(port);

    if (bind(sock, (struct sockaddr *)&svr_addr, sizeof(svr_addr)) == -1)
    {
        close(sock);
        err(1, "Can't bind");
    }

    int n = 0;

    // Создаём потоки
    for (int i = 0; i < threads_number; i++)
    {
        // Создаём i-тый поток
        er = pthread_create(&(threads[i]), NULL, thread_job, NULL);

        if (er != 0)
        {
            cout << "Cannot create a thread number " << i << ":" << strerror(er) << endl;
            exit(-1);
        }
    }

    listen(sock, 20000);

    while (1)
    {

        client_fd = accept(sock, (struct sockaddr *)&cli_addr, &sin_len);

        if (client_fd == -1)  
        {
            perror("Can't accept");
            continue;
        }

        n++;
        state = FULL;

        thread_params p;

        p.clientFd = client_fd;
        p.number = n;

        client_q.push(p);

        er = pthread_cond_signal(&cond);
        if (er != 0)
            err_exit(er, "Cannot send signal");
    }
}
