#include <iostream>
#include "mpi.h"
#include <chrono>
#include <cmath>

using namespace std;

//Переменные и функции
const double a = 10e5; //Параметр уравнения
const double e = 10e-8; //Порог сходимости
const double phi0 = 0.0; //Начальное приближение

//Область моделирования
const double x_0 = -1.0;
const double y_0 = -1.0;
const double z_0 = -1.0;

const double Dx = 2.0;
const double Dy = 2.0;
const double Dz = 2.0;

//Размер сетки
int Nx = 160;
int Ny = 160;
int Nz = 160;

int sizeProc = 0;//Количество процессов
int rankProc = 0;//Номер процесса

//Функция φ
double phi(double x, double y, double z){
    return x*x + y*y + z*z;
}

//Правая часть
double ro(double x, double y, double z){
    return 6.0 - a*phi(x, y, z);
}

//Шаги сетки
const double Hx = Dx/(double)(Nx-1);
const double Hy = Dy/(double)(Ny-1);
const double Hz = Dz/(double)(Nz-1);

//Координаты узла
double xi (int i){
    return x_0 + i*Hx;
}

double yj (int j){
    return y_0 + j*Hy;
}

double zk (int k){
    return z_0 + k*Hz;
}

//Формула 2 
double nextPhi(double **phi_val, int k, int ij){

    double part1 = 1.0 / (2.0/(Hx*Hx)+2.0/(Hy*Hy)+2.0/(Hz*Hz)+a);

    double part2x = (phi_val[k][ij+1] + phi_val[k][ij-1])/(Hx*Hx);
    double part2y = (phi_val[k][ij+Ny] + phi_val[k][ij-Ny])/(Hy*Hy);
    double part2z = (phi_val[k+1][ij] + phi_val[k-1][ij])/(Hz*Hz);

    double x_ = xi(ij%Nx);
    double y_ = yj(ij/Nx);
    double z_ = zk(k + rankProc*Nz);

    double part2 = part2x + part2y + part2z - ro(x_, y_, z_);

    return part1*part2;
}

int main(int argc, char**argv){
    double startwtime = 0.0, endwtime;
    MPI_Init(&argc, &argv);//Инициализация

    MPI_Comm_size(MPI_COMM_WORLD, &sizeProc); //Записываем количество процессов
    
    if (Nz % sizeProc != 0){
    	cout << "ERROR:Process' number is too much or Process' number is not multiple to grid\n";
    	MPI_Abort(MPI_COMM_WORLD,-1);
    	MPI_Finalize();
    	return -1;
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rankProc); //Записываем номер текущего процесса

    int tag = 0;

    //Декомпозиция на каждый процесс
    //Используем декомпозицию на "линейке" по оси Z
    Nz = Nz/(double)sizeProc;

    //Массив значений
    double **phi_val = new double *[Nz];

    for (int i = 0; i < Nz; i++){
        phi_val[i] = new double[Ny*Nx];
    }

    //Массив значений (предыдущий)
    double **phi_val_pred = new double *[Nz];

    for (int i = 0; i < Nz; i++){
        phi_val_pred[i] = new double[Ny*Nx];
    }

    //Заполняем phi_val
    for (int k = 0; k < Nz; k++){
        for (int ij = 0; ij < Ny*Nx; ij++){
            if ((k == 0) || (k == Nz-1) || //Границы по Zk
                ((ij / Nx) == 0) || ((ij / Nx) == (Ny-1)) || //Границы по Yj
                ((ij % Nx) == 0) || ((ij % Nx) == (Nx-1))) //Границы по Xi
                {
                    //Граничные значения
                    double x_f = xi(ij%Nx);
                    double y_f = yj(ij/Nx);
                    double z_f = zk(k + rankProc*Nz);
                    phi_val[k][ij] = phi(x_f, y_f, z_f);
                } else {
                    //Внутренняя часть
                    phi_val[k][ij] = phi0;
                }
        }
    }

    //Вычисляем приближение до выполнения условния
    double max = 1.0;
    int it = 1; //Номер итерации
    startwtime = MPI_Wtime();
    while (max >= e)
    {

        //Запомним пердыдущие значение функций
        for (int k = 0; k < Nz; k++){
            for (int ij = 0; ij < Ny*Nx; ij++){
                phi_val_pred[k][ij] = phi_val[k][ij];
            }
        }

        //Вычисляем сеточные значения, прилегающие к границе локальной подобласти
        for (int k = 0; k < Nz; k++){
            for (int ij = 0; ij < Ny*Nx; ij++){
                if (
                    (((k == 1) || (k == Nz-2)) && ((ij / Nx) >= 1) && ((ij / Nx) <= (Ny-2)) && ((ij % Nx) >= 1) && ((ij % Nx) <= (Nx-2))) || //Прилегающие к границе по Zk
                    ((((ij / Nx) == 1)||((ij / Nx) == (Ny-2))) && (k >= 1) && (k <= (Nz-2)) && ((ij % Nx) >= 1) && ((ij % Nx) <= (Nx-2))) || //Прилегающие к границе по Yj
                    ((((ij % Nx) == 1)||((ij % Nx) == (Nx-2))) && (k >= 1) && (k <= (Nz-2)) && ((ij / Nx) >= 1) && ((ij / Nx) <= (Ny-2)))    //Прилегающие к границе по Xi
                   )
                    {
                        phi_val[k][ij] = nextPhi(phi_val_pred, k, ij);
                    }
            }
        }

        //Запускаем асинхронный обмен граничных значений
        MPI_Request req[4];
        MPI_Status sta[4];

        if (rankProc != 0){
            MPI_Isend(
                phi_val[1], Nx*Ny, MPI_DOUBLE, rankProc-1, tag,
                MPI_COMM_WORLD, &req[0]
            );
            MPI_Irecv(
                phi_val[0], Nx*Ny, MPI_DOUBLE, rankProc-1, tag,
                MPI_COMM_WORLD, &req[2]
            );
        }

        if (rankProc != sizeProc-1){
            MPI_Isend(
                phi_val[Nz-2], Nx*Ny, MPI_DOUBLE, rankProc+1, tag,
                MPI_COMM_WORLD, &req[1]
            );
            MPI_Irecv(
                phi_val[Nz-1], Nx*Ny, MPI_DOUBLE, rankProc+1, tag,
                MPI_COMM_WORLD, &req[3]
            );
        }

        //Выполняем вычисление остальных точек подобласти
        for (int k = 0; k < Nz; k++){
            for (int ij = 0; ij < Ny*Nx; ij++){
                if ((k > 1) && (k < Nz-2) && //Остальные точки по Zk
                    ((ij / Nx) > 1) && ((ij / Nx) < (Ny-2)) && //Остальные точки по Yj
                    ((ij % Nx) > 1) && ((ij % Nx) < (Nx-2))) //Остальные точки по Xi
                    {
                        phi_val[k][ij] = nextPhi(phi_val_pred, k, ij);
                    }
            }
        }

        //Ожидание завершения обменов
        if (rankProc != 0){
            MPI_Wait(&req[0], &sta[0]);
            MPI_Wait(&req[2], &sta[2]);
        }

        if (rankProc != sizeProc-1){
            MPI_Wait(&req[1], &sta[1]);
            MPI_Wait(&req[3], &sta[3]);
        }

        double max_i = 0.0;

        for (int k = 0; k < Nz; k++){
            for (int ij = 0; ij < Ny*Nx; ij++){
                    if (fabs(phi_val[k][ij] - phi_val_pred[k][ij]) > max_i){
                        max_i = fabs(phi_val[k][ij] - phi_val_pred[k][ij]);
                    }
            }
        }

        max = max_i;
        MPI_Allreduce(&max_i, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        //Результат
        if (rankProc == 0){
            cout << it << ") max(i,j,k) = " << max << endl;
        }
        it++;
    }
    endwtime = MPI_Wtime();

    //Считаем дельта
    double x_f;
    double y_f;
    double z_f;
    //Считаем истинное значение функции в точках
    for (int i = 0; i < Nz; i++){
    	for (int j = 0; j < Ny*Nx; j++){
    		//Значения точек и функции в точке
                x_f = xi(j%Nx);
                y_f = yj(j/Nx);
                z_f = zk(i + rankProc*Nz);
                phi_val_pred[i][j] = phi(x_f, y_f, z_f);

    	}
    }
    double max_i = 0.0;
    //Ищем максимальную по модулю разность
    for(int i=0; i < Nz; i++){
    	for (int j=0; j < Ny*Nx; j++){
    	    if(fabs(phi_val[i][j]-phi_val_pred[i][j])>max_i){
    	    	max_i = fabs(phi_val[i][j]-phi_val_pred[i][j]);
    	    }
        }
    }
    //Определим дельту из всех потоков и выберем наибульшую
    max = max_i;
    MPI_Allreduce(&max_i, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rankProc == 0){
        cout << "Delta = "<< max << endl;
        cout << "Время работы программы:" << endwtime - startwtime << endl;
    }
    //Очистим память
    for (int i = 0; i < Nz; i++){
        delete[] phi_val[i];
        delete[] phi_val_pred[i];
    }

    delete[] phi_val;
    delete[] phi_val_pred;

    MPI_Finalize();
    return 0;
}
