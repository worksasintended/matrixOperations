#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
 
// inverts Matrix of type Matrix
extern "C" {
// LU decomoposition of a general matrix
void dgetrf_(int *M, int *N, double *A, int *lda, int *IPIV, int *INFO);

// generate inverse of a matrix given its LU decomposition
void dgetri_(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork,
             int *INFO);
}

struct Matrix{
  std::vector<double> arr;
  int Nx;
  int Ny;

  Matrix(int size){
    arr.resize(size);
  }

  Matrix(int nx, int ny){
    Nx=nx;
    Ny=ny;
    arr.resize(Nx*Ny,0);
  }

  Matrix(std::vector<double> vec){
    //assume 1D array here
    arr=vec;
    Nx=1;
    Ny=arr.size();
  }

  Matrix(std::vector<double> vec, int nx, int ny):Nx(nx),Ny(ny){
    arr=vec;
  }

  Matrix(const Matrix& other){
    copy(other);
  }

  inline Matrix& copy(const Matrix& other){
    if(this==&other) return *this;
    arr=other.arr;
    Nx=other.Nx;
    Ny=other.Ny;
    return *this;
  }

  //some useful operators here for division/multiplication
  inline Matrix& operator =(const Matrix& other){
    return copy(other);
  }

  inline Matrix& operator *=(const double& d){
    std::for_each(arr.begin(),arr.end(),[&d](double& db){db*=d;});
    return *this;
  }

  inline Matrix& operator /=(const double& d){
    std::for_each(arr.begin(),arr.end(),[&d](double& db){db/=d;});
    return *this;
  }

  inline Matrix operator *(const double& d){
    Matrix result=*this;
    return result*=d;
  }

  inline Matrix operator /(const double& d){
    Matrix result=*this;
    return result/=d;
  }

  inline bool sanity_check(){
    return Nx*Ny == arr.size();
  }

  inline double& get(int x, int y){
    return arr[y*Nx+x];
  }

  inline double& get(int x){
    return arr[x];
  }

  inline Matrix& transpose(){
    if(Nx==Ny){
      for(auto x=0;x<Nx;++x){
	for(auto y=0;y<Ny;++y){
	  double tmp=get(x,y);
	  get(x,y) = get(y,x);
	  get(y,x) = tmp;
	}
      }
    }else{
      //FIXME do this without copying
      Matrix trans(Ny,Nx);
      for(auto y=0;y<Ny;++y){
	for(auto x=0;x<Nx;++x){
	  trans.get(y,x) = this->get(x,y);
	}
      }
      *this=trans;
    }
    return *this;
  }

  //operators for matrix multiplication
  inline Matrix matmul(const Matrix& other){
    //first check for matching dimensions
    if(this->Nx != other.Ny){
      std::cout<<"Matrix dimensions not matching for multiplication!"<<std::endl;
      return *this;
    }else{
      std::vector<double> result(other.Nx*Ny,0);
      for(auto i=0;i<Ny;++i){
	for(auto j=0;j<other.Nx;++j){
	  for(auto k=0;k<Nx;++k){
	    result[i*other.Nx+j] += arr[i*Nx+k] * other.arr[k*other.Nx+j];
	  }
	}
      }
      return Matrix(result,other.Nx,Ny);
    }
  }

  inline Matrix xTx(){
    std::vector<double> result(Nx*Nx,0);
    //ny is resulting y dimension, nx resulting x dimension
    for(auto ry=0;ry<Nx;++ry){
      for(auto rx=0;rx<Nx;++rx){
	for(auto y=0;y<Ny;++y){
	  result[ry*Nx+rx] += this->get(ry,y) * this->get(rx,y);
	}
      }
    }
    return Matrix(result,Nx,Nx);
  }

  inline Matrix invert(){
    if((Nx != Ny) || sanity_check()==false){
      std::cout<<"dimensions not matching for inversion!"<<std::endl;
      return Matrix(std::vector<double>(),0,0);
    }
    Matrix result = *this;
    int N = Nx;
    int *IPIV = new int[Nx];
    int LWORK = arr.size();
    double *WORK = new double[LWORK];
    int INFO;
    dgetrf_(&N, &N, result.arr.data(), &N, IPIV, &INFO);
    dgetri_(&N, result.arr.data(), &N, IPIV, WORK, &LWORK, &INFO);
    delete[] IPIV;
    delete[] WORK;
    return result;
  }

  void print(){
    std::cout<<std::endl;
    for(auto y=0;y<Ny;++y){
      for(auto x=0;x<Nx;++x){
	std::cout<<get(x,y)<<" ";
      }
      std::cout<<std::endl;
    }
  }

  void print(std::string fn){
    std::ofstream ofs(fn);
    for(auto y=0;y<Ny;++y){
      for(auto x=0;x<Nx;++x){
	ofs<<get(x,y)<<" ";
      }
      ofs<<std::endl;
    }
  }

  inline Matrix makeIdentity(){
    if((Nx != Ny) || sanity_check()==false){
	std::cout<<"dimensions not matching!"<<std::endl;
	return Matrix(std::vector<double>(),0,0);
    }
    std::fill(this->arr.begin(), this->arr.end(), 0.0);
    for(int i=0; i<Nx; ++i){
      this->arr[i+Nx*i]=1.;
    }
    return *this;
  } 
  
  inline Matrix cron(const Matrix& other){
    Matrix tempMatrix(Nx*other.Nx,Ny*other.Ny);
    for(int i=0; i<other.Nx; ++i){
    for(int h=0; h<other.Ny; ++h){
      for(int j=0; j<Nx; ++j){
	for(int k=0; k<Ny; ++k){
	  tempMatrix.arr[Nx*i+j+Nx*other.Nx*(Ny*h+k)]=other.arr[i+other.Nx*h]*arr[j+Nx*k];
	}
      }
    }
    }
    return tempMatrix;
  }
};
