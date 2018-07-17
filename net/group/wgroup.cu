#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#define FLT_MAX 1e35
#include <cassert>

__device__ inline bool isvalidxy(const int h, const int w,const int y,const int x)
{
    return (y >= 0) && (x >= 0) && (y < h) && (x < w);
}

__device__ inline void swapf(float & a, float & b)
{
    float tmp = a;
    a = b;
    b = tmp;
}

__device__ inline void swap(int & a, int & b)
{
    int tmp = a;
    a = b;
    b = tmp;
}

__device__ inline void bitonic_sort(float* dist, int* idx, int tmpn)
{   
    //Bitonic Sort
    for (unsigned int t = 2; t <= tmpn ; t <<= 1)
    {
    // Bitonic merge:
        for (unsigned int j = t >> 1; j>0; j >>= 1)
        {	
            for (unsigned int tid = threadIdx.x ; tid < tmpn ; tid += blockDim.x )
            {
                unsigned int ixj = tid ^ j;
                if (ixj > tid)
                {
                    if ((tid & t) == 0)
                    {
                        if (dist[tid] > dist[ixj])
                        {
                            swapf(dist[tid], dist[ixj]);
                            swap(idx[2*tid], idx[2*ixj]);
                            swap(idx[2*tid+1], idx[2*ixj+1]);
                        }
                    }
                    else
                    {
                        if (dist[tid] < dist[ixj])
                        {
                            swapf(dist[tid], dist[ixj]);
                            swap(idx[2*tid], idx[2*ixj]);
                            swap(idx[2*tid+1], idx[2*ixj+1]);
                        }
                    }
                }
                    		
            }
            __syncthreads();
        }
    }
}

__device__ inline void oddeven_sort(float* dist, int* idx , int tmpn)
{   
    for ( int cnt = 0 ; cnt < ( tmpn + 1 ) / 2 ; ++cnt )
    {
        for ( int j = 2*threadIdx.x + 1 ; j < tmpn ; j += 2*blockDim.x )
        {
            if ( dist[j] < dist[ j - 1 ] )
            {
                swapf(dist[j], dist[j-1]);
                swap(idx[2*j], idx[2*(j-1)]);
                swap(idx[2*j+1], idx[2*(j-1)+1]);
            }
        }
        __syncthreads();
        for ( int j = 2*threadIdx.x + 2 ; j < tmpn ; j += 2*blockDim.x )
        {
            if ( dist[j] < dist[ j - 1 ] )
            {
                swapf(dist[j], dist[j-1]);
                swap(idx[2*j], idx[2*(j-1)]);
                swap(idx[2*j+1], idx[2*(j-1)+1]);
            }   
        }
        __syncthreads();
    }
}

__global__ void KnnKernel1(const int b,const int h,const int w,const int d,const int dh,const int dw,const int tmpn,const int k,const float * f,float* tmpd,int* tmpi,float * result,int * result_i)
{
    int bi = blockIdx.x / h ;
    int y = blockIdx.x % h  ;
    int x = blockIdx.y ;
    const float* fcurrent = f + ((bi*h + y)*w+x)*d;
    float* dist = tmpd + ((bi*h + y)*w+x)*tmpn;
    int* idx = tmpi + ((bi*h + y)*w+x)*tmpn*2;
    float dif;
    float * r = result + ((bi*h + y)*w+x)*k;
    int * ri = result_i + ((bi*h + y)*w+x)*k*3;
    for( int j = threadIdx.x ; j < tmpn ; j += blockDim.x )
    {
        dist[j] = 3.33;
        if( j >= (2*dh+1)*(2*dw+1) )
        {
            dist[j] = FLT_MAX;
            idx[2*j] = INT_MAX;
            idx[2*j+1] = INT_MAX;
            continue;
        }
        int dy = j / (2*dh+1) - dh ;
        int dx = j % (2*dh+1) - dw;
        if( (dx==0) && (dy==0) )
        {
            dist[j] = 0.0;
            idx[2*j] = y;
            idx[2*j+1] = x;
            continue;
        }
        if( ! isvalidxy(h,w, y+dy , x+dx ) )
        {
            dist[j] = FLT_MAX;
            idx[2*j] = y + dy;
            idx[2*j+1] = x + dx;
            continue;
        }
        dist[j] = 0.0;
        for ( int di = 0 ; di < d ; ++di )
        {
            dif = fcurrent[di] - f[((bi*h+(y+dy))*w+(x+dx))*d+di];
            dist[j] += dif*dif;
        }
        idx[2*j] = y + dy;
        idx[2*j+1] = x + dx;
    }
    __syncthreads();
    bitonic_sort(dist,idx,tmpn);
    //copy result
    //float * r = result + ((bi*h + y)*w+x)*k;
    //int * ri = result_i + ((bi*h + y)*w+x)*k*3;
    for ( int ki = threadIdx.x ; ki < k  ; ki += blockDim.x )
    {
        r[ki] = dist[ki];
        ri[3*ki+0] = bi;
        ri[3*ki+1] = idx[2*ki];
        ri[3*ki+2] = idx[2*ki+1];
        //debug
    }
}

__global__ void KnnKernel2(const int b,const int h,const int w,const int d,const int dh,const int dw,const int tmpn,const int k,const float * f,float* tmpd,int* tmpi,float * result,int * result_i)
{
    int bi = blockIdx.x ;
    int y = blockIdx.y / w  ;
    int x = blockIdx.y % w  ;
    const float* fcurrent = f + ((bi*h + y)*w+x)*d;
    float* dist = tmpd + ((bi*h + y)*w+x)*tmpn;
    int* idx = tmpi + ((bi*h + y)*w+x)*tmpn*2;
    float dif;
    float * r = result + ((bi*h + y)*w+x)*k;
    int * ri = result_i + ((bi*h + y)*w+x)*k*3;
    for( int j = threadIdx.x ; j < tmpn ; j += blockDim.x )
    {
        dist[j] = 3.33;
        if( j >= (2*dh+1)*(2*dw+1) )
        {
            dist[j] = FLT_MAX;
            idx[2*j] = INT_MAX;
            idx[2*j+1] = INT_MAX;
            continue;
        }
        int dy = j / (2*dh+1) - dh ;
        int dx = j % (2*dh+1) - dw;
        if( (dx==0) && (dy==0) )
        {
            dist[j] = 0.0;
            idx[2*j] = y;
            idx[2*j+1] = x;
            continue;
        }
        if( ! isvalidxy(h,w, y+dy , x+dx ) )
        {
            dist[j] = FLT_MAX;
            idx[2*j] = y + dy;
            idx[2*j+1] = x + dx;
            continue;
        }
        dist[j] = 0.0;
        for ( int di = 0 ; di < d ; ++di )
        {
            dif = fcurrent[di] - f[((bi*h+(y+dy))*w+(x+dx))*d+di];
            dist[j] += dif*dif;
        }
        idx[2*j] = y + dy;
        idx[2*j+1] = x + dx;
    }
    __syncthreads();
    bitonic_sort(dist,idx,tmpn);
    //copy result
    //float * r = result + ((bi*h + y)*w+x)*k;
    //int * ri = result_i + ((bi*h + y)*w+x)*k*3;
    for ( int ki = threadIdx.x ; ki < k  ; ki += blockDim.x )
    {
        r[ki] = dist[ki];
        ri[3*ki+0] = bi;
        ri[3*ki+1] = idx[2*ki];
        ri[3*ki+2] = idx[2*ki+1];
        //debug
    }
}

void KnnKernelLauncher(const int b,const int h,const int w,const int d,const int dh,const int dw,const int tmpn,const int k,const float * f,float* tmpd,int* tmpi,float * result,int * result_i){
    if( (b*h<=65536) && (w<=65536)  )
    {
        KnnKernel1<<<dim3(b*h,w,1),512>>>(b,h,w,d,dh,dw,tmpn,k,f,tmpd,tmpi,result,result_i);
    }else if( (b<=65536) && (h*w<=65536) ){
        KnnKernel2<<<dim3(b,h*w,1),512>>>(b,h,w,d,dh,dw,tmpn,k,f,tmpd,tmpi,result,result_i);
    }else{
        assert( ((b*h<=65536)&&(w<=65536)) || ((b<=65536)&&(h*w<=65536)) );
    }
}
#endif