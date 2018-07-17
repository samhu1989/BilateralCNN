#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include<cassert>
__device__ inline void swapf(float & a, float & b)
{   
    a += b ;
    b = a - b;
    a -= b;
}

__device__ inline void swap(int & a, int & b)
{
    a += b ;
    b = a - b;
    a -= b;
}

__global__ void KnnKernel( int b, const int n,const int dim,const float * xyz,const int k,float* tmpd,int* tmpi,float * result,int * result_i)
{
    float* dist = tmpd + ( blockIdx.x + blockIdx.y*gridDim.x )*n;
    int* idx  = tmpi + ( blockIdx.x + blockIdx.y*gridDim.x )*n; 
    for ( int bi = blockIdx.x ; bi < b ; bi += gridDim.x )
    {
        for ( int i = blockIdx.y ;  i < n  ; i += gridDim.y )
        {
            for ( int j = threadIdx.x ; j < n ; j += blockDim.x )
            {
                if( i == j ){
                    dist[j] = 0;
                    idx[j]  = j;
                    continue;
                }
                dist[j] = 0.0;
                for ( int dimi = 0 ; dimi < dim ; ++dimi )
                {
                    float dif = xyz[(bi*n+i)*dim+dimi] - xyz[(bi*n+j)*dim+dimi];
                    dist[j] += dif*dif;
                }
                idx[j] = j;
            }
            __syncthreads();
            //odd-even sort
            int pownum = int(log2(float(n)));
            if ( n != pow(2, pownum) )
            {
                for ( int cnt = 0 ; cnt < ( n + 1 ) / 2 ; ++cnt )
                {
                    for ( int j = 2*threadIdx.x + 1 ; j < n ; j += 2*blockDim.x )
                    {
                        if ( dist[j] < dist[ j - 1 ] )
                        {
                            swapf(dist[j], dist[j-1]);
                            swap(idx[j], idx[j-1]);
                        }
                    }
                    __syncthreads();
                    for ( int j = 2*threadIdx.x + 2 ; j < n ; j += 2*blockDim.x )
                    {
                        if ( dist[j] < dist[ j - 1 ] )
                        {
                            swapf(dist[j], dist[j-1]);
                            swap(idx[j], idx[j-1]);
                        }   
                    }
                    __syncthreads();
                }
            }else{	
            //Bitonic Sort
                for (unsigned int t = 2; t <= n ; t *= 2)
                {
                    // Bitonic merge:
                    for (unsigned int j = t / 2; j>0; j /= 2)
                    {	
                        for (unsigned int tid = threadIdx.x ; tid < n ; tid += blockDim.x )
                    	{
                            unsigned int ixj = tid ^ j;
                    		if (ixj > tid)
                    		{
                        		if ((tid & t) == 0)
                        		{
                            			if (dist[tid] > dist[ixj])
                            			{
                                			swapf(dist[tid], dist[ixj]);
                                			swap(idx[tid], idx[ixj]);
                            			}
                        		}
                        		else
                        		{
                            			if (dist[tid] < dist[ixj])
                            			{
                                			swapf(dist[tid], dist[ixj]);
                                			swap(idx[tid], idx[ixj]);
                            			}
                        		}
                    		}
                    		
                        }
                        __syncthreads();	
                    }
                }
            }
            __syncthreads();
            //copy result
            for ( int j = threadIdx.x ; j < k  ; j += blockDim.x )
            {
                result[(bi*n+i)*k+j] = dist[j];
                result_i[ ((bi*n+i)*k+j)*2+0 ] = bi;
                result_i[ ((bi*n+i)*k+j)*2+1 ] = idx[j];
            }
            
        }
    }
}
void KnnKernelLauncher(const int b,const int subn, const int n,const int dim,const float * xyz,const int k,float* tmpd,int* tmpi,float * result,int * result_i){
    KnnKernel<<<dim3(b,subn,1),512>>>(b,n,dim,xyz,k,tmpd,tmpi,result,result_i);
}
#endif