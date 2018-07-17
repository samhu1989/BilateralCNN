#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <limits>
using namespace tensorflow;
REGISTER_OP("WinKnn")
	.Input("f: float32")
    .Attr("k: int")
    .Attr("dh: int")
    .Attr("dw: int")
	.Output("dist: float32")
	.Output("idx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
      int k;
      TF_RETURN_IF_ERROR(c->GetAttr("k", &k));
      ::tensorflow::shape_inference::DimensionOrConstant dimk(k);
      ::tensorflow::shape_inference::DimensionOrConstant dim2(3);
      ::tensorflow::shape_inference::ShapeHandle outshape0;
      TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(0),3,c->MakeDim(dimk),&outshape0));
      ::tensorflow::shape_inference::ShapeHandle outshape1;
      TF_RETURN_IF_ERROR(c->Concatenate(outshape0,c->Vector(dim2),&outshape1));
      c->set_output(0,outshape0);
      c->set_output(1,outshape1);
      return Status::OK();
    });

    
static inline int iindex(
                 const int h,const int w,const int k,
    const int bi,const int y,const int x,const int ki,const int i)
{
    return ((( bi*h + y )*w + x ) * k + ki)*3+i;
}
    
static inline int dindex(
                 const int h,const int w,const int k,
    const int bi,const int y,const int x,const int ki)
{
    return (( bi*h + y )*w + x ) * k + ki;
}

static inline int findex(
                 const int h,const int w,const int d,
    const int bi,const int y,const int x,const int di)
{
    return (( bi*h + y )*w + x ) * d + di;
}

static inline bool isvalidfindex(const int h, const int w,const int y,const int x)
{
    return ( y >= 0) && ( x >= 0 ) && (y < h) && (x < w);
}

template<typename T>
void swap(T& a, T& b)
{   
    a += b ;
    b = a - b;
    a -= b;
}

static void wknnsearch(
    const int b,
    const int h,
    const int w,
    const int d,
    const int dh,
    const int dw,
    const int k,
    const float* f,
    float * dist,
    int * idx){
	for(int bi=0 ; bi < b ; bi++ ){
		for (int y=0 ; y < h ; y++){
			for (int x = 0 ; x < w ; x++ )
            {
                for (int ki = 0 ; ki < k ; ki++ )
                {
                    dist[dindex(h,w,k,bi,y,x,ki)] = -1;
                }
                const float* fcurrent = f + (( bi*h + y )*w + x ) * d;
                for( int dy= -dh ; dy <= dh ; dy ++ )
                for( int dx= -dw ; dx <= dw ; dx ++ )
                {
                    if( !isvalidfindex(h,w,y+dy,x+dx) )continue;
                    float dd = 0.0;
                    float dif = 0.0;
                    for(int di=0 ; di < d ; ++di)
                    {
                        dif = fcurrent[di] - f[findex(h,w,d,bi,y+dy,x+dx,di)];
                        //std::cout<<"dif("<<di<<")="<<dif<<std::endl;
                        dd += dif*dif;
                    }
                    //std::cout<<"d=["<<y<<"+("<<dy<<"),"<<x<<"+("<<dx<<")]="<<dd<<"\n";
                    float* disttmp = dist + (( bi*h + y )*w + x ) * k;
                    if( disttmp[k-1] >= 0 &&  disttmp[k-1] < dd )continue;
                    disttmp[k-1] = dd;
                    int* idxtmp = idx + ((( bi*h + y )*w + x ) * k)*3;
                    idxtmp[(k-1)*3] = bi ; 
                    idxtmp[(k-1)*3+1] = y+dy;
                    idxtmp[(k-1)*3+2] = x+dx;
                    float* current_d = disttmp + k - 1;
                    int* current_bi = idxtmp + (k-1)*3;
                    int* current_y = idxtmp  + (k-1)*3 + 1;
                    int* current_x = idxtmp  + (k-1)*3 + 2;
                    for(int ki = k - 2 ; ki >= 0 ; ki-- )
                    {
                        if ( disttmp[ki] < 0 || disttmp[ki] > *current_d )
                        {
                            swap<float>(*current_d,disttmp[ki]);
                            swap<int>(*current_bi,idxtmp[ki*3]);
                            swap<int>(*current_y,idxtmp[ki*3+1]);
                            swap<int>(*current_x,idxtmp[ki*3+2]);
                        
                            current_d = disttmp + ki;
                            current_bi = idxtmp + ki*3;
                            current_y = idxtmp  + ki*3 + 1;
                            current_x = idxtmp  + ki*3 + 2;
                        }
                    }
                }
                
            }
		}
	}
}

class KnnOp : public OpKernel{
	public:
		explicit KnnOp(OpKernelConstruction* context):OpKernel(context){
			k = 0;
            dh = 0;
            dw = 0;
            OP_REQUIRES_OK(context,context->GetAttr("k", &k));
            OP_REQUIRES_OK(context,context->GetAttr("dh", &dh));
            OP_REQUIRES_OK(context,context->GetAttr("dw", &dw));
		}
		void Compute(OpKernelContext * context)override{
			const Tensor& f_tensor = context->input(0);
			OP_REQUIRES(context,f_tensor.dims()==4,errors::InvalidArgument("Knn requires input be of shape (batch,#h,#w,#dim)"));
			int b = f_tensor.shape().dim_size(0);
			int h = f_tensor.shape().dim_size(1);
            int w = f_tensor.shape().dim_size(2);
            int d = f_tensor.shape().dim_size(3);
			OP_REQUIRES(context,k>0&&k<=dh*dw,errors::InvalidArgument("WKnn requires k be larger than 0 and no larger than dh*dw but got k=",k," dh*dw=",dh*dw));
			auto f_flat = f_tensor.flat<float>();
			const float * f = &f_flat(0);

			Tensor* dist_tensor=NULL;
			Tensor* idx_tensor=NULL;

			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,h,w,k},&dist_tensor));
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,h,w,k,3},&idx_tensor));
            
			auto dist_flat = dist_tensor->flat<float>();
			auto idx_flat = idx_tensor->flat<int>();
			float * dist = &(dist_flat(0));
			int * idx = &(idx_flat(0));
			wknnsearch(b,h,w,d,dh,dw,k,f,dist,idx);
		}
	private:
		int k;
        int dh;
        int dw;
};
REGISTER_KERNEL_BUILDER(Name("WinKnn").Device(DEVICE_CPU), KnnOp);

void KnnKernelLauncher(const int b,const int h,const int w,const int d,const int dh,const int dw,const int tmpn,const int k,const float * f,float* tmpd,int* tmpi,float * result,int * result_i);

//find the smalleast integer that is larger than n and is power of 2
static inline void power2(int & n)
{
    unsigned int x = (unsigned int)(n);
    x -= 1;
    x |= (x >>  1);
    x |= (x >>  2);
    x |= (x >>  4);
    x |= (x >>  8);
    x |= (x >> 16);
    x += 1;
    n = int(x);
}

class KnnGpuOp : public OpKernel{
	public:
		explicit KnnGpuOp(OpKernelConstruction* context):OpKernel(context){
			k = 0;
            dh = 0;
            dw = 0;
            OP_REQUIRES_OK(context,context->GetAttr("k", &k));
            OP_REQUIRES_OK(context,context->GetAttr("dh", &dh));
            OP_REQUIRES_OK(context,context->GetAttr("dw", &dw));
		}
		void Compute(OpKernelContext * context)override{
			const Tensor& f_tensor = context->input(0);
			OP_REQUIRES(context,f_tensor.dims()==4,errors::InvalidArgument("Knn requires input be of shape (batch,#h,#w,#d)"));
			int b = f_tensor.shape().dim_size(0);
			int h = f_tensor.shape().dim_size(1);
            int w = f_tensor.shape().dim_size(2);
            int d = f_tensor.shape().dim_size(3);
			OP_REQUIRES(context,k>0&&k<=dh*dw,errors::InvalidArgument("Knn requires k be larger than 0 and no larger than dh*dw but got k=",k," dh*dw=",dh*dw));

			auto f_flat=f_tensor.flat<float>();
			const float * f = &(f_flat(0));
			
			Tensor * dist_tensor = NULL;
			Tensor * idx_tensor = NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,h,w,k},&dist_tensor));
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,h,w,k,3},&idx_tensor));
			auto dist_flat=dist_tensor->flat<float>();
			auto idx_flat=idx_tensor->flat<int>();
			float * dist = &(dist_flat(0));
			int * idx = &(idx_flat(0));
            
            int tmpn = (2*dh+1)*(2*dw+1);
            power2(tmpn);
            
            Tensor tmpd_tensor;
            OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,TensorShape{b,w,h,tmpn},&tmpd_tensor));
            auto tmpd_flat=tmpd_tensor.flat<float>();
            float* tmpd=&(tmpd_flat(0));
            
            Tensor tmpi_tensor;
            OP_REQUIRES_OK(context,context->allocate_temp(DT_INT32,TensorShape{b,w,h,tmpn,2},&tmpi_tensor));
            auto tmpi_flat=tmpi_tensor.flat<int>();
            int* tmpi=&(tmpi_flat(0));
            
			KnnKernelLauncher(b,h,w,d,dh,dw,tmpn,k,f,tmpd,tmpi,dist,idx);
		}
	private:
		int k;
        int dh;
        int dw;
};
REGISTER_KERNEL_BUILDER(Name("WinKnn").Device(DEVICE_GPU), KnnGpuOp);
