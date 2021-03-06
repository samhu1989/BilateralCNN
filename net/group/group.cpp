#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
using namespace tensorflow;
REGISTER_OP("Knn")
	.Input("xyz: float32")
	.Attr("k: int")
	.Output("dist: float32")
	.Output("idx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      int k;
      TF_RETURN_IF_ERROR(c->GetAttr("k", &k));
      ::tensorflow::shape_inference::DimensionOrConstant dimk(k);
      ::tensorflow::shape_inference::DimensionOrConstant dim2(2);
      ::tensorflow::shape_inference::ShapeHandle outshape0;
      TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(0),2,c->MakeDim(dimk),&outshape0));
      ::tensorflow::shape_inference::ShapeHandle outshape1;
      TF_RETURN_IF_ERROR(c->Concatenate(outshape0,c->Vector(dim2),&outshape1));
      c->set_output(0,outshape0);
      c->set_output(1,outshape1);
      return Status::OK();
    });


static void knnsearch(int b,int n,int dim,const float * xyz,const int k,float * dist,int * idx){
	for (int i=0;i<b;i++){
		for (int j=0;j<n;j++){
			for (int idxi = 0 ; idxi < k ; idxi++ )
			{
				dist[(i*n+j)*k+idxi] = -1;
			}
			for (int t=0;t<n;t++){
                float d = 0.0;
                for(int dimi=0;dimi<dim;++dimi)
				{
                    float dif = xyz[(i*n+t)*dim+dimi] - xyz[(i*n+j)*dim+dimi];
                    d += dif*dif;
                }
				if( dist[(i*n+j)*k+k-1] >= 0 &&  dist[(i*n+j)*k+k-1] < d )continue;
				dist[(i*n+j)*k+k-1] = d;
				idx[((i*n+j)*k+k-1)*2+0] = i;
                idx[((i*n+j)*k+k-1)*2+1] = t;
				float* current_d = &(dist[(i*n+j)*k+k-1]);
				int* current_bi = &(idx[((i*n+j)*k+k-1)*2+0]);
                int* current_i = &(idx[((i*n+j)*k+k-1)*2+1]);
				for(int idxi = k - 2 ; idxi >= 0 ; idxi-- )
				{
					if ( dist[(i*n+j)*k+idxi] < 0 || dist[(i*n+j)*k+idxi] > *current_d )
					{
						float tmpd = *current_d;
						int tmpi = *current_i;
                        int tmpbi = *current_bi;
                        
						*current_d = dist[(i*n+j)*k+idxi];
						*current_bi = idx[((i*n+j)*k+idxi)*2+0];
                        *current_i = idx[((i*n+j)*k+idxi)*2+1];
                        
						current_d = &(dist[(i*n+j)*k+idxi]);
                        current_bi = &(idx[((i*n+j)*k+idxi)*2+0]);
						current_i = &(idx[((i*n+j)*k+idxi)*2+1]);
                        
						*current_d = tmpd;
                        *current_bi = tmpbi;
						*current_i = tmpi;
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
			OP_REQUIRES_OK(context,context->GetAttr("k", &k));
		}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz_tensor = context->input(0);
			OP_REQUIRES(context,xyz_tensor.dims()==3,errors::InvalidArgument("Knn requires input be of shape (batch,#points,#dim)"));
			int b=xyz_tensor.shape().dim_size(0);
			int n=xyz_tensor.shape().dim_size(1);
            int dim=xyz_tensor.shape().dim_size(2);
			OP_REQUIRES(context,k>0&&k<n,errors::InvalidArgument("Knn requires k be larger than 0 and smaller than point number but got k=",k," n=",n));
			auto xyz_flat=xyz_tensor.flat<float>();
			const float * xyz=&xyz_flat(0);

			Tensor* dist_tensor=NULL;
			Tensor* idx_tensor=NULL;

			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,k},&dist_tensor));
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n,k,2},&idx_tensor));
            
			auto dist_flat = dist_tensor->flat<float>();
			auto idx_flat = idx_tensor->flat<int>();
			float * dist = &(dist_flat(0));
			int * idx = &(idx_flat(0));
			knnsearch(b,n,dim,xyz,k,dist,idx);
		}
	private:
		int k;
};
REGISTER_KERNEL_BUILDER(Name("Knn").Device(DEVICE_CPU), KnnOp);

void KnnKernelLauncher(const int b,const int subn,const int n,const int dim,const float * xyz,const int k,float* tmpd,int* tmpi,float * result,int * result_i);

class KnnGpuOp : public OpKernel{
	public:
		explicit KnnGpuOp(OpKernelConstruction* context):OpKernel(context){
			k = 0;
			OP_REQUIRES_OK(context,context->GetAttr("k", &k));
		}
		void Compute(OpKernelContext * context)override{
			const Tensor& xyz_tensor = context->input(0);
			const Tensor& k_tensor = context->input(1);
			
			OP_REQUIRES(context,xyz_tensor.dims()==3,errors::InvalidArgument("Knn requires input be of shape (batch,#points,#dim)"));
			int b=xyz_tensor.shape().dim_size(0);
			int n=xyz_tensor.shape().dim_size(1);
            int dim=xyz_tensor.shape().dim_size(2);
            int subn = n > 512 ? 512 : n ;
			OP_REQUIRES(context,k>0&&k<n,errors::InvalidArgument("Knn requires k be larger than 0 and smaller than point number but got k=",k," n=",n));

			auto xyz_flat=xyz_tensor.flat<float>();
			const float * xyz=&xyz_flat(0);
			
			Tensor * dist_tensor=NULL;
			Tensor * idx_tensor=NULL;
			OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,k},&dist_tensor));
			OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n,k,2},&idx_tensor));
			auto dist_flat=dist_tensor->flat<float>();
			auto idx_flat=idx_tensor->flat<int>();
			float * dist = &(dist_flat(0));
			int * idx = &(idx_flat(0));
            
            Tensor tmpd_tensor;
            OP_REQUIRES_OK(context,context->allocate_temp(DT_FLOAT,TensorShape{b,subn,n},&tmpd_tensor));
            auto tmpd_flat=tmpd_tensor.flat<float>();
            float* tmpd=&(tmpd_flat(0));
            
            Tensor tmpi_tensor;
            OP_REQUIRES_OK(context,context->allocate_temp(DT_INT32,TensorShape{b,subn,n},&tmpi_tensor));
            auto tmpi_flat=tmpi_tensor.flat<int>();
            int* tmpi=&(tmpi_flat(0));
            
			KnnKernelLauncher(b,subn,n,dim,xyz,k,tmpd,tmpi,dist,idx);
		}
	private:
		int k;
};
REGISTER_KERNEL_BUILDER(Name("Knn").Device(DEVICE_GPU), KnnGpuOp);
