/* Copyright 2023 The DeepRec Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/util.h"


namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("ElasticPartition")
    .Input("data: TKey")
    .Input("indices: int32")
    .Output("p_data: num_partitions * TKey")
    .Output("p_indices: num_partitions * int32")
    .Attr("num_partitions: int")
    .Attr("TKey: {int64, int32}")
    .SetShapeFn([](InferenceContext* c) {
      int64 num_partitions;
      TF_RETURN_IF_ERROR(c->GetAttr("num_partitions", &num_partitions));

      ShapeHandle data_shape = c->input(0);

      // The partition shape is dynamic in the 0th dimension, and matches
      // data_shape in the remaining dimensions.
      ShapeHandle unknown_dim0 = c->MakeShape({c->UnknownDim()});

      const int64 rank = c->Rank(data_shape);
      ShapeHandle data_suffix_shape;
      TF_RETURN_IF_ERROR(c->Subshape(data_shape, rank, &data_suffix_shape));
      ShapeHandle result_shape;
      TF_RETURN_IF_ERROR(
          c->Concatenate(unknown_dim0, data_suffix_shape, &result_shape));

      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, result_shape);
      }

      return Status::OK();
    });

template<typename TKey>
class ElasticPartitionOp : public OpKernel {
  public:
    explicit ElasticPartitionOp(OpKernelConstruction* c) : OpKernel(c) {
      OP_REQUIRES_OK(c, c->GetAttr("num_partitions", &num_partitions_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor* data;
        const Tensor* indices;
        Tensor partitions;
        OpOutputList data_output, indices_output;
        ValidateAndAllocateOutputs(ctx, &data, &indices, &partitions,
                                   &data_output, &indices_output);
        if (!ctx->status().ok()) return;
        if (num_partitions_ == 0 || data->NumElements() == 0) return;

        auto e_partitions = partitions.flat<int32>();
        const int64 N = e_partitions.dimension(0);
        gtl::InlinedVector<int, 32> output_index(num_partitions_);

        if (partitions.dims() == data->dims()) {
            // Walk through data and copy the data to the appropriate output tensor
            const auto data_flat = data->flat<TKey>();
            std::vector<Eigen::TensorMap<Eigen::Tensor<TKey, 1, Eigen::RowMajor>,
                                        Eigen::Aligned> >
                out_vec;
            out_vec.reserve(num_partitions_);
            for (int p = 0; p < num_partitions_; p++) {
                out_vec.push_back(data_output[p]->vec<TKey>());
            }
            for (int64 i = 0; i < N; i++) {
                const int32 p = internal::SubtleMustCopy(e_partitions(i));
                auto oi = output_index[p];
                OP_REQUIRES(ctx, FastBoundsCheck(oi, out_vec[p].size()),
                            errors::InvalidArgument(
                                "out_vec[", p, "] size: ", out_vec[p].size(),
                                " is not LTE output_index[", p, "] : ", oi));
                out_vec[p](oi) = data_flat(i);
                output_index[p]++;
            }
        } else {
            // If data has extra dimensions, use Eigen slices
            std::vector<Eigen::TensorMap<Eigen::Tensor<TKey, 2, Eigen::RowMajor>,
                                        Eigen::Aligned> >
                out_flat;
            out_flat.reserve(num_partitions_);
            for (int p = 0; p < num_partitions_; p++) {
                out_flat.push_back(data_output[p]->flat_outer_dims<TKey>());
            }

            // Walk through data and copy the data to the appropriate output tensor
            const int64 slice_size = data->NumElements() / N;
            const auto data_flat = data->shaped<TKey, 2>({N, slice_size});
            Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, slice_size);
            for (int64 i = 0; i < N; i++) {
                // outputs[p][output_index[p]++] = data[i]
                const int32 p = internal::SubtleMustCopy(e_partitions(i));
                auto oi = output_index[p];
                OP_REQUIRES(ctx, FastBoundsCheck(oi, out_flat[p].dimension(0)),
                            errors::InvalidArgument("Size of output_index: ", oi,
                                                    " is no longer in range."));
                Eigen::DSizes<Eigen::DenseIndex, 2> out_indices(oi, 0);
                Eigen::DSizes<Eigen::DenseIndex, 2> data_indices(i, 0);
                out_flat[p].slice(out_indices, sizes) =
                    data_flat.slice(data_indices, sizes);
                output_index[p]++;
            }
        }
    }

    void ValidateAndAllocateOutputs(OpKernelContext* c, const Tensor** data,
                                    const Tensor** indices, Tensor* partitions,
                                    OpOutputList* Tdata, OpOutputList* Tindices) {
        OP_REQUIRES_OK(c, c->input("data", data));
        OP_REQUIRES_OK(c, c->input("indices", indices));
        OP_REQUIRES(
            c,
            TensorShapeUtils::StartsWith((*data)->shape(), (*indices)->shape()),
            errors::InvalidArgument(
                "data.shape must start with partitions.shape, ",
                "got data.shape = ", (*data)->shape().DebugString(),
                ", partitions.shape = ", (*indices)->shape().DebugString()));
        int N = (*data)->dim_size(0);
        AllocatorAttributes attr;
        attr.set_on_host(true);
        OP_REQUIRES_OK(c, c->allocate_temp(DT_INT32, TensorShape({static_cast<int64>(N)}), partitions, attr));
        auto d_partitions = partitions->flat<int32>().data();

        CalculateModulo((*data)->flat<TKey>().data(), N, d_partitions);
        // Count how many occurrences of each partition id we have in partitions
        gtl::InlinedVector<int, 32> partition_count(num_partitions_);
        for (int64 i = 0; i < N; i++) {
            const int32 p = internal::SubtleMustCopy(d_partitions[i]);
            OP_REQUIRES(c, FastBoundsCheck(p, num_partitions_),
                        errors::InvalidArgument(
                            "partitions", p, " is not in [0, ", num_partitions_, ")"));
            partition_count[p]++;
        }

        // Allocate output tensors of the right size
        OP_REQUIRES_OK(c, c->output_list("p_data", Tdata));
        OP_REQUIRES_OK(c, c->output_list("p_indices", Tindices));
        for (int p = 0; p < num_partitions_; p++) {
            TensorShape shape;
            shape.AddDim(partition_count[p]);
            for (int i = 1; i < (*data)->dims(); i++) {
                shape.AddDim((*data)->dim_size(i));
            }
            Tensor* out;
            OP_REQUIRES_OK(c, Tdata->allocate(p, shape, &out));
            OP_REQUIRES_OK(c, Tindices->allocate(p, shape, &out));
        }
    }
    
    void CalculateModulo(const int64* data, int nums, int32* partitions) {
      
    }

    void CalculateModulo(const int32* data, int nums, int32* partitions) {
        
    }


  private:
    int num_partitions_;
};

#define REGISTER_CPU_KERNELS(key_type)                              \
  REGISTER_KERNEL_BUILDER(Name("ElasticPartition")                  \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<key_type>("TKey"),    \
                          ElasticPartitionOp<key_type>)

REGISTER_CPU_KERNELS(int32);
REGISTER_CPU_KERNELS(int64);
#undef REGISTER_CPU_KERNELS

} // namespace tensorflow