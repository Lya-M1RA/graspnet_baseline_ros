#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
//#include <THC/THC.h>
//extern THCState *state;
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#endif



int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{
    // TODO check dimensions
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float *ref_dev = ref.data<float>();
    float *query_dev = query.data<float>();
    long *idx_dev = idx.data<long>();

    if (ref.is_cuda()) {
#ifdef WITH_CUDA
        // TODO raise error if not compiled with CUDA
        float *dist_dev;
        cudaError_t err = cudaMalloc((void**)&dist_dev, ref_nb * query_nb * sizeof(float));
        if (err != cudaSuccess) {
            printf("error in cudaMalloc: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("cudaMalloc failed");
        }

        for (int b = 0; b < batch; b++) {
            knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
                dist_dev, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
        }

        err = cudaFree(dist_dev);
        if (err != cudaSuccess) {
            printf("error in cudaFree: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("cudaFree failed");
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("error in knn: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("CUDA error in knn");
        }
        return 1;
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }

    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
        knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
            dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;
}
