/*
* @author: 孙明志
* @mail: 531483935@qq.com
* @date: 2024-05-27
*/

#include "kernel_operator.h"
#include <type_traits>
constexpr int32_t BUFFER_NUM = 2;
template<typename T> class KernelInstanceNorm {
public:
    __aicore__ inline KernelInstanceNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance, uint64_t totalSize[], uint64_t batchSize[], uint64_t stepSize[], float epsilon) {
        this->maxbatchSize = 0;
        this->maxstepSize = 0;
        this->maxtotalSize = 0;
        for (int i = 0; i < 3; ++i) {
            if (batchSize[i] > this->maxbatchSize)
                this->maxbatchSize = batchSize[i];
            if (stepSize[i] > this->maxstepSize)
                this->maxstepSize = stepSize[i]; 
            if (totalSize[i] > this->maxtotalSize)
                this->maxtotalSize = totalSize[i];  
            this->batchSize[i] = batchSize[i];
            this->stepSize[i] = stepSize[i];
            this->squareSize[i] = totalSize[i] / batchSize[i] / stepSize[i];
            this->batchOffset[i] = totalSize[i] / batchSize[i];
        }
        this->epsilon = epsilon;
        this->maxsquareSize = maxtotalSize / maxbatchSize / maxstepSize;
        Gm_x.SetGlobalBuffer((__gm__ T*)x, maxtotalSize);
        Gm_gamma.SetGlobalBuffer((__gm__ T*)gamma, maxtotalSize);
        Gm_beta.SetGlobalBuffer((__gm__ T*)beta, maxtotalSize);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, maxtotalSize);
        Gm_mean.SetGlobalBuffer((__gm__ T*)mean, maxbatchSize * maxstepSize);
        Gm_variance.SetGlobalBuffer((__gm__ T*)variance, maxbatchSize * maxstepSize);
    }
    __aicore__ inline void Process() {
        for (uint64_t i = 0; i < maxbatchSize; ++i) {
            for (uint64_t j = 0; j < maxstepSize; ++j) {
                float sum = 0.0;
                for (uint64_t k = 0; k < maxsquareSize; ++k) {
                    float val = Gm_x.GetValue(i * maxsquareSize * maxstepSize + k * maxstepSize + j);
                    sum += val;
                }
                float avg = sum / maxsquareSize;
                Gm_mean.SetValue(i * maxstepSize + j, (T)avg);
            }
        }
        for (uint64_t i = 0; i < maxbatchSize; ++i) {
            for (uint64_t j = 0; j < maxstepSize; ++j) {
                float avg = Gm_mean.GetValue(i * maxstepSize + j);
                float sum = 0.0;
                for (uint64_t k = 0; k < maxsquareSize; ++k) {
                    float val = Gm_x.GetValue(i * maxsquareSize * maxstepSize + k * maxstepSize + j);
                    sum += (val - avg) * (val - avg);
                }
                float var = sum / maxsquareSize;
                Gm_variance.SetValue(i * maxstepSize + j, (T)var);
            }
        }
        for (uint64_t i = 0; i < maxbatchSize; ++i) {
            for (uint64_t j = 0; j < maxstepSize; ++j) {
                float mean = Gm_mean.GetValue(i * maxstepSize + j);
                float variance = Gm_variance.GetValue(i * maxstepSize + j);
                float sum = 0.0;
                for (uint64_t k = 0; k < maxsquareSize; ++k) {
                    auto index = i * maxsquareSize * maxstepSize + k * maxstepSize + j;
                    float x = Gm_x.GetValue(index);
                    float gamma = Gm_gamma.GetValue(i % batchSize[1] * batchOffset[1] + k % squareSize[1] * stepSize[1] + j % stepSize[1]);
                    float beta = Gm_beta.GetValue(i % batchSize[2] * batchOffset[2] + k % squareSize[2] * stepSize[2] + j % stepSize[2]);
                    float result = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta;
                    Gm_y.SetValue(index, (T)result);
                }
            }
        }
    }
private:
    AscendC::GlobalTensor<T> Gm_x, Gm_gamma, Gm_beta, Gm_y, Gm_mean, Gm_variance;
    uint64_t maxtotalSize, maxbatchSize, maxstepSize, maxsquareSize;
    uint64_t batchSize[3], squareSize[3], stepSize[3], batchOffset[3];
    float epsilon;
};

template<typename T> __aicore__ inline void GroupReduce(const AscendC::LocalTensor<T> &y, const AscendC::LocalTensor<T> &x, int32_t group_size, int32_t group_count) {
    static constexpr int32_t SIZE = sizeof(T);
    static constexpr int32_t ALIGN = 32 / SIZE;
    const int32_t factor = group_size / (group_size & -group_size);
    int32_t number = (256 / SIZE) / factor;
    number |= (number >> 1);
    number |= (number >> 2);
    number |= (number >> 4);
    int32_t reduceCount = (number ^ (number >> 1)) * factor;
    if (group_size / reduceCount > 1) {
        if (group_size / reduceCount % ALIGN) {
            reduceCount /= ALIGN * reduceCount / group_size;
        }
        int32_t repeatTimes = group_count * group_size / reduceCount;
        int32_t repStride = (reduceCount * SIZE - 1) / 32 + 1;
        AscendC::WholeReduceSum(x, x, reduceCount, repeatTimes, 1, 1, repStride);
        group_size /= reduceCount;
    }
    AscendC::WholeReduceSum(y, x, group_size, group_count, 1, 1, group_size * SIZE / 32);
}
template<typename T> class KernelInstanceNorm_Fast {
public:
    __aicore__ inline KernelInstanceNorm_Fast() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance, uint64_t totalSize[], uint64_t batchSize[], uint64_t stepSize[], float epsilon, uint64_t packNumber) {
        this->maxbatchSize = 0;
        this->maxstepSize = 0;
        this->maxtotalSize = 0;
        for (int i = 0; i < 3; ++i) {
            if (batchSize[i] > this->maxbatchSize)
                this->maxbatchSize = batchSize[i];
            if (stepSize[i] > this->maxstepSize)
                this->maxstepSize = stepSize[i]; 
            if (totalSize[i] > this->maxtotalSize)
                this->maxtotalSize = totalSize[i];  
            this->batchSize[i] = batchSize[i];
            this->stepSize[i] = stepSize[i];
            this->squareSize[i] = totalSize[i] / batchSize[i] / stepSize[i];
            this->batchOffset[i] = totalSize[i] / batchSize[i];
        }
        this->epsilon = epsilon;
        this->maxsquareSize = maxtotalSize / maxbatchSize;
        this->packNumber = packNumber;
        this->tileLength = this->packNumber * this->maxsquareSize;
        Gm_x.SetGlobalBuffer((__gm__ T*)x, maxtotalSize);
        Gm_gamma.SetGlobalBuffer((__gm__ T*)gamma, maxtotalSize);
        Gm_beta.SetGlobalBuffer((__gm__ T*)beta, maxtotalSize);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, maxtotalSize);
        Gm_mean.SetGlobalBuffer((__gm__ T*)mean, maxbatchSize);
        Gm_variance.SetGlobalBuffer((__gm__ T*)variance, maxbatchSize);
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(Q_tmp, BUFFER_NUM, this->tileLength * sizeof(T) * 2);
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(Q_mean, 1, this->maxbatchSize * sizeof(T));
        pipe.InitBuffer(Q_variance, 1, this->maxbatchSize * sizeof(T));
    }
    __aicore__ inline void Process() {
        auto cof = T(1.0f / maxsquareSize);
        AscendC::LocalTensor<T> mean = Q_mean.AllocTensor<T>();
        packNumber *= 2;
        tileLength *= 2;
        for (uint64_t i = 0; i < maxbatchSize; i += packNumber) {
            {
                AscendC::LocalTensor<T> x = Q_tmp.AllocTensor<T>();
                AscendC::DataCopy(x, Gm_x[i * maxsquareSize], tileLength);
                Q_tmp.EnQue(x);
            }
            {
                AscendC::LocalTensor<T> x = Q_tmp.DeQue<T>();
                GroupReduce(mean[i], x, maxsquareSize, packNumber);
                Q_tmp.FreeTensor(x);
            }
        }
        AscendC::Muls(mean, mean, cof, maxbatchSize);
        Q_mean.EnQue(mean);
        AscendC::LocalTensor<T> mean_out = Q_mean.DeQue<T>();
        AscendC::DataCopy(Gm_mean, mean_out, maxbatchSize);

        AscendC::LocalTensor<T> variance = Q_variance.AllocTensor<T>();
        for (uint64_t i = 0; i < maxbatchSize; i += packNumber) {
            {
                AscendC::LocalTensor<T> x = Q_tmp.AllocTensor<T>();
                AscendC::DataCopy(x, Gm_x[i * maxsquareSize], tileLength);
                Q_tmp.EnQue(x);
            }
            {
                AscendC::LocalTensor<T> x = Q_tmp.DeQue<T>();
                for (int j = 0; j < packNumber; ++j) {
                    float avg = mean_out.GetValue(i + j);
                    AscendC::Adds(x[j * maxsquareSize], x[j * maxsquareSize], T(-avg), maxsquareSize);
                }
                AscendC::Mul(x, x, x, tileLength);
                AscendC::Muls(x, x, cof, tileLength);
                GroupReduce(variance[i], x, maxsquareSize, packNumber);
                Q_tmp.FreeTensor(x);
            }
        }
        Q_variance.EnQue(variance);
        AscendC::LocalTensor<T> variance_out = Q_variance.DeQue<T>();
        AscendC::DataCopy(Gm_variance, variance_out, maxbatchSize);
        
        packNumber /= 2;
        tileLength /= 2;

        uint64_t broadcastSize = batchSize[1] < batchSize[2] ? batchSize[1] : batchSize[2];
        for (uint64_t z = 0; z < broadcastSize; z += packNumber) {
            {
                AscendC::LocalTensor<T> tmp = Q_tmp.AllocTensor<T>();
                AscendC::DataCopy(tmp, Gm_gamma[z * maxsquareSize], tileLength);
                AscendC::DataCopy(tmp[tileLength], Gm_beta[z * maxsquareSize], tileLength);
                Q_tmp.EnQue(tmp);
            }
            AscendC::LocalTensor<T> tmp = Q_tmp.DeQue<T>();
            for (uint64_t i = z; i < maxbatchSize; i += broadcastSize) {
                {
                    AscendC::LocalTensor<T> x = Q_x.AllocTensor<T>();
                    AscendC::DataCopy(x, Gm_x[i * maxsquareSize], tileLength);
                    Q_x.EnQue(x);
                }
                {
                    AscendC::LocalTensor<T> y = Q_y.AllocTensor<T>();
                    AscendC::LocalTensor<T> x = Q_x.DeQue<T>();
                    for (int j = 0; j < packNumber; ++j) {
                        float avg = mean_out.GetValue(i + j);
                        float var = variance_out.GetValue(i + j);
                        float deno = 1.0f / sqrt(var + epsilon);
                        AscendC::Adds(x[j * maxsquareSize], x[j * maxsquareSize], T(-avg), maxsquareSize);
                        AscendC::Muls(x[j * maxsquareSize], x[j * maxsquareSize], T(deno), maxsquareSize);
                    }
                    AscendC::Mul(x, x, tmp, tileLength);
                    AscendC::Add(y, x, tmp[tileLength], tileLength);

                    Q_y.EnQue<T>(y);
                    Q_x.FreeTensor(x);
                }
                {
                    AscendC::LocalTensor<T> y = Q_y.DeQue<T>();
                    AscendC::DataCopy(Gm_y[i * maxsquareSize], y, tileLength);
                    Q_y.FreeTensor(y);
                }
            }
            Q_tmp.FreeTensor(tmp);
        }
        Q_mean.FreeTensor(mean_out);
        Q_variance.FreeTensor(variance_out);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> Q_x, Q_tmp;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> Q_y, Q_mean, Q_variance;
    AscendC::GlobalTensor<T> Gm_x, Gm_gamma, Gm_beta, Gm_y, Gm_mean, Gm_variance;
    uint64_t tileLength, packNumber;
    uint64_t maxtotalSize, maxbatchSize, maxstepSize, maxsquareSize;
    uint64_t batchSize[3], squareSize[3], stepSize[3], batchOffset[3];
    float epsilon;
};
extern "C" __global__ __aicore__ void instance_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.stepSize[0] > 1 || tiling_data.totalSize[0] / tiling_data.batchSize[0] * sizeof(DTYPE_X) % 256 != 0) {
        KernelInstanceNorm<DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, variance, tiling_data.totalSize, tiling_data.batchSize, tiling_data.stepSize, tiling_data.epsilon);
        op.Process();
    }
    else {
        KernelInstanceNorm_Fast<DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, variance, tiling_data.totalSize, tiling_data.batchSize, tiling_data.stepSize, tiling_data.epsilon, tiling_data.packNumber);
        op.Process();
    }
}