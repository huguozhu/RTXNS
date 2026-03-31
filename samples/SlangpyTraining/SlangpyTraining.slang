// SPDX-License-Identifier: Apache-2.0
// clang-format off

/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/*
 系统架构概览
    这是一个端到端的神经纹理训练系统，包含：
        1. 可训练神经网络模型
        2. 自动微分引擎
        3. Adam优化器实现
        4. 实时训练循环
 */

 /*
 训练流程详解-单步训练：
    1. 随机生成UV → RNG
    2. 采样目标颜色 → targetTex[uv]
    3. 网络前向 → predictedRGB
    4. 损失计算 → loss = L(target, predicted)
    5. 梯度计算 → ∇loss (自动微分)
    6. 反向传播 → ∇weights (bwd_diff)
    7. 优化器更新 → weights = Adam(∇weights)
    8. 梯度清零 → grad = 0
 */
import slangpy;

__exported import NeuralModules;
__exported import Loss;
__exported import Optimizers;

#define ADAM_BETA1                               0.9f
#define ADAM_BETA2                               0.999f
#define ADAM_EPSILON                             1E-8f


struct RWNDBufferAccessor : optimizers::IBufferAccessor
{
    RWNDBuffer<float, 1> m_buffer;
    
    __init(RWNDBuffer<float, 1> buffer)
    {
        m_buffer = buffer;
    }

    float get(uint index)
    {
        return m_buffer[index];
    }

    void set(uint index, float value)
    {
        m_buffer[index] = value;
    }
};

struct RNG
{
    uint state;

    __init(uint state) { this.state = state; }

    [mutating]
    float next()
    {
        float r = (state >> 8) * 0x1p-24;
        state = state * 2739110765U + 2739110765U;
        return r;
    }
}

// An example of adding a custom activation to your network
// This implements the Sigmoid Linear Unit (SiLU)
/*
自定义激活函数:
SiLU (Sigmoid Linear Unit)：
    1. 公式：x * sigmoid(x) = x / (1 + e^(-x))
    2. 平滑、非单调，在深度学习中表现优异
    3. 优于ReLU的特性
*/
struct SiLUActivation<T : __BuiltinFloatingPointType, let K : int> : rtxns::mlp::IActivation<T, K>
{
    [Differentiable]
    CoopVec<T, K> eval(CoopVec<T, K> x)
    {
        return x * no_diff CoopVec<T, K>(T(1.)) / (no_diff CoopVec<T, K>(T(1.)) + exp(-x));
    }
}

// Utility function for nearest-neighbor sampling of texture
T SampleTexture<T : ITexelElement>(Texture2D<T> tex, float2 uv)
{
    float2 size;
    tex.GetDimensions(size[0], size[1]);
    uint2 xy = uint2(uv * size);
    return tex[xy];
}

// Take one step with the adam optimizer
/*
优化器实现的特征:
    1. 混合精度训练：FP32累积更新 + FP16计算梯度
    2. Adam优化器：自适应学习率（β1=0.9, β2=0.999, ε=1e-8）
    3. 梯度清零：更新后自动清零
*/
void OptimizerStep(
    RWNDBuffer<float, 1> moments1,      // Adam一阶矩
    RWNDBuffer<float, 1> moments2,      // Adam二阶矩
    RWNDBuffer<float, 1> paramF,        // FP32参数（主）
    RWNDBuffer<half, 1> paramH,         // FP16参数（计算）
    RWNDBuffer<half, 1> grad,           // 梯度
    uint idx,
    float learningRate,
    float gradScale,
    int iteration)
{
    var optimizer = optimizers::AdamAccessor<RWNDBufferAccessor>(RWNDBufferAccessor(moments1), RWNDBufferAccessor(moments2), learningRate, gradScale);
    // Parameters are converted to FP16 for computing gradients,
    // but we keep the FP32 originals around so we don't accumulate
    // rounding errors
    float parameter = paramF[idx];
    float gradient = (float)grad[idx];

    parameter = optimizer.step(parameter, idx, gradient, iteration);

    // Update the reference FP32 parameter, and convert the new value back to FP16
    paramF[idx] = parameter;
    paramH[idx] = (half)parameter;
    // Zero out gradients
    grad[idx] = 0.0h;
}

/*
训练流程(TrainTexture)：
    1. 随机采样UV坐标
    2. 采样目标纹理颜色
    3. 网络前向传播预测颜色
    4. 计算损失梯度
    5. 反向传播更新权重
*/
void TrainTexture<Model : rtxns::IModule<half, 2, 3>, Loss : rtxns::mlp::ILoss<float, 3>>(Model model, inout RNG rng, Texture2D<float4> targetTex, float lossScale)
{
    // Get a random uv coordinate for the input
    float2 inputUV = clamp(float2(rng.next(), rng.next()), 0.0, 1.0);

    // Sample the target texture at the generated UV
    float3 targetRGB = SampleTexture(targetTex, inputUV).rgb;

    // Evaluate the current output of the model
    float3 predictedRGB = EvalModel(model, inputUV);

    // Evaluate the loss gradient
    float3 lossGradient = rtxns::mlp::LossDerivFromVector<3, Loss>(
        targetRGB, predictedRGB, lossScale);

    // Backpropragate gradient through network parameters
    bwd_diff(EvalModel)(model, inputUV, lossGradient);
}

// Convenience functions for evaluating the model from vector inputs
// Converts to/from CoopVec internally
/*
自动微分系统:
    1. [Differentiable]属性标记可微函数
    2. bwd_diff自动生成反向传播代码
    3. no_diff标记非微分参数
*/
[Differentiable]
float3 EvalModel<Model: rtxns::IModule<half, 2, 3>>(Model model, no_diff float2 inputUV)
{
    var inputVec = rtxns::CoopVecFromVector<half>(inputUV);

    var result = model.forward(inputVec);

    return rtxns::VectorFromCoopVec(result);
}

// Computes the loss between the predicted RGB at a given UV coordinate and a reference texture
float3 EvalLoss<Loss : rtxns::mlp::ILoss<float, 3>>(float2 inputUV, float3 predictedRGB, Texture2D<float4> targetTex)
{
    float3 targetRGB = SampleTexture(targetTex, inputUV).rgb;
    return rtxns::mlp::LossValueFromVector<3, Loss>(
        targetRGB, predictedRGB, 1.0f);
}

// Computes the difference between the predicted RGB at a given UV coordinate and a reference texture
// for visualization
float3 TextureDifference(float2 inputUV, float3 predictedRGB, Texture2D<float4> targetTex, float scale)
{
    float3 targetRGB = SampleTexture(targetTex, inputUV).rgb;

    return (predictedRGB - targetRGB) * scale + 0.5f;
}

// Convenience function to convert from half to float params
float ConvertToFloat(half paramH)
{
    return (float)paramH;
}
