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
相对于SimpleTraining_Optimizer.slang，添加了对于梯度的判断：
    if (isfinite(gradient))
为什么需要这个检查？
    1. 在神经网络训练中，梯度可能变成非有限值：

    Inf（无穷大）：梯度爆炸
        -Inf（负无穷大）
    2. 数值不稳定的常见原因
        问题类型	        产生原因	                后果
        梯度爆炸	        链式法则累积过大值	        权重变成Inf
        除以零	        分母接近0	                产生NaN
        激活函数饱和	    sigmoid/tanh导数接近0	    梯度消失（接近0，不是NaN）
        损失函数异常	    log(0)等运算	            NaN
    3. 实际应用中的重要性
        这种数值稳定性检查在以下场景特别重要：
            长时间训练：防止一次数值溢出破坏整个训练
            混合精度训练：float16更容易溢出/下溢
            复杂损失函数：自定义损失可能产生NaN
            非标准化输入：输入数据范围异常
    4. 这个增强版Adam优化器的关键特性：
        特性	            作用	                    实现方式
        数值稳定性检查	    防止NaN/Inf破坏训练	    isfinite()检查
        安全清零	        无论梯度是否有效都重置	    清零在检查之前
        静默跳过	        无效梯度时不更新	        空if分支
        保持状态	        矩估计状态不变	            不调用optimizer.step()
    5. 可能的改进建议
        if (isfinite(gradient))
        {
            // 正常更新
        }
        else if (isnan(gradient))
        {
            // 可以记录NaN计数，用于调试
            // 或者减小学习率
        }
        else if (isinf(gradient))
        {
            // 可以梯度裁剪
            gradient = copysign(MAX_GRADIENT, gradient);
            // 然后更新
        }
*/

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

import Optimizers;

DECLARE_CBUFFER(TrainingConstantBufferEntry, gConst, 0, 0);
RWBuffer<half> gMLPParams             : REGISTER_UAV(0, 0);
RWBuffer<float> gMLPParams32          : REGISTER_UAV(1, 0);
RWBuffer<half> gMLPParamsGradients    : REGISTER_UAV(2, 0);
RWBuffer<float> gMoments1             : REGISTER_UAV(3, 0);
RWBuffer<float> gMoments2             : REGISTER_UAV(4, 0);

[numthreads(THREADS_PER_GROUP_OPTIMIZE, 1, 1)]
void adam_cs(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    // 调度分析：
    //  1. 每个线程处理一个权重参数
    //  2. 一维线程组织（X维度表示参数索引）
    //  3. THREADS_PER_GROUP_OPTIMIZE：每个线程组的线程数（如256）
    //  4. 边界检查确保不越界
    uint i = dispatchThreadID.x;
    if (i >= gConst.maxParamSize)
        return;

    // 梯度处理分析：
    // 类型提升：将半精度梯度转换为单精度（避免精度损失）
    // 清零操作：使用后立即清零梯度缓冲区
    //      这是多线程安全的关键
    //      确保下一轮迭代的梯度从零开始累积
    //      原子操作不需要，因为每个线程写不同的位置
    float gradient = (float)gMLPParamsGradients[i];
    gMLPParamsGradients[i] = half(0.0);

    // 为什么需要这个检查？
    // 在神经网络训练中，梯度可能变成非有限值：
    //      NaN（Not a Number）：除以零、0*∞等运算
    //      Inf（无穷大）：梯度爆炸
    //      -Inf（负无穷大）
    if (isfinite(gradient))
    {
        // On the first step load from the FP16 buffer
        // 权重加载策略:
        // 加载策略分析：
        // 第一步特殊处理（currentStep == 1）：
        //      从半精度缓冲区加载初始权重
        //      因为单精度缓冲区可能还未初始化
        // 后续步骤：
        //      从单精度缓冲区加载
        //      使用高精度值进行优化计算
        float weightBias = gConst.currentStep == 1 ? (float)gMLPParams[i] : gMLPParams32[i];

        // Adam优化器初始化:
        //      创建Adam优化器实例（每个线程独立）
        //      传入共享的矩估计缓冲区（通过引用）
        //      learningRate：学习率（来自常量缓冲区）
        //      LOSS_SCALE：损失缩放因子（与前向传播匹配）
        optimizers::Adam optimizer = optimizers::Adam(gMoments1, gMoments2, gConst.learningRate, LOSS_SCALE);

        /*
        # Adam优化器的数学原理(Python)
        def step(weight, index, gradient, step):
            # 更新一阶矩（动量）
            moments1[index] = beta1 * moments1[index] + (1 - beta1) * gradient

            # 更新二阶矩（自适应学习率）
            moments2[index] = beta2 * moments2[index] + (1 - beta2) * gradient²

            # 偏差修正
            m_hat = moments1[index] / (1 - beta1^step)
            v_hat = moments2[index] / (1 - beta2^step)

            # 参数更新（移除损失缩放）
            update = learning_rate * m_hat / (sqrt(v_hat) + epsilon)
            return weight - update / LOSS_SCALE
        */
        float adjustedWeightBias = optimizer.step(weightBias, i, gradient, gConst.currentStep);

        // 权重存储:
        gMLPParams32[i] = adjustedWeightBias;
        gMLPParams[i] = (half) adjustedWeightBias;
    }
}