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
  这是一个Adam优化器的GPU实现，用于更新神经网络的权重参数：
  Adam 的名字来源于 Adaptive Moment Estimation（自适应矩估计）。它是一种在深度学习领域非常流行的优化算法，可以理解为 
    将两种其他优化方法的优点结合在了一起：
        动量法：它像物理学中的动量，不仅考虑当前的梯度，还会考虑过去梯度的方向，从而让更新过程更平稳，减少震荡，并能更快地冲向最优解。
        RMSProp：它为每个参数自适应不同的学习率，对于经常更新的参数（特征），学习率会较小；对于不常更新的参数（特征），学习率会较大。这使得模型在处理稀疏数据时表现很好。
  简单来说，Adam 为每个参数都维护了两个“记忆”：
    一阶矩（First Moment）：记录梯度的平均值（相当于“动量”），决定更新的方向。
    二阶矩（Second Moment）：记录梯度平方的平均值（相当于 RMSProp 中的自适应学习率部分），决定每个参数自身的学习率有多大。
  通过这种方式，Adam 能够自动调整每个参数的学习率，从而使模型在训练过程中收敛速度更快，效果也更稳定。
 */

 /*
 完整工作流程：
┌─────────────────┐
│  前向传播着色器   │  ← 计算预测值
│  (training_cs)  │  ← 累积梯度到 gMLPParamsGradients
└────────┬────────┘
         ↓
┌─────────────────┐
│  梯度同步屏障     │ ← 所有前向传播线程完成
└────────┬────────┘
         ↓
┌─────────────────┐
│  Adam优化着色器  │  ← 每个参数一个线程
│  (adam_cs)      │  ← 读取梯度，更新权重
└────────┬────────┘
         ↓
┌─────────────────┐
│  下一轮训练      │  ← 使用更新后的权重
└─────────────────┘
 */

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

import Optimizers;

// 双精度存储策略：
//      gMLPParams：半精度权重（节省显存，加速计算）
//      gMLPParamsf：单精度权重（保持数值稳定性）
//      两者保持同步，但优化器使用单精度计算
// Adam状态：
//      gMoments1：一阶矩估计（动量项）
//      gMoments2：二阶矩估计（RMSprop项）
//      使用单精度保证优化稳定性
// 梯度缓冲区：
//      gMLPParamsGradients：半精度梯度（由前向传播着色器累积）
DECLARE_CBUFFER(NeuralConstants, gConst, 0, 0);
RWBuffer<half> gMLPParams             :REGISTER_UAV(0, 0);      // 半精度权重参数
RWBuffer<float> gMLPParamsf           :REGISTER_UAV(1, 0);      // 单精度权重参数
RWBuffer<half> gMLPParamsGradients    :REGISTER_UAV(2, 0);      // 梯度（半精度）
RWBuffer<float> gMoments1             :REGISTER_UAV(3, 0);      // 一阶矩（动量）
RWBuffer<float> gMoments2             :REGISTER_UAV(4, 0);      // 二阶矩（自适应学习率）

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

    // On the first step load from the FP16 buffer
    // 权重加载策略:
    // 加载策略分析：
    // 第一步特殊处理（currentStep == 1）：
    //      从半精度缓冲区加载初始权重
    //      因为单精度缓冲区可能还未初始化
    // 后续步骤：
    //      从单精度缓冲区加载
    //      使用高精度值进行优化计算
    float weightBias = gConst.currentStep == 1 ? (float)gMLPParams[i] : gMLPParamsf[i];
   
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
    gMLPParamsf[i] = adjustedWeightBias;
    gMLPParams[i] = (half)adjustedWeightBias;
}