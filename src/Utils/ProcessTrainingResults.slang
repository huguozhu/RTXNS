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
┌─────────────────┐
│  训练着色器      │  ← 每个线程计算一个样本的损失
│  (training_cs)  │  ← 写入损失到 lossBuffer
└────────┬────────┘
        ↓
┌─────────────────┐
│  第一阶段        │  ← 多个线程组并行执行
│  损失归约        │  ← 每个组归约部分样本
│  (多个线程组)    │  ← 结果累积到 accumulationBuffer
└────────┬────────┘
        ↓
┌─────────────────┐
│  第二阶段        │  ← 单线程执行
│  平均计算        │  ← 计算最终平均损失
│  (average_cs)   │  ← 写入 outputBuffer
└────────┬────────┘
        ↓
┌─────────────────┐
│  CPU读取结果     │  ← 获取训练统计信息
└─────────────────┘

                    关键设计特点
特性	            作用	                    优势
两阶段处理	    分离归约和平均计算	        灵活，可扩展
并行归约	        O(log N)快速求和	        比串行快N/log N倍
共享内存	        线程组内高速通信	        比全局内存快~100倍
原子累积	        安全合并结果	            避免竞争条件
结构化输出	    TrainingResults	        CPU友好接口
*/

#include "TrainingResults.h"
#include <donut/shaders/binding_helpers.hlsli>

import LossAccumulation;

#if defined(DXIL) && defined(NVAPI_INIT)
// Reserved a space and slot for NVAPI extensions 
#define NV_SHADER_EXTN_REGISTER_SPACE space0
#define NV_SHADER_EXTN_SLOT u99
#endif

DECLARE_CBUFFER(LossConstants, gConst, 0, 0);

Buffer<float> lossBuffer                            :REGISTER_SRV(0,0);     // 存储每个训练线程计算的损失值
RWByteAddressBuffer accumulationBuffer              :REGISTER_UAV(0,0);     // 字节寻址缓冲区，累积部分和
RWStructuredBuffer<TrainingResults> outputBuffer    :REGISTER_UAV(1,0);     // 结构化缓冲区，存储最终训练结果


static const uint THREADS_PER_GROUP = RESULTS_THREADS_PER_GROUP;
static const uint MAX_COMPONENTS = 1;

groupshared float gLossShared[THREADS_PER_GROUP][ MAX_COMPONENTS];  // 用于线程组内的损失值归约

[shader("compute")]
[numthreads(THREADS_PER_GROUP, 1, 1)]
// 损失归约着色器
void lossReduction_cs(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    const uint index = dispatchThreadID.x;
    
    const LossConfig lossConfig = MakeLossConfig(MAX_COMPONENTS, gConst.bufferOffset);

    float loss = 0.f;

    if (index < gConst.batchSize)
    {
        loss = lossBuffer[index];
    }
    gLossShared[groupThreadID.x][0] = loss;
    GroupMemoryBarrierWithGroupSync();

    // 归约算法分析（并行归约）：
    //      初始状态：   [L0][L1][L2][L3][L4][L5][L6][L7]
    //      步长=4：    [L0+L4][L1+L5][L2+L6][L3+L7][L4][L5][L6][L7]
    //      步长=2：    [L0+L4+L2+L6][L1+L5+L3+L7][L2+L6][L3+L7][L4][L5][L6][L7]
    //      步长=1：    [总和][L1+L5+L3+L7][L2+L6][L3+L7][L4][L5][L6][L7]
    // 算法特点：
    //      时间复杂度：O(log N)（N=THREADS_PER_GROUP）
    //      空间复杂度：O(N)（共享内存）
    //      线程利用率：逐步降低（活动线程数：N/2 → N/4 → ...）    
    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1)
    {
        if (groupThreadID.x < stride)
        {
            AddInPlace<MAX_COMPONENTS>(gLossShared[groupThreadID.x], gLossShared[groupThreadID.x + stride], lossConfig.GetComponentCount());
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // 累积分析：
    //      只有组内第一个线程执行累积
    //      将归约后的部分和添加到全局累积缓冲区
    //      AccumulateComponents处理原子操作（避免竞争条件）
    if (groupThreadID.x == 0)
    {
       AccumulateComponents<MAX_COMPONENTS>(accumulationBuffer, gLossShared[0], lossConfig);
    }
}


// 平均计算着色器
//      单线程执行：[numthreads(1,1,1)]
//      加载总和：从累积缓冲区读取所有损失的和
//      计算平均值：总和 / 样本总数
//      结果组装：填充TrainingResults结构体
//      调试信息：debug0/debug1用于验证和调试
[shader("compute")]
[numthreads(1, 1, 1)]
void average_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float lossSum =  asfloat(accumulationBuffer.Load(0u));
    float epochLoss = lossSum / gConst.epochSampleCount;
    
    TrainingResults result = {}; 
    result.l2Loss = epochLoss; 
    result.epoch = gConst.epoch;
    result.debug0 = lossSum;
    result.debug1 = gConst.epochSampleCount;
    
    outputBuffer[0] = result; 
}

