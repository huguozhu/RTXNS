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
用于学习迪士尼BSDF（双向散射分布函数）的神经网络训练内核:
┌─────────────────────┐
│  随机生成方向/粗糙度  │ ← 每个线程独立
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  计算Disney BSDF     │ ← 参考实现（真值）
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  频率编码输入参数     │ ← 5维 → 高维
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  MLP前向传播         │ ← 预测BSDF
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  计算相对L2损失       │ ← 比较预测和真实
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  反向传播            │ ← 累积梯度
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  存储损失值          │ ← 监控训练
└─────────────────────┘
*/

/*
关键技术创新:
    特性	            作用	                    实现方式
    频率编码	        提高高频细节学习能力	    EncodeFrequency
    相对L2损失	    关注相对误差	            L2Relative
    指数激活	        保证正输出	            ExponentialAct
    自动微分	        简化反向传播	            CooperativeVectorAutoDiff
    PCG32随机	    高质量采样	            PCG32生成器


应用场景:用于学习迪士尼BSDF的近似表示，包括：
    实时渲染：用轻量级MLP替代复杂BSDF计算
    材质压缩：用网络参数存储材质信息
    可微渲染：BSDF可微分便于优化
    重要性采样：快速评估BSDF值
*/ 
#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

import CooperativeVectorAutoDiff;   // 协处理向量的自动微分支持（核心优化）
import CooperativeVectorFunctions;  
import Utils;
import Activation;
import MLP;                         // 封装好的多层感知机训练接口
import Loss;                        // 预定义的损失函数
import PCG32;                       // 高质量的随机数生成器
import Disney;                      // 迪士尼BSDF的参考实现（作为训练目标）

DECLARE_CBUFFER(TrainingConstantBufferEntry, gConst, 0, 0);
ByteAddressBuffer gMLPParams               : REGISTER_SRV(0, 0);        // 网络权重
RWByteAddressBuffer gMLPParamsGradients    : REGISTER_UAV(0, 0);        // 梯度累积
RWBuffer<float>     gLossBuffer            : REGISTER_UAV(1, 0);        // 损失值存储

[shader("compute")]
[numthreads(THREADS_PER_GROUP_TRAIN, 1, 1)] 
void main_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //----------- Randomly generate input parameters
    // 生成随机数
    uint idx = dispatchThreadID.x;
    PCG32 rng = PCG32(gConst.seed, idx);

    // Using tangent coordinate system. N = (0,0,1)
    // L is arbitrary, but (N,L) >= 0 => L.z > 0, so generate random L in XZ plane's first quadrant
    // 光照方向L：
    //      限制在XZ平面（y=0）
    //      与法线夹角在[0, π/2]之间（确保N·L ≥ 0）
    //      使用sincos高效计算
    float3 L;
    L.y = 0.f;
    sincos(rng.nextFloat()*PI/2, L.z, L.x);

    // V is random direction, but (N,V) >= 0 => V.z > 0
    // 视角方向V：
    //      方位角：[-π, π]范围（完整圆）
    //      仰角：[0, π/2]范围（上半球）
    //      使用球坐标转换
    float sa, ca; // Azimuth [-PI/2,PI/2]
    sincos(-PI + 2 * PI * rng.nextFloat(), sa, ca);
    float se, ce; // Elevation [0,PI/2]
    sincos(PI/2 * rng.nextFloat(), se, ce);
    float3 V = float3(ce*ca, ce*sa, se);

    // 派生向量计算:
    //      半向量H：光照和视角的中间方向
    // 点积值：
    //      NdotL：光照与法线夹角余弦
    //      NdotV：视角与法线夹角余弦
    //      NdotH：半向量与法线夹角余弦
    //      LdotH：光照与半向量夹角余弦
    // 粗糙度：在[0.3, 1.0]范围（避免太光滑）
    float NdotL = L.z;
    float NdotV = V.z;

    float3 H = normalize(L+V);
    float NdotH = H.z;
    float LdotH = dot(L,H);

    float roughness = rng.nextFloat()*0.7f+0.3f;

    //----------- Calculate core shader part DIRECTLY
    // 目标值分析：
    //      Disney BSDF：工业标准的材质模型
    //      输入：5个几何参数
    //      输出：4分量BSDF值（可能是颜色或反射率）
    //      这是网络要学习的"真实值"
    float4 actualDisney = Disney(NdotL, NdotV, NdotH, LdotH, roughness);

    //----------- Training step
    // 输入分析：
    //      原始特征：5个几何/材质参数
    //      频率编码：将低维输入映射到高维空间
    //      INPUT_FEATURES > 5（编码后维度）
    float params[INPUT_FEATURES] = {NdotL, NdotV, NdotH, LdotH, roughness};
    var inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);

    // 模型初始化分析：
    //  模板参数：
    //      half：数据类型
    //      网络架构参数（层数、神经元数）
    //      矩阵布局优化
    //      组件类型
    //  构造函数参数：
    //      权重缓冲区
    //      梯度缓冲区
    //      权重/偏置偏移量
    var model = rtxns::mlp::TrainingMLP<half, 
        NUM_HIDDEN_LAYERS, INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, 
        CoopVecMatrixLayout::TrainingOptimal, CoopVecComponentType::Float16>(
        gMLPParams, 
        gMLPParamsGradients, 
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets), 
        rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets));

    // 激活函数分析：
    //  隐藏层：ReLU（线性整流单元）:
    //      公式：f(x) = max(0, x)
    //      稀疏激活，缓解梯度消失
    //  输出层：Exponential（指数激活）:
    //      公式：f(x) = exp(x)
    //      保证输出为正数（BSDF值应为正）
    //      适合物理量预测
    var hiddenActivation = rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>();
    var finalActivation = rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>();

    // 前向传播:
    //      输入经过MLP得到预测的BSDF值
    //      输出维度为4（对应BSDF的4个分量）
    //      使用指数激活确保值为正
    var outputParams = model.forward(inputParams, hiddenActivation, finalActivation);

    float4 predictedDisney = { outputParams[0], outputParams[1], outputParams[2], outputParams[3] };

    // 损失梯度计算:
    //  L2Relative损失：相对L2损失:
    //      公式：L = ||(pred - actual) / (actual + ε)||²
    //      关注相对误差而非绝对误差
    //      适合物理量（BSDF值可能范围很大）
    //  缩放因子：
    //      LOSS_SCALE：防止梯度消失
    //      batchSize * 4：批量归一化
    //      COMPONENT_WEIGHTS：各分量权重（可能不同）
    float4 lossGradient = rtxns::mlp::LossDerivFromVector<4, rtxns::mlp::L2Relative<float,4>>(
        actualDisney, predictedDisney, float4(LOSS_SCALE / (gConst.batchSize * 4)) * COMPONENT_WEIGHTS);

    // 反向传播:
    //      传入输入参数（用于梯度计算）
    //      传入激活函数（用于导数计算）
    //      传入损失梯度（上游梯度）
    //      自动累积权重梯度到gMLPParamsGradients
    model.backward(inputParams, hiddenActivation, finalActivation, rtxns::HCoopVec<OUTPUT_NEURONS>(lossGradient[0], lossGradient[1], lossGradient[2], lossGradient[3]));

    // Store L2 relative loss
    // 损失计算-相对L2误差:
    //      预测误差的L2范数 / 目标值的L2范数
    //      归一化的误差度量
    //      添加epsilon避免除零
    float4 diff = predictedDisney - actualDisney;
    float l2Error  = sqrt(dot(diff, diff));
    float l2Target = sqrt(dot(actualDisney, actualDisney));
    float epsilon = 1e-6f;

    gLossBuffer[idx] = l2Error / (l2Target + epsilon);
}