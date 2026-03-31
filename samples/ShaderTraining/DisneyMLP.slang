/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/*****************************************************************************
 *
 * 作用：
        使用MLP（多层感知机）替代传统BRDF计算的Slang着色器代码，
 *      展示了神经渲染技术在实际图形中的应用。
 * 核心思想:
 *      使用训练好的神经网络来近似替代之前分析的Disney BRDF函数，
 *      将复杂的物理计算转换为轻量级的推理操作。
 *
 ****************************************************************************/

/*
            与原始Disney BRDF的对比
特性	            原始Disney	                    MLP版本
计算复杂度	多个三角函数、除法、幂运算	        矩阵乘法 + 激活函数
精度	        物理精确	                        近似（取决于训练）
灵活性	    参数可调	                        固定权重
硬件利用	    标量/向量单元	                    张量核心
内存占用	    无	                            存储权重和偏置
*/

/*
潜在应用场景:
    1. 材质烘焙：将复杂材质预训练为神经网络
    2. 实时渲染：在移动端或VR设备上替代昂贵的光照计算
    3. 风格化渲染：通过定制训练数据实现非真实感效果
    4. 降噪后处理：结合光线追踪降噪

性能考量
优势：
    1. 利用张量核心并行计算
    2. FP16精度减少带宽
    3. 无分支预测失败（固定网络结构）
劣势：
    1. 权重存储开销（取决于网络大小）
    2. 训练数据依赖（需要覆盖所有输入空间）
    3. 可能丢失高频细节（频率编码部分补偿）
总结:
    这是一个工业级的神经渲染实现，展示了如何将传统图形学算法与现代AI硬件加速结合。
    通过将Disney BRDF蒸馏为优化的神经网络，在保持视觉质量的同时可能获得显著的性能提升，
    特别适合需要大量材质计算的场景。
*/

import MLP;
import CooperativeVectorFunctions;
import Activation;
import Utils;

/*
 * 网络结构：
 *      输入层：5 × 6 = 30 个神经元（经过频率编码）
 *      隐藏层：可配置（HIDDEN_LAYERS 层，每层 HIDDEN_NEURONS 个神经元）
 *      输出层：4个神经元
 */、 
// 5 inputs are passed into this function; NdotL, NdotV, NdotH, LdotH, roughness
#define INPUT_FEATURES 5    // 输入：NdotL, NdotV, NdotH, LdotH, roughness

// The output is float4
#define OUTPUT_NEURONS 4    // 输出：float4 (漫反射、镜面等)

// EncodeFrequency expands the input by 6 per input feature
#define FREQUENCY_EXPANSION 6

float4 DisneyMLP<let HIDDEN_LAYERS : int, let HIDDEN_NEURONS : int>(
    float NdotL, float NdotV, float NdotH, float LdotH, float roughness, ByteAddressBuffer mlpBuffer,
    uint weightOffsets[HIDDEN_LAYERS+1], uint biasOffsets[HIDDEN_LAYERS+1])
{
    // Calculate approximated core shader part using MLP
    float params[INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };

    // 将5个输入特征扩展为30维（每维6个频率分量）
    // 类似NeRF的位置编码，帮助网络学习高频细节
    // 使用half精度减少内存和计算开销
    var inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);

    // MLP推理引擎 - InferenceMLP
    // 关键特性：
    //      Cooperative Vector (CoopVec)：利用GPU张量核心进行加速
    //      训练最优布局：权重重排以提高内存访问效率
    //      FP16计算：平衡精度和性能
    var model = rtxns::mlp::InferenceMLP<half, 
        HIDDEN_LAYERS, 
        INPUT_FEATURES * FREQUENCY_EXPANSION, 
        HIDDEN_NEURONS, 
        OUTPUT_NEURONS, 
        CoopVecMatrixLayout::TrainingOptimal, 
        CoopVecComponentType::Float16>
        (mlpBuffer, weightOffsets, biasOffsets);

    // 激活函数:
    //  ReLU：隐藏层标准选择，计算简单
    //  Exponential：输出层使用指数激活，确保结果为正值（BRDF值应为非负）
    var outputParams = model.forward(inputParams, rtxns::mlp::ReLUAct<half, HIDDEN_NEURONS>(), rtxns::mlp::ExponentialAct<half, OUTPUT_NEURONS>());
    return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
}
