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
与训练版本的关键对比:
特性	    训练版本 (training_cs)	推理版本 (inference_cs)
输入	        随机UV坐标	            固定像素坐标
输出	        单个像素值 + 梯度	        完整图像
状态保持	    需要RNG状态	            无状态
变量存储	    缓存所有层输出	            复用单个变量
内存访问	    读写多个缓冲区         	只读权重，只写输出
随机性	    有（采样随机）	            完全确定
性能	        较慢（需要梯度）	        极快（只前向）
*/

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

import CooperativeVectorFunctions;
import Utils;
import LinearOps;

DECLARE_CBUFFER(NeuralConstants, gConst, 0, 0);
ByteAddressBuffer gMLPParams            :REGISTER_SRV(0, 0);        // 训练好的网络权重（只读）
Texture2D<float4> inputTexture          :REGISTER_SRV(1, 0);        // 原始输入图像（可能用于参考，但本代码未使用）
RWTexture2D<float4> outputTexture       :REGISTER_UAV(0, 0);        // 输出纹理（生成结果）

[shader("compute")]
[numthreads(THREADS_PER_GROUP_X, THREADS_PER_GROUP_Y, 1)] 
void inference_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Set the input ID as the uv coordinate and frequency encode it for the network
    // 输入坐标生成:
    //      确定性采样：使用线程ID计算UV坐标（不像训练时使用随机采样）
    //      归一化：将像素坐标转换为[0,1]范围的UV坐标
    //      频率编码：与训练版本相同，确保输入分布一致
    float2 inputUV = float2(dispatchThreadID.x / float(gConst.imageWidth), dispatchThreadID.y / float(gConst.imageHeight));
    CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams = rtxns::EncodeFrequency<half, 2>({inputUV.x, inputUV.y});

    // Load offsets
    // 权重偏移加载
    // 与训练版本完全相同的偏移量解包逻辑
    // 确保使用与训练一致的权重布局
    uint weightOffsets[NUM_TRANSITIONS] = rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets);
    uint biasOffsets[NUM_TRANSITIONS] = rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets);

    // 变量简化分析：
    //  不需要缓存所有层的输出（相比训练版本）：
    //      训练需要保留每层的激活值用于反向传播
    //      推理只需要当前层的值，可以复用变量
    //  显著减少寄存器使用和内存访问
    CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenParams;
    CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputParams;

    // Forward propagation through the neural network
    // Input to hidden layer, then apply activation function
    // 第一层前向传播:
    //      线性变换：hidden = W₁ * input + b₁
    //      激活函数：LeakyReLU（与训练一致）
    //      变量复用：直接覆盖hiddenParams，不需要新变量
    hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
        inputParams, gMLPParams, weightOffsets[0], biasOffsets[0],
        MATRIX_LAYOUT, TYPE_INTERPRETATION);
    hiddenParams = rtxns::leakyReLU(hiddenParams, RELU_LEAK);

    // Hidden layers to hidden layers, then apply activation function
    // 隐藏层循环:
    //      链式处理：每层的输出直接作为下一层的输入
    //      原地更新：不断覆盖hiddenParams（节省寄存器）
    //      [ForceUnroll]：强制展开循环优化性能
    //      激活函数：每层后都应用LeakyReLU
    [ForceUnroll]
    for (uint layer = 1; layer < NUM_HIDDEN_LAYERS; layer++)
    {
        hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
            hiddenParams, gMLPParams, weightOffsets[layer], 
            biasOffsets[layer], 
            MATRIX_LAYOUT, TYPE_INTERPRETATION);
        hiddenParams = rtxns::leakyReLU(hiddenParams, RELU_LEAK);
    }

    // Hidden layer to output layer, then apply final activation function
    // 输出层:
    //      线性变换：output = W_out * hidden + b_out
    //      Sigmoid激活：将输出压缩到[0,1]范围（适合颜色值）
    //      输出维度：通常为3（RGB），但代码使用OUTPUT_NEURONS配置
    outputParams = rtxns::LinearOp<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
        hiddenParams, gMLPParams, weightOffsets[NUM_HIDDEN_LAYERS], biasOffsets[NUM_HIDDEN_LAYERS],
        MATRIX_LAYOUT, TYPE_INTERPRETATION);
    outputParams = rtxns::sigmoid(outputParams);

    // Take the output from the neural network as the output color
    // 结果写入:
    //      颜色组装：取前三个神经元作为RGB，Alpha设为1
    //      直接写入：将结果写入对应像素位置
    //      并行完成：所有像素同时生成
    float4 color = {outputParams[0], outputParams[1], outputParams[2], 1.f};
    outputTexture[dispatchThreadID.xy] = color;
}