/*
 * Copyright (c) 2015 - 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

import CooperativeVectorDerivatives;
import CooperativeVectorFunctions;
import Utils;
import LinearOps;

DECLARE_CBUFFER(NeuralConstants, gConst, 0, 0);
ByteAddressBuffer gMLPParams                    :REGISTER_SRV(0, 0);            // 神经网络权重参数
Texture2D<float4> inputTexture                  :REGISTER_SRV(1, 0);            // 输入图像（训练目标）
RWByteAddressBuffer gMLPParamsGradients         :REGISTER_UAV(0, 0);            // 梯度累积缓冲区
RWBuffer<uint> gRandState                       :REGISTER_UAV(1, 0);            // 随机数生成器状态
RWTexture2D<float4> outputTexture               :REGISTER_UAV(2, 0);            // 网络预测的输出图像
RWTexture2D<float4> lossTexture                 :REGISTER_UAV(3, 0);            // 损失可视化
RWBuffer<float> gLossBuffer                     :REGISTER_UAV(4, 0);            // 损失值存储

struct RNG  // 随机数生成器结构体
{
    uint state;

    __init(uint state) { this.state = state; }

    [mutating]      // 用于标记一个函数或方法可以修改它所属结构体（struct）的成员变量
    float next()    // 生成下一个【0,1】的随机数
    {
        float r = (state >> 8) * 0x1p-24;               // 生成[0,1)范围内的浮点数
        state = state * 2739110765U + 2739110765U;      // 更新状态
        return r;
    }
}

/*
整体运行机制总结:
这个着色器实现了一个完整的单样本训练循环：
    1. 并行性：每个GPU线程独立处理一个随机样本
    2. 内存效率：使用协处理向量优化内存访问
    3. 数值稳定性：通过损失缩放防止梯度消失
    4. 状态持久化：每个线程维护自己的RNG状态
    5. 梯度累积：多个线程的梯度累加到同一缓冲区
训练过程通过多次调度这个着色器完成，每次处理一批随机样本，不断更新网络参数。
*/
[shader("compute")]
[numthreads(THREADS_PER_GROUP_X, THREADS_PER_GROUP_Y, 1)] 
void training_cs(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 batchSize = uint2(gConst.batchSizeX, gConst.batchSizeY);

    uint dispatchThreadIdxy = dispatchThreadID.y * batchSize.x + dispatchThreadID.x;
    RNG rng = RNG(gRandState[dispatchThreadIdxy]);

    // Get a random uv coordinate for the input and frequency encode it for improved convergance
    // 使用RNG（随机数生成器）生成随机的[0, 1]的UV坐标
    // 将2D输入编码到高维空间
    float2 inputUV = clamp(float2(rng.next(), rng.next()), 0.0, 1.0);
    CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams = rtxns::EncodeFrequency<half, 2>({inputUV.x, inputUV.y});

    // Load offsets
    // 加载 权重和bias的偏移值
    // NUM_TRANSITIONS: 网络层之间的转换次数（=层数-1）
    // 这些偏移量用于在gMLPParams缓冲区中定位每层的权重和偏置
    uint weightOffsets[NUM_TRANSITIONS] = rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.weightOffsets);
    uint biasOffsets[NUM_TRANSITIONS] = rtxns::UnpackArray<NUM_TRANSITIONS_ALIGN4, NUM_TRANSITIONS>(gConst.biasOffsets);

    // Create variables to cache the results from each stage
    CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenParams[NUM_HIDDEN_LAYERS];     // 隐藏层的线性变换结果（激活前）
    CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenActivated[NUM_HIDDEN_LAYERS];  // 隐藏层的激活函数结果
    CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputParams;                        // 输出层的线性变换结果
    CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputActivated;                     // 输出层的激活函数结果（最终预测）

    // Forward propagation through the neural network
    // Input to hidden layer, then apply activation function
    // 前向传播 - 输入到第一个隐藏层
    // LinearOp - 线性变换：output = weight * input + bias
    //      模板参数：输出维度、输入维度
    //      参数：输入向量、权重缓冲区、权重偏移、偏置偏移
    //      MATRIX_LAYOUT：矩阵内存布局（行主序 / 列主序）
    //      TYPE_INTERPRETATION：数据类型解释方式
    // leakyReLU：带泄露的ReLU激活函数
    //      RELU_LEAK：负半轴的斜率（通常为0 .01）
    //      公式：f(x) = max(x, leak * x)
    hiddenParams[0] = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
        inputParams, gMLPParams, weightOffsets[0], biasOffsets[0], MATRIX_LAYOUT, TYPE_INTERPRETATION);
    hiddenActivated[0] = rtxns::leakyReLU(hiddenParams[0], RELU_LEAK);

    // Hidden layers to hidden layers, then apply activation function
    // 前向传播 - 隐藏层循环
    // [ForceUnroll]：强制编译器展开循环（优化性能）
    // 层间连接：每个隐藏层的输入是前一层的激活输出
    // 维度不变：隐藏层之间维度相同（HIDDEN_NEURONS）
    // 权重矩阵：每层有自己的权重矩阵和偏置向量
    [ForceUnroll]
    for (uint layer = 1; layer < NUM_HIDDEN_LAYERS; layer++)
    {
        hiddenParams[layer] = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
            hiddenActivated[layer - 1], gMLPParams, weightOffsets[layer], biasOffsets[layer], 
            MATRIX_LAYOUT, TYPE_INTERPRETATION);
        hiddenActivated[layer] = rtxns::leakyReLU(hiddenParams[layer], RELU_LEAK);
    }

    // Hidden layer to output layer, then apply final activation function
    // 前向传播 - 输出层
    // 输出层线性变换：
    //      输入：最后一个隐藏层的激活值
    //      输出维度：OUTPUT_NEURONS（通常为3，对应RGB）
    // Sigmoid激活：
    //      将输出压缩到[0,1]范围
    //      公式：f(x) = 1/(1 + e^(-x))
    //      适合作为颜色输出    
    outputParams = rtxns::LinearOp<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
        hiddenActivated[NUM_HIDDEN_LAYERS - 1], gMLPParams, weightOffsets[NUM_HIDDEN_LAYERS],
        biasOffsets[NUM_HIDDEN_LAYERS], MATRIX_LAYOUT, TYPE_INTERPRETATION);
    outputActivated = rtxns::sigmoid(outputParams);

    // Take the output from the neural network as the output color
    // RGB提取：取前三个神经元的值作为RGB颜色
    float3 predictedRGB = {outputActivated[0], outputActivated[1], outputActivated[2]};

    // Now transform the input UVs according to the NetworkModel enum.
    // This can easily be extended to try many different transforms.
    // 变换类型：
    //      Flip：交换UV坐标（旋转/镜像效果）
    //      Zoom：放大到中心区域（0.25-0.75范围）
    //      None：直接映射
    // 像素坐标计算：
    //      将浮点UV乘以图像尺寸
    //      转换为整数像素坐标
    uint2 actualUV;
    if (gConst.networkTransform == NetworkTransform.Flip)
    {
        float2 flipUV = inputUV.yx;
        actualUV = uint2(flipUV.xy * float2(gConst.imageHeight, gConst.imageWidth));
    }
    else if (gConst.networkTransform == NetworkTransform.Zoom)
    {
        float2 zoomUV = inputUV * 0.5 + 0.25;
        actualUV = uint2(zoomUV.xy * float2(gConst.imageWidth, gConst.imageHeight));
    }
    else
    {
        actualUV = uint2(inputUV.xy * float2(gConst.imageWidth, gConst.imageHeight));
    }

    // Load the texture according to the transformed input UVs. This will
    // provide the RGB that the model is trying to train towards.
    // 从输入纹理中读取实际像素值, 作为训练的目标值（ground truth）
    // 网络的目标是让predictedRGB接近actualRGB
    float3 actualRGB = inputTexture[actualUV].rgb;

    // Output the loss, scaled to greyscale for output
    // 损失可视化：
    //      计算预测值和实际值的绝对差, 乘以10倍缩放以便观察
    //      存储为RGBA（RGB差值，Alpha=1）,用于调试和监控训练过程
    uint2 lossUV = uint2(inputUV.xy * float2(gConst.imageWidth, gConst.imageHeight));
    const float lossScaleFactor = 10.0f; // scale it up for better vis
    lossTexture[lossUV] = abs(float4((predictedRGB - actualRGB), 1));  

    // Compute the L2 loss gradient
    // L2Loss = (a-b)^2
    // L2Loss Derivative = 2(a-b)
    // L2损失梯度：
    //      损失函数：L = (p - t)^2
    //      梯度：dL/dp = 2(p - t)
    float3 lossGradient = 2.0 * (predictedRGB - actualRGB);

    // Scale by batch size
    // 批量归一化：
    //      除以批次大小（样本数量）
    //      使梯度大小与批次大小无关
    lossGradient /= (batchSize.x * batchSize.y);

    // Apply the LOSS_SCALE factor to retain precision. Remove it in the optimizer pass before use.
    // 损失缩放（训练技巧）：
    //      LOSS_SCALE通常是一个大数（如1024或4096）
    //      防止梯度在反向传播中消失
    //      优化器阶段会移除这个缩放
    lossGradient *= LOSS_SCALE;

    // 转换为协处理向量:
    //      将float3类型的梯度转换为CoopVec类型
    //      VECTOR_FORMAT转换确保数据类型匹配
    //      为反向传播做准备
    CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> lossGradientCV = CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS>(VECTOR_FORMAT(lossGradient[0]), VECTOR_FORMAT(lossGradient[1]), VECTOR_FORMAT(lossGradient[2]));

    // Back-propagation pass, generate the gradients and accumulate the results into memory to be applied in the optimisation pass.
    // 反向传播变量声明:
    //      outputGradient：输出层的误差梯度
    //      hiddenGradient：隐藏层的误差梯度（会逐层复用）
    //      这些梯度将用于计算权重更新
    CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputGradient;
    CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenGradient;

    // Output layer (loss gradient) to final hidden layer
    // 输出层反向传播:
    // sigmoid导数：
    //      公式：sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    //      计算输出层的误差梯度
    // LinearOp_Backward（关键）：
    //      1. 计算前一层的梯度（返回）
    //      2. 累加当前层的权重梯度到gMLPParamsGradients
    outputGradient = rtxns::sigmoid_Derivative(outputParams, lossGradientCV);
    hiddenGradient = rtxns::LinearOp_Backward<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
       hiddenActivated[NUM_HIDDEN_LAYERS - 1], outputGradient, gMLPParams, gMLPParamsGradients, 
       weightOffsets[NUM_HIDDEN_LAYERS], biasOffsets[NUM_HIDDEN_LAYERS], MATRIX_LAYOUT, TYPE_INTERPRETATION);

    // Hidden layer to hidden layer
    // 隐藏层反向传播循环
    // 链式法则：梯度从输出层向输入层传播
    // leakyReLU导数：
    //      公式：f'(x) = 1 if x > 0 else leak
    //      乘以上游梯度
    // 循环方向：从最后一层向第一层反向传播
    // 梯度累加：每层都累加自己的权重梯度
    for(int layer = NUM_HIDDEN_LAYERS - 1; layer >= 1; layer--)
    {
        hiddenGradient = rtxns::leakyReLU_Derivative(hiddenParams[layer], RELU_LEAK, hiddenGradient);
        hiddenGradient = rtxns::LinearOp_Backward<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>
            (hiddenActivated[layer - 1], hiddenGradient, gMLPParams, gMLPParamsGradients, 
            weightOffsets[layer], biasOffsets[layer], MATRIX_LAYOUT, TYPE_INTERPRETATION);
    }

    // First hidden layer to input layer
    // 输入层反向传播:
    // 最后一层反向传播：
    //      1. 计算第一个隐藏层的激活函数导数
    //      2. 计算输入层的权重梯度（不需要返回梯度到输入）
    // 至此，所有层的梯度都已累加到gMLPParamsGradients
    hiddenGradient = rtxns::leakyReLU_Derivative(hiddenParams[0], RELU_LEAK, hiddenGradient);
    rtxns::LinearOp_Backward<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
        inputParams, hiddenGradient, gMLPParams, gMLPParamsGradients, weightOffsets[0], 
        biasOffsets[0], MATRIX_LAYOUT, TYPE_INTERPRETATION);

    // Store the random state to continue iterating next time.
    // 状态保存和损失记录:
    // RNG状态保存：
    //      保存更新后的随机数生成器状态
    //      确保下一轮训练可以继续生成随机数
    gRandState[dispatchThreadIdxy] = rng.state;

    // Store L2 loss
    // 损失记录：
    //      计算L2损失：diff·diff（点积）
    //      存储到损失缓冲区
    //      可用于监控训练进度和早停
    float3 diff = predictedRGB - actualRGB;
    gLossBuffer[dispatchThreadIdxy] = dot(diff, diff);
}