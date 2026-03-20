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
实时渲染管线中集成神经网络BSDF的完整着色器，包含顶点着色器、像素着色器和MLP推理函数

与传统渲染对比:
方面	        传统渲染	            神经渲染（本代码）
BSDF计算	    复杂解析公式	        MLP推理
性能	        取决于公式复杂度	    固定时间（网络推理）
精度	        精确物理模型	        近似（但足够好）
可调性	    修改公式参数	        重新训练网络
内存	        着色器代码	        网络权重

性能优化特点:
优化技术	                应用位置	            效果
半精度计算	            神经网络推理	        减半显存，加快速度
InferencingOptimal布局	矩阵乘法	            优化推理内存访问
ReLU激活	                隐藏层	            计算简单，无指数运算
频率编码	                输入层	            提高收敛性和精度
常数缓冲区	            统一参数	            减少带宽占用
*/

import CooperativeVectorFunctions;
import Utils;
import LinearOps;

#include "NetworkConfig.h"
#include <donut/shaders/binding_helpers.hlsli>

DECLARE_CBUFFER(NeuralConstants, gConst, 0, 0);
ByteAddressBuffer gMLPParams            :REGISTER_SRV(0, 0);

struct VertexIn
{
    float3 pos : POSITION; 
    float3 norm : NORMAL; 
};

struct VertexOut
{
    float4 pos : SV_Position;
    float3 norm : NORMAL;
    float3 view : VIEW;
}

[shader("vertex")]
void main_vs(
    VertexIn vIn,
    out VertexOut vOut)
{
    vOut.pos = mul(float4(vIn.pos, 1), gConst.viewProject);
    vOut.norm = vIn.norm;
    vOut.view = gConst.cameraPos.xyz - vIn.pos;
}

// 神经网络BSDF函数 - DisneyMLP
float4 DisneyMLP(float NdotL, float NdotV, float NdotH, float LdotH, float roughness)
{   
    uint4 weightOffsets = gConst.weightOffsets; 
    uint4 biasOffsets = gConst.biasOffsets;  

   CoopVec<VECTOR_FORMAT, INPUT_NEURONS> inputParams;
   CoopVec<VECTOR_FORMAT, HIDDEN_NEURONS> hiddenParams;
   CoopVec<VECTOR_FORMAT, OUTPUT_NEURONS> outputParams;

   // Encode input parameters, 5 inputs to 30 parameters 
   //   INPUT_FEATURES：编码后的维度（通常是原始维度的倍数）
   //   使用频率编码提升高频细节学习能力
   //   半精度存储平衡速度和精度
   float params[INPUT_FEATURES] = { NdotL, NdotV, NdotH, LdotH, roughness };
   inputParams = rtxns::EncodeFrequency<half, INPUT_FEATURES>(params);

   // Forward propagation through the neural network
   // Input to hidden layer, then apply activation function
   // 前向传播网络结构:
   //       4层MLP：输入层 → 隐藏层1 → 隐藏层2 → 隐藏层3 → 输出层
   //       激活函数：ReLU（线性整流，计算高效）
   //       矩阵布局：InferencingOptimal（推理优化布局，不同于训练时的TrainingOptimal）
   hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, INPUT_NEURONS>(
       inputParams, gMLPParams, weightOffsets[0], biasOffsets[0], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   hiddenParams = rtxns::relu(hiddenParams);
   
   // Hidden layer to hidden layer, then apply activation function 
   hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
       hiddenParams, gMLPParams, weightOffsets[1], biasOffsets[1], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   hiddenParams = rtxns::relu(hiddenParams);
   
   // Hidden layer to hidden layer, then apply activation function    
   hiddenParams = rtxns::LinearOp<VECTOR_FORMAT, HIDDEN_NEURONS, HIDDEN_NEURONS>(
       hiddenParams, gMLPParams, weightOffsets[2], biasOffsets[2], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   hiddenParams = rtxns::relu(hiddenParams);
   
   // Hidden layer to output layer, then apply final activation function
   // 输出层：
   //       指数激活：exp(outputParams)确保输出为正（BSDF值应为正）
   //       输出维度：4个分量（对应BSDF的分解项）
   outputParams = rtxns::LinearOp<VECTOR_FORMAT, OUTPUT_NEURONS, HIDDEN_NEURONS>(
       hiddenParams, gMLPParams, weightOffsets[3], biasOffsets[3], CoopVecMatrixLayout::InferencingOptimal, TYPE_INTERPRETATION);
   outputParams = exp(outputParams);

    // Take the output from the neural network as the output color
    return float4(outputParams[0], outputParams[1], outputParams[2], outputParams[3]);
}

[shader("fragment")]
void main_ps( 
    VertexOut vOut,
    out float4 o_color : SV_Target0)
{
    float4 lightIntensity = gConst.lightIntensity;
    float4 lightDir =  gConst.lightDir;
    float4 baseColor = gConst.baseColor;
    float specular = gConst.specular;
    float roughness = gConst.roughness;
    float metallic = gConst.metallic;

    // Prepare input parameters
    float3 view = normalize(vOut.view);
    float3 norm = normalize(vOut.norm);
    float3 h = normalize(-lightDir.xyz + view);

    float NdotL = max(0.f, dot(norm, -lightDir.xyz));
    float NdotV = max(0.f, dot(norm, view));
    float NdotH = max(0.f, dot(norm, h));
    float LdotH = max(0.f, dot(h, -lightDir.xyz));

    // Calculate approximated core shader part using MLP
    // MLP输出的4个分量可能代表BSDF的不同组成部分：
    //      outParams.x：漫反射分量权重
    //      outParams.y：镜面反射分量权重
    //      outParams.z：菲涅尔项调整
    //      outParams.w：附加分量（如环境光遮蔽）
    float4 outParams = DisneyMLP(NdotL, NdotV, NdotH, LdotH, roughness);

    // Calculate final color
    // Gamma校正：将sRGB转线性空间（pow 2.2）
    float3 Cdlin = float3(pow(baseColor.r, 2.2), pow(baseColor.g, 2.2), pow(baseColor.b, 2.2));

    // 镜面反射颜色:
    //      非金属：固定值0.08 * specular
    //      金属：使用baseColor
    float3 Cspec0 = lerp(specular * .08f * float3(1,1,1), Cdlin, metallic);
    
    // BSDF合成:
    //      项1：漫反射贡献（非金属部分）
    //      项2：镜面反射贡献
    //      项3：附加贡献（可能的环境光）
    float3 brdfn = outParams.x * Cdlin * (1 - metallic) 
                    + outParams.y * lerp(Cspec0, float3(1), outParams.z) 
                    + outParams.w;
    // 光照计算：颜色 = BSDF * 余弦项 * 光照强度
    float3 colorh = brdfn * float3(NdotL) * lightIntensity.rgb;

    o_color = float4(colorh, 1.f);
 }
