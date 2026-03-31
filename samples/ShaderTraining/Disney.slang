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
特点总结
    1. 分层材质: 基础层(漫反射+镜面反射) + 清漆层
    2. 能量守恒: 所有分布函数都经过归一化
    3. 性能优化: 使用近似公式而非完整物理积分
    4. Disney原则: 遵循"艺术导向"的参数化设计
 */
//----------- Core part of the shader

const static float PI = 3.14159265358979323846;

// 实现Schlick近似Fresnel方程，用于计算不同入射角下的反射率
float SchlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

// GTR1 (Gtr1) - 用于清漆层：
//  各向同性的GGX分布变体
//  当粗糙度接近1时返回均匀分布(1 / π)
//  使用对数函数确保能量守恒
float Gtr1(float NdotH, float a)
{
    if (a >= 1)
    {
        return 1 / PI;
    }
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

// GTR2 (Gtr2) - 用于基础层：
//  标准的GGX / Trowbridge - Reitz分布
//  处理各向同性粗糙度
float Gtr2(float NdotH, float ax)
{
    float a = ax * (1 / ax / ax * (1 - NdotH * NdotH) + NdotH * NdotH);
    return 1 / (PI * a * a);
}

// SmithGGX - 各向同性版本：标准的Smith几何遮蔽项
float SmithGGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

// 各向异性版本:处理各向异性粗糙度
float SmithGGXAnisotropy(float NdotV, float ax)
{
    return 1 / (NdotV + sqrt(ax * ax * (1 - NdotV * NdotV) + NdotV * NdotV));
}

float4 Disney(float NdotL, float NdotV, float NdotH, float LdotH, float roughness)
{
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fss90 = LdotH * LdotH * roughness;
    float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - .5f) + .5f);

    // specular
    float ax = max(.001f, roughness * roughness);
    float Ds = Gtr2(NdotH, ax);
    float FH = SchlickFresnel(LdotH);
    float Gs = SmithGGXAnisotropy(NdotL, ax);
    Gs *= SmithGGXAnisotropy(NdotV, ax);

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = Gtr1(NdotH, .01f);
    float Fr = lerp(.04f, 1.0f, FH);
    float Gr = SmithGGX(NdotL, .25f) * SmithGGX(NdotV, .25f);

    return float4((1 / PI) * ss, Gs * Ds, FH, .25 * Gr * Fr * Dr);
}
