Shader "Hidden/Screen Space Outlines"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        HLSLINCLUDE

        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareNormalsTexture.hlsl"

        #define NEIGHBOR_OFFSET_COUNT 8
        static const float2 _NeighborOffsets[NEIGHBOR_OFFSET_COUNT] = 
        {
			float2(-1.0f, 1.0f),
			float2(0.0f, 1.0f),
			float2(1.0f, 1.0f),

            float2(-1.0f, 0.0f),
			float2(1.0f, 0.0f),

			float2(-1.0f, -1.0f),
			float2(0.0f, -1.0f),
			float2(1.0f, -1.0f),
        };

        #define EXTENDED_NEIGHBOR_OFFSET_COUNT 24
        static const float2 _ExtendedNeighborOffsets[EXTENDED_NEIGHBOR_OFFSET_COUNT] = 
		{
			float2(-1.0f, 0.0f),
            float2(1.0f, 0.0f),
            float2(0.0f, -1.0f),
            float2(0.0f, 1.0f),
            float2(-1.0f, -1.0f),
            float2(-1.0f, 1.0f),
            float2(1.0f, -1.0f),
            float2(1.0f, 1.0f),
            float2(-2.0f, 0.0f),
            float2(2.0f, 0.0f),
            float2(0.0f, -2.0f),
            float2(0.0f, 2.0f),
            float2(-2.0f, -1.0f),
            float2(-2.0f, 1.0f),
            float2(2.0f, -1.0f),
            float2(2.0f, 1.0f),
            float2(-1.0f, -2.0f),
            float2(-1.0f, 2.0f),
            float2(1.0f, -2.0f),
            float2(1.0f, 2.0f),
            float2(-2.0f, -2.0f),
            float2(-2.0f, 2.0f),
            float2(2.0f, -2.0f),
            float2(2.0f, 2.0f),
		};

        #define ONE_THIRD 0.3333333333333f

        static const float2x4 _UVFromFeature =
        {
            float4(1.0f, 0.0f, 0.0f, 0.0f),
            float4(0.0f, 1.0f, 0.0f, 0.0f),
        };

        static const float4 _FeatureWeights = float4(0.0f, 0.0f, 1.0f, 0.0f);

        uniform TEXTURE2D(_MainTex);
        uniform SAMPLER(sampler_MainTex);
        uniform float4 _MainTex_TexelSize;

        uniform int2 _JfaExtents;

        uniform float _ColorThresholdFactor;
        uniform float _NormalThresholdFactor;
        uniform float _GeometryThresholdFactor;
        uniform float _OutlineThreshold;

        uniform float4 _OutlineColor;
        uniform float _OutlineIntensity;
        uniform float _OutlineDrawTightness;
        uniform float _OutlineDrawThreshold;

        uniform float _ConvexityThreshold;
        uniform float _ConcavityThreshold;
        uniform float4 _ConvexityOverlayColor;
        uniform float4 _ConcavityOverlayColor;

        uniform float _DistanceFadeFactor;

        struct appdata
        {
            float4 vertex : POSITION;
            float2 uv : TEXCOORD0;
        };

        struct v2f
        {
            float2 uv : TEXCOORD0;
            float4 vertex : SV_POSITION;
        };

        v2f vert(appdata v)
        {
            v2f o;
            o.vertex = TransformObjectToHClip(v.vertex);
            o.uv = v.uv;
            return o;
        }

        float ChebyshovDistance(float2 from, float2 to)
        {
			float2 displacement = abs(to - from);
			return max(displacement.x, displacement.y);
		}

        float TaxicabDistance(float2 from, float2 to)
		{
            float2 displacement = abs(to - from);
			return displacement.x + displacement.y;
        }

        float EuclideanSquared(float2 from, float2 to)
        {
			float2 displacement = to - from;
            return dot(displacement, displacement);
        }

        ENDHLSL

        Pass
        {
            Name "Outline Distance Field"

            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag

            float3 WorldPositionFromRawDepthAt(float2 screenUV, float rawDepth)
		    {
			    #if !UNITY_REVERSED_Z
			        // Adjust Z to match NDC for OpenGL ([-1, 1])
			        rawDepth = lerp(UNITY_NEAR_CLIP_VALUE, 1.0f, rawDepth);
			    #endif

			    // Reconstruct the world space positions.
			    return ComputeWorldSpacePosition(screenUV, rawDepth, UNITY_MATRIX_I_VP);
		    }

            float LinearEyeDepthFromRaw(float rawDepth)
		    {
			    float ortho = unity_OrthoParams.w;

			    float orthoLinearDepth = _ProjectionParams.x > 0 ? rawDepth : 1 - rawDepth;
			    float orthoEyeDepth = lerp(_ProjectionParams.y, _ProjectionParams.z, orthoLinearDepth);

			    return LinearEyeDepth(rawDepth, _ZBufferParams) * (1.0f - ortho) + orthoEyeDepth * ortho;
		    }

            float SDPlane(float3 planePoint, float3 planeNormal, float3 position)
            {
                return dot(position - planePoint, planeNormal);
            }
            
		    float RedMeanColorDifference(float3 c1, float3 c2)
		    {			
			     float3 difference = c2 - c1;
			     float differenceSquared = dot(difference, difference);
			     float redMean = (c1.r + c2.r) * 0.5f;
            
			     float3 k = float3(2.0f + redMean, 4.0f, 3.0f - redMean);
			     return sqrt(dot(k, differenceSquared));// / 9.0f;
		    }            

            float NormalDifference(float3 n1, float3 n2)
            {
                return 1.0f - dot(n1, n2);
            }

            void GetColorDepthNormalsAt(float2 uv, out float3 color, out float rawDepth, out float3 normal)
            {
                color = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, uv).rgb;
                normal = SampleSceneNormals(uv).xyz;
                rawDepth = SampleSceneDepth(uv).r;
            }

            float SDEdge(float2 uv, out float curvature, out float rawDepth, out float2 edgeUV)
            {
                curvature = 0.0f;
                edgeUV = uv;
                float minDistanceToEdge = 1.#INF;

                float3 centralColor = 0.0f;
                float3 centralNormalWS = 0.0f;
                GetColorDepthNormalsAt(uv, centralColor, rawDepth, centralNormalWS);

                float3 centralPositionWS = WorldPositionFromRawDepthAt(uv, rawDepth);
                float positionVariance = 0.0f;

                [unroll]
                for (uint i = 0; i < NEIGHBOR_OFFSET_COUNT; ++i)
                {
                    float2 neighborOffset = _NeighborOffsets[i] * _MainTex_TexelSize.xy;
                    float2 neighborUV = uv + neighborOffset;

                    float3 neighborColor = 0.0f;
                    float3 neighborNormalWS = 0.0f;
                    float neighborRawDepth = 0.0f;
                    GetColorDepthNormalsAt(neighborUV, neighborColor, neighborRawDepth, neighborNormalWS);

                    float3 neighborPositionWS = WorldPositionFromRawDepthAt(neighborUV, neighborRawDepth);

                    float colorDistance = RedMeanColorDifference(centralColor, neighborColor);
                    float normalDistance = NormalDifference(centralNormalWS, neighborNormalWS);
                    float geometricDistance = SDPlane(centralPositionWS, centralNormalWS, neighborPositionWS);

                    curvature -= geometricDistance;
                    float positionDifferenceWS = neighborPositionWS - centralPositionWS;
                    positionVariance += dot(positionDifferenceWS, positionDifferenceWS);

                    float2 tentativeEdgeUV = lerp(uv, neighborUV, 0.5f);
                    float tentativeEdgeDistance = TaxicabDistance(uv, tentativeEdgeUV);

                    if ((colorDistance > _ColorThresholdFactor
                        || normalDistance > _NormalThresholdFactor
                        || abs(geometricDistance) > _GeometryThresholdFactor)
                            && tentativeEdgeDistance < minDistanceToEdge)
                    {
						edgeUV = tentativeEdgeUV;
                        minDistanceToEdge = tentativeEdgeDistance;
                    }
                }

                curvature /= sqrt(positionVariance / NEIGHBOR_OFFSET_COUNT);
                return minDistanceToEdge;
            }

            float4 frag(v2f i) : SV_TARGET
            {
                float curvature = 0.0f;
                float rawDepth = 0.0f;
                float2 edgeUV = i.uv;
                float minDistanceToEdge = SDEdge(i.uv, curvature, rawDepth, edgeUV);
                return float4(edgeUV, curvature * 0.5f + 0.5f, minDistanceToEdge);
            }

            ENDHLSL
        }

        Pass
        {
            Name "JFA"

            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag

            bool HasFiniteDistanceToEdge(float4 feature)
            {
                return (feature.w < 1.0f);
            }

            float DistanceToFeature(float2 uv, float2 featureUV, float4 feature)
            {
                return TaxicabDistance(uv, featureUV);
            //    float2 featureDisplacement = featureUV - uv;
            //    return dot(featureDisplacement, featureDisplacement);
            }

            bool IsSeeded(float4 feature, out float2 featureUV)
            {
                //More correct
                //featureUV = mul(uvFromFeature, feature);
                featureUV = feature.xy;
                return HasFiniteDistanceToEdge(feature);
            }

            float4 ClosestFeatureAt(float2 uv, int2 extentsSS, out float minDistanceToFeature)
			{
                float4 closestFeature = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, uv);
                float2 featureUV = 0.0f;
                minDistanceToFeature = IsSeeded(closestFeature, featureUV)
									   ? DistanceToFeature(uv, featureUV, closestFeature)
									   : 1.#INF;

                [unroll]
                for (uint i = 0; i < NEIGHBOR_OFFSET_COUNT; ++i)
                {
                    float2 neighborOffset = _NeighborOffsets[i] * _MainTex_TexelSize.xy * extentsSS;
                    float2 neighborUV = frac(uv + neighborOffset);

                    float4 neighborFeature = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, neighborUV);
                    float2 neighborFeatureUV = 0.0f;
                    if (!IsSeeded(neighborFeature, neighborFeatureUV))
                        continue;
                    
                    float distanceToNeighborFeature = DistanceToFeature(uv, neighborFeatureUV, neighborFeature);
                    if (distanceToNeighborFeature < minDistanceToFeature)
                    {
					    closestFeature = neighborFeature;
                    	minDistanceToFeature = distanceToNeighborFeature;
                    }
                }
                return closestFeature;
            }

            float4 ClosestFeatureInAxisWith(float4 centralFeature, float2 uv, int2 extentsSS, float2 axisDirection, out float minDistanceToFeature)
			{
                float4 closestFeature = centralFeature;
                float2 featureUV = 0.0f;
                minDistanceToFeature = IsSeeded(closestFeature, featureUV)
									   ? DistanceToFeature(uv, featureUV, closestFeature)
									   : 1.#INF;

                [unroll]
                for (int i = -1; i <= 1; i += 2)
                {
                    float2 neighborOffset = axisDirection * i * _MainTex_TexelSize.xy * extentsSS;
                    float2 neighborUV = frac(uv + neighborOffset);

                    float4 neighborFeature = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, neighborUV);
                    float2 neighborFeatureUV = 0.0f;
                    if (!IsSeeded(neighborFeature, neighborFeatureUV))
                        continue;
                    
                    float distanceToNeighborFeature = DistanceToFeature(uv, neighborFeatureUV, neighborFeature);
                    if (distanceToNeighborFeature < minDistanceToFeature)
                    {
					    closestFeature = neighborFeature;
                    	minDistanceToFeature = distanceToNeighborFeature;
                    }
                }
                return closestFeature;
            }


            float4 frag(v2f i) : SV_TARGET
            {
                float4 closestFeature = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv);

                float distanceToClosestFeatureX = 1.#INF;
                float4 closestFeatureX = ClosestFeatureInAxisWith(closestFeature, i.uv, _JfaExtents, float2(1.0f, 0.0f), distanceToClosestFeatureX);

                float distanceToClosestFeatureY = 1.#INF;
				float4 closestFeatureY = ClosestFeatureInAxisWith(closestFeature, i.uv, _JfaExtents, float2(0.0f, 1.0f), distanceToClosestFeatureY);

                float xFurtherThanY = step(distanceToClosestFeatureY, distanceToClosestFeatureX);
                return xFurtherThanY ? closestFeatureY : closestFeatureX;
            }
            ENDHLSL
        }

        Pass
        {
            Name "Outlines Composite"
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag

            float4 Overlay(float4 a, float4 b)
            {
                float isAboveGrey = step(0.5f, a);
                return (2.0f * a * b) * isAboveGrey +
                       (1.0f - 2.0f * (1.0f - a) * (1.0f - b)) * (1.0f - isAboveGrey);
            }

		    float4 Screen(float4 a, float4 b)
		    {
			    return 1.0f - (1.0f - a) * (1.0f - b);
		    }

            float4 SoftLight(float4 a, float4 b)
		    {
			    return (1.0f - 2.0f * b) * a * a + 2.0f * b * a;
		    }

            float4 HardLight(float4 a, float4 b)
            {
                float isAboveGrey = step(0.5f, a);
		        return (1.0f - (1.0f - b) * (1.0f - 2.0f * (a - 0.5f))) * isAboveGrey +
                       (a * b * 2.0f) * (1.0f - isAboveGrey);
            }

            float4 PinLight(float4 a, float4 b)
		    {
                float isAboveGrey = step(0.5f, a);
			    return (max(b, 2.0f * (a - 0.5f))) * isAboveGrey +
                       (min(b, 2.0f * a)) * (1.0f - isAboveGrey);
		    }

            float4 frag(v2f i) : SV_TARGET
            {
	            float4 feature = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv);
                //More correct
                //float2 featureUV = mul(_UVFromFeature, feature);
                if (feature.w == 1.0f)
                    return 0.0f;
                float2 featureUV = feature.xy;
                float distanceToFeature = distance(i.uv, featureUV);

                float outlineFactor = saturate(1.0f - distanceToFeature);
                float lowerEdge = _OutlineDrawThreshold - _OutlineDrawTightness;
                float upperEdge = _OutlineDrawThreshold + _OutlineDrawTightness;

                outlineFactor = pow(outlineFactor, _OutlineIntensity);
                outlineFactor = smoothstep(lowerEdge, upperEdge, outlineFactor);

                float linearEyeDepth = LinearEyeDepth(SampleSceneDepth(i.uv).r, _ZBufferParams);
                float outlineAlpha = outlineFactor * exp(-pow(linearEyeDepth * _DistanceFadeFactor, 2));

                float curvature = feature.z * 2.0f - 1.0f;
                float curvatureFactor =
                    smoothstep(
                        _ConcavityThreshold,
                        _ConvexityThreshold,
                        saturate((curvature * outlineFactor) * 0.5f + 0.5f));

                float4 curvatureColor = lerp(_ConcavityOverlayColor, _ConvexityOverlayColor, curvatureFactor);
                float4 compositeColor = HardLight(_OutlineColor, curvatureColor);
                return float4(compositeColor.rgb, compositeColor.a * outlineAlpha);
            }

            ENDHLSL
		}
    }
}
