/**
 *	Copyright (c) 2025 Wenchao Huang <physhuangwenchao@gmail.com>
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in all
 *	copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *	SOFTWARE.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <photon/macros.h>
#include <photon/payload.cuh>
#include <photon/sbt_record.h>
#include <device_launch_parameters.h>
#include <optix_device.h>
#include "launch_params.h"

__RT_CONSTANT__ LaunchParams launchParams;

/*********************************************************************************
*******************************    Utilities    **********************************
*********************************************************************************/

static __device__ __forceinline__ float3 make_float3_v(float x, float y, float z)
{
	return float3{ x, y, z };
}

static __device__ __forceinline__ float3 operator+(float3 a, float3 b)
{
	return float3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

static __device__ __forceinline__ float3 operator-(float3 a, float3 b)
{
	return float3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

static __device__ __forceinline__ float3 operator*(float s, float3 a)
{
	return float3{ s * a.x, s * a.y, s * a.z };
}

static __device__ __forceinline__ float3 operator*(float3 a, float3 b)
{
	return float3{ a.x * b.x, a.y * b.y, a.z * b.z };
}

static __device__ __forceinline__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __device__ __forceinline__ float3 normalize(float3 v)
{
	float invLen = rsqrtf(dot(v, v));
	return invLen * v;
}

static __device__ __forceinline__ float3 cross(float3 a, float3 b)
{
	return float3{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

/*********************************************************************************
*****************************    Phong Shading    ********************************
*********************************************************************************/

static __device__ float3 phongShade(float3 hitPoint, float3 normal, const Material & mat)
{
	float3 L = normalize(launchParams.lightPos - hitPoint);
	float3 V = normalize(launchParams.camPos - hitPoint);
	float3 R = normalize(2.0f * dot(normal, L) * normal - L);

	//	Ambient
	float3 ambient = launchParams.ambientColor * mat.diffuse;

	//	Diffuse
	float NdotL = fmaxf(dot(normal, L), 0.0f);
	float3 diffuse = NdotL * (launchParams.lightColor * mat.diffuse);

	//	Specular
	float RdotV = fmaxf(dot(R, V), 0.0f);
	float spec = powf(RdotV, mat.shininess);
	float3 specular = spec * (launchParams.lightColor * mat.specular);

	return ambient + diffuse + specular;
}

/*********************************************************************************
*****************************    Payload Types    ********************************
*********************************************************************************/

using PayloadColor = pt::Payload<float3, 0, 1, 2>;

/*********************************************************************************
*********************************    Raygen    ***********************************
*********************************************************************************/

__RT_KERNEL__ void __raygen__()
{
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	//	Compute normalized pixel coordinates [-1, 1]
	float u = 2.0f * (float(idx.x) + 0.5f) / float(dim.x) - 1.0f;
	float v = 2.0f * (float(idx.y) + 0.5f) / float(dim.y) - 1.0f;

	//	Compute ray direction
	float3 rayDir = normalize(u * launchParams.camU + v * launchParams.camV + launchParams.camW);

	//	Initialize payload
	PayloadColor payload = float3{ 0.0f, 0.0f, 0.0f };

	optixTrace(launchParams.traversable,
			   launchParams.camPos,
			   rayDir,
			   0.0f,					//	tmin
			   1e16f,					//	tmax
			   0.0f,					//	rayTime
			   OptixVisibilityMask(255),
			   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
			   0,						//	SBT offset
			   1,						//	SBT stride
			   0,						//	miss SBT index
			   payload.encodes[0],
			   payload.encodes[1],
			   payload.encodes[2]);

	//	Write to output image
	unsigned int pixelIdx = idx.y * dim.x + idx.x;
	launchParams.image.data()[pixelIdx] = payload.value;
}

/*********************************************************************************
*********************************    Miss    *************************************
*********************************************************************************/

__RT_KERNEL__ void __miss__()
{
	//	Background gradient (sky color)
	float3 rayDir = optixGetWorldRayDirection();
	float t = 0.5f * (rayDir.y + 1.0f);
	float3 bgColor = (1.0f - t) * float3{ 1.0f, 1.0f, 1.0f } + t * float3{ 0.5f, 0.7f, 1.0f };

	pt::set_payload<float3, 0, 1, 2>(bgColor);
}

/*********************************************************************************
***********************    ClosestHit - Triangle    ******************************
*********************************************************************************/

__RT_KERNEL__ void __closesthit__triangle()
{
	const HitGroupData * data = reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());

	//	Get triangle barycentrics
	float2 bary = optixGetTriangleBarycentrics();
	unsigned int primIdx = optixGetPrimitiveIndex();

	//	Get vertex indices
	unsigned int i0 = data->indices.data()[primIdx * 3 + 0];
	unsigned int i1 = data->indices.data()[primIdx * 3 + 1];
	unsigned int i2 = data->indices.data()[primIdx * 3 + 2];

	//	Get vertex positions
	float3 v0 = data->vertices.data()[i0];
	float3 v1 = data->vertices.data()[i1];
	float3 v2 = data->vertices.data()[i2];

	//	Compute geometric normal
	float3 e1 = v1 - v0;
	float3 e2 = v2 - v0;
	float3 normal = normalize(cross(e1, e2));

	//	Compute hit point using barycentrics
	float3 hitPoint = (1.0f - bary.x - bary.y) * v0 + bary.x * v1 + bary.y * v2;

	//	Transform to world space
	hitPoint = optixTransformPointFromObjectToWorldSpace(hitPoint);
	normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));

	float3 color = phongShade(hitPoint, normal, data->material);
	pt::set_payload<float3, 0, 1, 2>(color);
}

/*********************************************************************************
***********************    ClosestHit - Sphere    ********************************
*********************************************************************************/

__RT_KERNEL__ void __closesthit__sphere()
{
	const HitGroupData * data = reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());

	unsigned int primIdx = optixGetPrimitiveIndex();
	float3 center = data->sphereCenters.data()[primIdx];

	//	Compute hit point from ray
	float tHit = optixGetRayTmax();
	float3 rayOrigin = optixGetWorldRayOrigin();
	float3 rayDir = optixGetWorldRayDirection();
	float3 hitPoint = rayOrigin + tHit * rayDir;

	//	Transform center to world space for normal computation
	float3 worldCenter = optixTransformPointFromObjectToWorldSpace(center);
	float3 normal = normalize(hitPoint - worldCenter);

	float3 color = phongShade(hitPoint, normal, data->material);
	pt::set_payload<float3, 0, 1, 2>(color);
}

/*********************************************************************************
***********************    ClosestHit - Curve    *********************************
*********************************************************************************/

__RT_KERNEL__ void __closesthit__curve()
{
	const HitGroupData * data = reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());

	//	For curves, use the built-in normal attribute
	float tHit = optixGetRayTmax();
	float3 rayOrigin = optixGetWorldRayOrigin();
	float3 rayDir = optixGetWorldRayDirection();
	float3 hitPoint = rayOrigin + tHit * rayDir;

	//	Approximate normal as perpendicular to the ray direction at the hit point
	//	For a cylinder-like curve, compute the normal from hit point to curve axis
	float3 normal = normalize(hitPoint - optixTransformPointFromObjectToWorldSpace(
		make_float3_v(hitPoint.x, hitPoint.y, hitPoint.z)));

	//	Fallback: use ray direction perpendicular
	if (dot(normal, normal) < 1e-6f)
	{
		//	Use a simple normal perpendicular to the ray
		float3 up = make_float3_v(0.0f, 1.0f, 0.0f);
		if (fabsf(dot(rayDir, up)) > 0.9f)
			up = make_float3_v(1.0f, 0.0f, 0.0f);
		normal = normalize(cross(rayDir, up));
	}

	float3 color = phongShade(hitPoint, normal, data->material);
	pt::set_payload<float3, 0, 1, 2>(color);
}

/*********************************************************************************
***********************    Intersection - AABB    ********************************
*********************************************************************************/

__RT_KERNEL__ void __intersection__aabb()
{
	const HitGroupData * data = reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());

	unsigned int primIdx = optixGetPrimitiveIndex();
	float3 center = data->aabbCenters.data()[primIdx];
	float radius = data->aabbRadius;

	//	Ray-sphere intersection for custom AABB primitive
	float3 rayOrigin = optixGetObjectRayOrigin();
	float3 rayDir = optixGetObjectRayDirection();

	float3 oc = rayOrigin - center;
	float a = dot(rayDir, rayDir);
	float b = dot(oc, rayDir);
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;

	if (discriminant >= 0.0f)
	{
		float sqrtDisc = sqrtf(discriminant);
		float t = (-b - sqrtDisc) / a;

		if (t < optixGetRayTmin())
			t = (-b + sqrtDisc) / a;

		if (t >= optixGetRayTmin() && t <= optixGetRayTmax())
		{
			//	Report hit; attribute 0 stores the normal info
			float3 hitLocal = rayOrigin + t * rayDir;
			float3 normal = normalize(hitLocal - center);
			unsigned int n0, n1, n2;
			memcpy(&n0, &normal.x, sizeof(unsigned int));
			memcpy(&n1, &normal.y, sizeof(unsigned int));
			memcpy(&n2, &normal.z, sizeof(unsigned int));
			optixReportIntersection(t, 0, n0, n1, n2);
		}
	}
}

/*********************************************************************************
***********************    ClosestHit - AABB    **********************************
*********************************************************************************/

__RT_KERNEL__ void __closesthit__aabb()
{
	const HitGroupData * data = reinterpret_cast<const HitGroupData *>(optixGetSbtDataPointer());

	//	Recover normal from attributes
	float3 normal;
	unsigned int a0 = optixGetAttribute_0();
	unsigned int a1 = optixGetAttribute_1();
	unsigned int a2 = optixGetAttribute_2();
	memcpy(&normal.x, &a0, sizeof(float));
	memcpy(&normal.y, &a1, sizeof(float));
	memcpy(&normal.z, &a2, sizeof(float));

	normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));

	float tHit = optixGetRayTmax();
	float3 rayOrigin = optixGetWorldRayOrigin();
	float3 rayDir = optixGetWorldRayDirection();
	float3 hitPoint = rayOrigin + tHit * rayDir;

	float3 color = phongShade(hitPoint, normal, data->material);
	pt::set_payload<float3, 0, 1, 2>(color);
}
