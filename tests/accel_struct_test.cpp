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

#include <nucleus/device.h>
#include <nucleus/stream.h>
#include <nucleus/context.h>
#include <nucleus/array_1d.h>

#include <photon/accel_struct.h>
#include <photon/device_context.h>

/*********************************************************************************
****************************    accel_struct_test    *****************************
*********************************************************************************/

void accel_struct_test()
{
	auto device = ns::Context::getInstance()->device(0);
	auto deviceContext = pt::SharedContext(device);
	auto allocator = device->defaultAllocator();
	auto & stream = device->defaultStream();
	unsigned int geomFlags = OPTIX_GEOMETRY_FLAG_NONE;
	CUdeviceptr aabbPtr = 0;
	CUdeviceptr vertexPtr = 0;
	CUdeviceptr radiusPtr = 0;
	CUdeviceptr widthPtr = 0;
	uint32_t triIndices[1][3] = {};
	uint32_t curveIndices[1] = {};
	pt::BuildInputTriangles triangleBuildInput;
	triangleBuildInput.setVertexBuffers<ns::float3_16a>(&vertexPtr)
					  .setNumVertices(0)
					  .setIndexBuffer(triIndices)
					  .setNumIndexTriplets(1)
					  .setNumSbtRecords(1)
					  .setFlags(&geomFlags);
	pt::BuildInputAabb aabbBuildInput;
	aabbBuildInput.setAabbBuffers<pt::Aabb>(&aabbPtr)
				  .setNumPrimitives(0)
				  .setNumSbtRecords(1)
				  .setFlags(&geomFlags);
#if OPTIX_VERSION >= 70100
	pt::BuildInputCurves curveBuildInput;
	curveBuildInput.setCurveType(OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
				   .setVertexBuffers<ns::float3_16a>(&vertexPtr)
				   .setWidthBuffers<float>(&widthPtr)
				   .setIndexBuffer(curveIndices)
				   .setNumVertices(0)
				   .setNumPrimitives(1);
#endif
#if OPTIX_VERSION >= 70500
	pt::BuildInputSphere sphereBuildInput;
	sphereBuildInput.setVertexBuffers<ns::float3_16a>(&vertexPtr)
					.setRadiusBuffers<float>(&radiusPtr)
					.setNumVertices(0)
					.setNumSbtRecords(1)
					.setFlags(&geomFlags);
#endif
	pt::BuildInputInstance instanceBuildInput;
	instanceBuildInput.setInstanceId(1)
					  .setSbtOffset(2)
					  .setVisibilityMask(pt::BuildInputInstance::DefaultVisibilityMask)
					  .setFlags(0)
					  .setTraversableHandle(0);
	assert(triangleBuildInput.vertexFormat == OPTIX_VERTEX_FORMAT_FLOAT3);
	assert(triangleBuildInput.indexFormat == OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
	assert(triangleBuildInput.indexStrideInBytes == sizeof(triIndices[0]));
	assert(aabbBuildInput.strideInBytes == sizeof(pt::Aabb));
#if OPTIX_VERSION >= 70100
	assert(curveBuildInput.indexStrideInBytes == sizeof(curveIndices[0]));
#endif
	assert(instanceBuildInput.visibilityMask == pt::BuildInputInstance::DefaultVisibilityMask);

	auto instAccelStruct = deviceContext->createInstAccelStruct();
	auto accelStructAabb = deviceContext->createAccelStructAabb();
	auto accelStructTriangle = deviceContext->createAccelStructTriangle();

	assert(instAccelStruct != nullptr);
	assert(accelStructAabb != nullptr);
	assert(accelStructTriangle != nullptr);
#if OPTIX_VERSION >= 70100
	auto accelStructCurve = deviceContext->createAccelStructCurve();
	assert(accelStructCurve != nullptr);
#endif
#if OPTIX_VERSION >= 70500
	auto accelStructSphere = deviceContext->createAccelStructSphere();
	assert(accelStructSphere != nullptr);
#endif

	accelStructAabb->refit(stream);
}
