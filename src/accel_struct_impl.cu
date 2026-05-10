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

#include "device_context.h"
#include <photon/accel_struct.h>
#include <nucleus/launch_utils.cuh>
#include <optix_stubs.h>

PHOTON_USING_NAMESPACE

/*********************************************************************************
*******************************    Validations    ********************************
*********************************************************************************/

static_assert(static_cast<int>(GeomAccelStruct::GeomFlags::None)								== OPTIX_GEOMETRY_FLAG_NONE);
static_assert(static_cast<int>(GeomAccelStruct::GeomFlags::DisableAnyhit)						== OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
static_assert(static_cast<int>(GeomAccelStruct::GeomFlags::RequireSingleAnyhitCall)				== OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
#if OPTIX_VERSION >= 70500
static_assert(static_cast<int>(GeomAccelStruct::GeomFlags::DisableTriangleFaceCulling)			== OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING);
#endif

#if OPTIX_VERSION >= 70100
static_assert(static_cast<int>(AccelStructCurve::CurveType::RoundLinear)						== OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR);
static_assert(static_cast<int>(AccelStructCurve::CurveType::RoundCubicBSpline)					== OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE);
static_assert(static_cast<int>(AccelStructCurve::CurveType::RoundQuadraticBSpline)				== OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE);
#endif
#if OPTIX_VERSION >= 70400
static_assert(static_cast<int>(AccelStructCurve::CurveType::RoundCatmullRom)					== OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM);
#endif
#if OPTIX_VERSION >= 70700
static_assert(static_cast<int>(AccelStructCurve::CurveType::RoundCubicBezier)					== OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER);
static_assert(static_cast<int>(AccelStructCurve::CurveType::FlatQuadraticBSpline)				== OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE);
#endif

static_assert(static_cast<int>(InstAccelStruct::InstFlags::None)								== OPTIX_INSTANCE_FLAG_NONE);
static_assert(static_cast<int>(InstAccelStruct::InstFlags::DisableAnyhit)						== OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT);
static_assert(static_cast<int>(InstAccelStruct::InstFlags::EnforceAnyhit)						== OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT);
static_assert(static_cast<int>(InstAccelStruct::InstFlags::FlipTriangleFacing)					== OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING);
static_assert(static_cast<int>(InstAccelStruct::InstFlags::DisableTriangleFaceCulling)			== OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING);
#if OPTIX_VERSION >= 70600
static_assert(static_cast<int>(InstAccelStruct::InstFlags::DisableOpacityMicromaps)				== OPTIX_INSTANCE_FLAG_DISABLE_OPACITY_MICROMAPS);
static_assert(static_cast<int>(InstAccelStruct::InstFlags::ForceOpacityMicromapAsTwoState)		== OPTIX_INSTANCE_FLAG_FORCE_OPACITY_MICROMAP_2_STATE);
#endif

/*********************************************************************************
********************************    AccelStruct    ********************************
*********************************************************************************/

AccelStruct::AccelStruct(std::shared_ptr<DeviceContext> deviceContext) : m_deviceContext(deviceContext), m_hTraversable(0), m_numSbtRecords(0), m_headerSize(0)
{
	m_buildOptions = OptixAccelBuildOptions{};
}


void AccelStruct::buildBase(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, OptixAccelBuildOptions buildOptions, size_t headerSize)
{
	OptixAccelBufferSizes accelBufferSizes = {};

	buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixResult err = optixAccelComputeMemoryUsage(m_deviceContext->handle(), &buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(), &accelBufferSizes);

	if (err == OPTIX_SUCCESS)
	{
		headerSize = ns::align_up(headerSize, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);

		//!	Last aligned 8-bytes for storing compacted size.
		m_tempBuffer.resize(allocator, ns::align_up(NS_MAX(accelBufferSizes.tempSizeInBytes, accelBufferSizes.tempUpdateSizeInBytes), alignof(uint64_t)) + sizeof(uint64_t));

		if (buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION)
		{
			m_outputBuffer.resize(allocator, accelBufferSizes.outputSizeInBytes);

			OptixAccelEmitDesc			emittedProp = {};
			OptixTraversableHandle		outputHandle = 0;
			emittedProp.type			= OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
			emittedProp.result			= CUdeviceptr(m_tempBuffer.data() + m_tempBuffer.size() - sizeof(uint64_t));

			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), (CUdeviceptr)m_outputBuffer.data(),
								  m_outputBuffer.bytes(), &outputHandle, &emittedProp, 1);

			if (err == OPTIX_SUCCESS)
			{
				uint64_t compactedSize = 0;

				stream.memcpy<uint64_t>(&compactedSize, (const uint64_t*)emittedProp.result, 1).sync();

				//!	First \p headerSize bytes for storing user data.
				m_compactedBuffer.resize(allocator, headerSize + compactedSize);

				err = optixAccelCompact(m_deviceContext->handle(), stream.handle(), outputHandle,
										CUdeviceptr(m_compactedBuffer.data() + headerSize), m_compactedBuffer.bytes() - headerSize, &m_hTraversable);
			}
		}
		else
		{
			//!	First \p headerSize bytes for storing user data.
			m_outputBuffer.resize(allocator, headerSize + accelBufferSizes.outputSizeInBytes);

			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), CUdeviceptr(m_outputBuffer.data() + headerSize),
								  m_outputBuffer.bytes() - headerSize, &m_hTraversable, nullptr, 0);
		}
	}

	if (err != OPTIX_SUCCESS)
	{
		NS_ERROR_LOG("Failed to build acceleration structure: %s.", optixGetErrorString(err));

		throw err;
	}

	m_buildOptions = buildOptions;
	m_headerSize = headerSize;
}


void AccelStruct::rebuild(ns::Stream & stream)
{
	if (m_hTraversable != 0)
	{
		OptixResult err = OPTIX_SUCCESS;

		OptixTraversableHandle outputHandle = 0;

		m_buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		if (this->allowCompaction())
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, m_cachedBuildInputs.data(), (uint32_t)m_cachedBuildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), (CUdeviceptr)m_outputBuffer.data(), m_outputBuffer.bytes(), &outputHandle, nullptr, 0);

			if (err == OPTIX_SUCCESS)
			{
				err = optixAccelCompact(m_deviceContext->handle(), stream.handle(), outputHandle,
										CUdeviceptr(m_compactedBuffer.data() + m_headerSize), m_compactedBuffer.bytes() - m_headerSize, &m_hTraversable);
			}
		}
		else
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, m_cachedBuildInputs.data(), (uint32_t)m_cachedBuildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), CUdeviceptr(m_outputBuffer.data() + m_headerSize),
								  m_outputBuffer.bytes() - m_headerSize, &outputHandle, nullptr, 0);

			m_hTraversable = outputHandle;
		}

		if (err != OPTIX_SUCCESS)
		{
			NS_ERROR_LOG("Failed to rebuild acceleration structure: %s.", optixGetErrorString(err));

			throw err;
		}
	}
}


void AccelStruct::refit(ns::Stream & stream)
{
	OptixResult err = OPTIX_SUCCESS;

	if (this->allowUpdate() && (m_hTraversable != 0))
	{
		m_buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

		if (this->allowCompaction())
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, m_cachedBuildInputs.data(), (uint32_t)m_cachedBuildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), CUdeviceptr(m_compactedBuffer.data() + m_headerSize),
								  m_compactedBuffer.bytes() - m_headerSize, &m_hTraversable, nullptr, 0);
		}
		else
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, m_cachedBuildInputs.data(), (uint32_t)m_cachedBuildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), CUdeviceptr(m_outputBuffer.data() + m_headerSize),
								  m_outputBuffer.bytes() - m_headerSize, &m_hTraversable, nullptr, 0);
		}

		if (err != OPTIX_SUCCESS)
		{
			NS_ERROR_LOG("Failed to refit acceleration structure: %s.", optixGetErrorString(err));

			throw err;
		}
	}
	else if (!this->allowUpdate())
	{
		NS_WARNING_LOG("Acceleration structure is non-updatable!");
	}
}


AccelStruct::~AccelStruct()
{

}

/*********************************************************************************
****************************    AccelStructTriangle    ****************************
*********************************************************************************/

AccelStructTriangle::AccelStructTriangle(std::shared_ptr<DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext))
{

}


void AccelStructTriangle::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	m_numSbtRecords = 0;
	m_buildInputs.resize(buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
		m_numSbtRecords += buildInputs[i].numSbtRecords;
		m_buildInputs[i] = buildInputs[i];
	}

	OptixAccelBuildOptions						buildOptions = {};
	buildOptions.operation						= OPTIX_BUILD_OPERATION_BUILD;
	buildOptions.buildFlags						= OPTIX_BUILD_FLAG_NONE;
//	buildOptions.buildFlags						|= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.buildFlags						|= preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
	buildOptions.buildFlags						|= allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0;
	buildOptions.motionOptions.numKeys			= 0;
	buildOptions.motionOptions.timeBegin		= 0.0f;
	buildOptions.motionOptions.timeEnd			= 0.0f;
	buildOptions.motionOptions.flags			= OPTIX_MOTION_FLAG_NONE;

	m_cachedBuildInputs.resize(m_buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
		m_cachedBuildInputs[i].type				= OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		m_cachedBuildInputs[i].triangleArray	= m_buildInputs[i];
	}

	AccelStruct::buildBase(stream, allocator, m_cachedBuildInputs, buildOptions, headerSize);
}


/*********************************************************************************
******************************    AccelStructAabb    *****************************
*********************************************************************************/

AccelStructAabb::AccelStructAabb(std::shared_ptr<DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext))
{

}


void AccelStructAabb::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	m_numSbtRecords = 0;
	m_buildInputs.resize(buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
		m_numSbtRecords += buildInputs[i].numSbtRecords;
		m_buildInputs[i] = buildInputs[i];
	}

	OptixAccelBuildOptions						buildOptions = {};
	buildOptions.operation						= OPTIX_BUILD_OPERATION_BUILD;
	buildOptions.buildFlags						= OPTIX_BUILD_FLAG_NONE;
//	buildOptions.buildFlags						|= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.buildFlags						|= preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
	buildOptions.buildFlags						|= allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0;
	buildOptions.motionOptions.numKeys			= 0;
	buildOptions.motionOptions.timeBegin		= 0.0f;
	buildOptions.motionOptions.timeEnd			= 0.0f;
	buildOptions.motionOptions.flags			= OPTIX_MOTION_FLAG_NONE;

	m_cachedBuildInputs.resize(m_buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
		m_cachedBuildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	#if OPTIX_VERSION >= 70100
		m_cachedBuildInputs[i].customPrimitiveArray		= m_buildInputs[i];
	#else
		m_cachedBuildInputs[i].aabbArray				= m_buildInputs[i];
	#endif
	}

	AccelStruct::buildBase(stream, allocator, m_cachedBuildInputs, buildOptions, headerSize);
}


/*********************************************************************************
*****************************    AccelStructCurve    *****************************
*********************************************************************************/

#if OPTIX_VERSION >= 70100

AccelStructCurve::AccelStructCurve(std::shared_ptr<DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext))
{

}


void AccelStructCurve::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	m_buildInputs.resize(buildInputs.size());
	m_numSbtRecords = static_cast<uint32_t>(buildInputs.size());

	for (size_t i = 0; i < buildInputs.size(); i++)
	{
		m_buildInputs[i] = buildInputs[i];
	}

	OptixAccelBuildOptions						buildOptions = {};
	buildOptions.operation						= OPTIX_BUILD_OPERATION_BUILD;
	buildOptions.buildFlags						= OPTIX_BUILD_FLAG_NONE;
//	buildOptions.buildFlags						|= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.buildFlags						|= preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
	buildOptions.buildFlags						|= allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0;
	buildOptions.motionOptions.numKeys			= 0;
	buildOptions.motionOptions.timeBegin		= 0.0f;
	buildOptions.motionOptions.timeEnd			= 0.0f;
	buildOptions.motionOptions.flags			= OPTIX_MOTION_FLAG_NONE;

	m_cachedBuildInputs.resize(m_buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
		m_cachedBuildInputs[i].type			= OPTIX_BUILD_INPUT_TYPE_CURVES;
		m_cachedBuildInputs[i].curveArray	= m_buildInputs[i];
	}

	AccelStruct::buildBase(stream, allocator, m_cachedBuildInputs, buildOptions, headerSize);
}


#endif	//	OPTIX_VERSION >= 70100

/*********************************************************************************
****************************    AccelStructSphere    *****************************
*********************************************************************************/

#if OPTIX_VERSION >= 70500

AccelStructSphere::AccelStructSphere(std::shared_ptr<DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext))
{

}


void AccelStructSphere::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	m_numSbtRecords = 0;
	m_buildInputs.resize(buildInputs.size());

	for (size_t i = 0; i < buildInputs.size(); i++)
	{
		m_numSbtRecords += buildInputs[i].numSbtRecords;
		m_buildInputs[i] = buildInputs[i];
	}

	OptixAccelBuildOptions						buildOptions = {};
	buildOptions.operation						= OPTIX_BUILD_OPERATION_BUILD;
	buildOptions.buildFlags						= OPTIX_BUILD_FLAG_NONE;
//	buildOptions.buildFlags						|= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.buildFlags						|= preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
	buildOptions.buildFlags						|= allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0;
	buildOptions.motionOptions.numKeys			= 0;
	buildOptions.motionOptions.timeBegin		= 0.0f;
	buildOptions.motionOptions.timeEnd			= 0.0f;
	buildOptions.motionOptions.flags			= OPTIX_MOTION_FLAG_NONE;

	m_cachedBuildInputs.resize(m_buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
		m_cachedBuildInputs[i].type				= OPTIX_BUILD_INPUT_TYPE_SPHERES;
		m_cachedBuildInputs[i].sphereArray		= m_buildInputs[i];
	}

	AccelStruct::buildBase(stream, allocator, m_cachedBuildInputs, buildOptions, headerSize);
}


#endif	//	OPTIX_VERSION >= 70500

/*********************************************************************************
******************************    InstAccelStruct    *****************************
*********************************************************************************/

InstAccelStruct::InstAccelStruct(std::shared_ptr<DeviceContext> deviceContext) : AccelStruct(std::move(deviceContext))
{

}


void InstAccelStruct::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixInstance> buildInputs, bool preferFastTrace, bool allowUpdate)
{
	m_hostInstances.resize(buildInputs.size());
	m_instances.resize(allocator, buildInputs.size());

	for (size_t i = 0; i < buildInputs.size(); i++)
	{
		m_hostInstances[i] = buildInputs[i];
	}

	stream.memcpy(m_instances.data(), m_hostInstances.data(), m_hostInstances.size());

	OptixAccelBuildOptions								buildOptions = {};
	buildOptions.operation								= OPTIX_BUILD_OPERATION_BUILD;
	buildOptions.buildFlags								= OPTIX_BUILD_FLAG_NONE;
//	buildOptions.buildFlags								|= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	buildOptions.buildFlags								|= preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
	buildOptions.buildFlags								|= allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0;
	buildOptions.motionOptions.numKeys					= 0;
	buildOptions.motionOptions.timeBegin				= 0.0f;
	buildOptions.motionOptions.timeEnd					= 0.0f;
	buildOptions.motionOptions.flags					= OPTIX_MOTION_FLAG_NONE;

	m_cachedBuildInputs.resize(1);

	m_cachedBuildInputs[0]												= {};
	m_cachedBuildInputs[0].type											= OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	m_cachedBuildInputs[0].instanceArray.instances						= (CUdeviceptr)m_instances.data();
#if OPTIX_VERSION >= 70600
	m_cachedBuildInputs[0].instanceArray.instanceStride					= sizeof(OptixInstance);
#endif
	m_cachedBuildInputs[0].instanceArray.numInstances					= static_cast<uint32_t>(m_instances.size());

	AccelStruct::buildBase(stream, allocator, m_cachedBuildInputs, buildOptions, 0);
}


void InstAccelStruct::rebuild(ns::Stream & stream)
{
	stream.memcpy(m_instances.data(), m_hostInstances.data(), m_hostInstances.size());

	AccelStruct::rebuild(stream);
}


void InstAccelStruct::refit(ns::Stream & stream)
{
	stream.memcpy(m_instances.data(), m_hostInstances.data(), m_hostInstances.size());

	AccelStruct::refit(stream);
}