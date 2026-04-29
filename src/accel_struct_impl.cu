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

#include "accel_struct_impl.h"
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
*****************************    AccelStructBase    ******************************
*********************************************************************************/

AccelStructBase::AccelStructBase(std::shared_ptr<DeviceContext> deviceContext) : m_deviceContext(deviceContext), m_hTraversable(0), m_numSbtRecords(0), m_headerSize(0)
{
	m_buildOptions = OptixAccelBuildOptions{};
}


void AccelStructBase::build(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, OptixAccelBuildOptions buildOptions, size_t headerSize)
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


void AccelStructBase::rebuild(ns::Stream & stream)
{
	if (m_hTraversable != 0)
	{
		OptixResult err = OPTIX_SUCCESS;

		OptixTraversableHandle outputHandle = 0;

		m_buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		const auto buildInputs = this->makeOptixBuildInputs();

		if (this->allowCompaction())
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), (CUdeviceptr)m_outputBuffer.data(), m_outputBuffer.bytes(), &outputHandle, nullptr, 0);

			if (err == OPTIX_SUCCESS)
			{
				err = optixAccelCompact(m_deviceContext->handle(), stream.handle(), outputHandle,
										CUdeviceptr(m_compactedBuffer.data() + m_headerSize), m_compactedBuffer.bytes() - m_headerSize, &m_hTraversable);
			}
		}
		else
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
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


void AccelStructBase::refit(ns::Stream & stream)
{
	OptixResult err = OPTIX_SUCCESS;

	if (this->allowUpdate() && (m_hTraversable != 0))
	{
		m_buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

		const auto buildInputs = this->makeOptixBuildInputs();

		if (this->allowCompaction())
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), CUdeviceptr(m_compactedBuffer.data() + m_headerSize),
								  m_compactedBuffer.bytes() - m_headerSize, &m_hTraversable, nullptr, 0);
		}
		else
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_buildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
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


AccelStructBase::~AccelStructBase()
{

}

/*********************************************************************************
*************************    AccelStructTriangleImpl    **************************
*********************************************************************************/

void AccelStructTriangleImpl::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	m_numSbtRecords = 0;
	m_geomFlags.resize(buildInputs.size());
	m_vertBuffers.resize(buildInputs.size());
	m_buildInputs.resize(buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
		const bool useInexBuffer = (buildInputs[i].indexBuffer != nullptr) && (buildInputs[i].numIndexTriplets > 0);

		if (buildInputs[i].perSbtRecordFlags.empty())
		{
			m_geomFlags[i].assign(buildInputs[i].numSbtRecords, OPTIX_GEOMETRY_FLAG_NONE);
		}
		else if (buildInputs[i].perSbtRecordFlags.size() == buildInputs[i].numSbtRecords)
		{
			m_geomFlags[i].resize(buildInputs[i].perSbtRecordFlags.size());

			std::memcpy(m_geomFlags[i].data(),buildInputs[i].perSbtRecordFlags.data(), sizeof(GeomFlags) * buildInputs[i].perSbtRecordFlags.size());
		}
		else
		{
			NS_ASSERT_LOG_IF(buildInputs[i].perSbtRecordFlags.size() != buildInputs[i].numSbtRecords, "Geometry flags does not match with numSbtRecords!");

			return;
		}

		m_vertBuffers[i]													= (CUdeviceptr)buildInputs[i].vertexBuffer.data();
		m_numSbtRecords														+= buildInputs[i].numSbtRecords;
		m_buildInputs[i]													= OptixBuildInputTriangleArray{};
		m_buildInputs[i].flags											= m_geomFlags[i].data();
		m_buildInputs[i].vertexFormat									= OPTIX_VERTEX_FORMAT_FLOAT3;
		m_buildInputs[i].vertexStrideInBytes							= sizeof(ns::float3_16a);
		m_buildInputs[i].vertexBuffers									= &m_vertBuffers[i];
		m_buildInputs[i].numVertices									= buildInputs[i].numVertices;
		m_buildInputs[i].indexBuffer									= useInexBuffer ? (CUdeviceptr)buildInputs[i].indexBuffer.data() : NULL;
		m_buildInputs[i].numIndexTriplets								= useInexBuffer ? buildInputs[i].numIndexTriplets : 0;
		m_buildInputs[i].indexStrideInBytes								= useInexBuffer ? sizeof(ns::int3_16a) : 0;
		m_buildInputs[i].preTransform									= NULL;
		m_buildInputs[i].numSbtRecords									= buildInputs[i].numSbtRecords;
		m_buildInputs[i].primitiveIndexOffset							= buildInputs[i].primitiveIndexOffset;
		m_buildInputs[i].sbtIndexOffsetBuffer							= (CUdeviceptr)buildInputs[i].sbtIndexOffsetBuffer.data();
		m_buildInputs[i].sbtIndexOffsetSizeInBytes						= sizeof(uint32_t);
		m_buildInputs[i].sbtIndexOffsetStrideInBytes					= sizeof(uint32_t);
	#if OPTIX_VERSION >= 70100
		m_buildInputs[i].indexFormat									= useInexBuffer ? OPTIX_INDICES_FORMAT_UNSIGNED_INT3 : OPTIX_INDICES_FORMAT_NONE;
		m_buildInputs[i].transformFormat								= OPTIX_TRANSFORM_FORMAT_NONE;
	#else
		m_buildInputs[i].indexFormat									= OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	#endif
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

	AccelStructBase::build(stream, allocator, makeOptixBuildInputs(), buildOptions, headerSize);
}


std::vector<OptixBuildInput> AccelStructTriangleImpl::makeOptixBuildInputs() const
{
	std::vector<OptixBuildInput> result(m_buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
		result[i].type				= OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		result[i].triangleArray		= m_buildInputs[i];
	}

	return result;
}

/*********************************************************************************
***************************    AccelStructAabbImpl    ****************************
*********************************************************************************/

void AccelStructAabbImpl::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	m_numSbtRecords = 0;
	m_geomFlags.resize(buildInputs.size());
	m_aabbBuffers.resize(buildInputs.size());
	m_buildInputs.resize(buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
		if (buildInputs[i].perSbtRecordFlags.empty())
		{
			m_geomFlags[i].assign(buildInputs[i].numSbtRecords, OPTIX_GEOMETRY_FLAG_NONE);
		}
		else if (buildInputs[i].perSbtRecordFlags.size() == buildInputs[i].numSbtRecords)
		{
			m_geomFlags[i].resize(buildInputs[i].perSbtRecordFlags.size());

			std::memcpy(m_geomFlags[i].data(),buildInputs[i].perSbtRecordFlags.data(), sizeof(GeomFlags) * buildInputs[i].perSbtRecordFlags.size());
		}
		else
		{
			NS_ASSERT_LOG_IF(buildInputs[i].perSbtRecordFlags.size() != buildInputs[i].numSbtRecords, "Geometry flags does not match with numSbtRecords!");

			return;
		}

		m_aabbBuffers[i]																= (CUdeviceptr)buildInputs[i].aabbBuffer.data();
		m_numSbtRecords																	+= buildInputs[i].numSbtRecords;
		m_buildInputs[i]																= OptixBuildInputCustomPrimitiveArray{};
		m_buildInputs[i].flags															= m_geomFlags[i].data();
		m_buildInputs[i].aabbBuffers													= &m_aabbBuffers[i];
		m_buildInputs[i].strideInBytes													= sizeof(Aabb);
		m_buildInputs[i].numPrimitives													= buildInputs[i].numPrimitives;
		m_buildInputs[i].numSbtRecords													= buildInputs[i].numSbtRecords;
		m_buildInputs[i].primitiveIndexOffset											= buildInputs[i].primitiveIndexOffset;
		m_buildInputs[i].sbtIndexOffsetBuffer											= (CUdeviceptr)buildInputs[i].sbtIndexOffsetBuffer.data();
		m_buildInputs[i].sbtIndexOffsetSizeInBytes										= sizeof(uint32_t);
		m_buildInputs[i].sbtIndexOffsetStrideInBytes									= sizeof(uint32_t);
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

	AccelStructBase::build(stream, allocator, makeOptixBuildInputs(), buildOptions, headerSize);
}


std::vector<OptixBuildInput> AccelStructAabbImpl::makeOptixBuildInputs() const
{
	std::vector<OptixBuildInput> result(m_buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
	#if OPTIX_VERSION >= 70100
		result[i].type						= OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		result[i].customPrimitiveArray		= m_buildInputs[i];
	#else
		result[i].type						= OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		result[i].aabbArray					= m_buildInputs[i];
	#endif
	}

	return result;
}

/*********************************************************************************
***************************    AccelStructCurveImpl    ***************************
*********************************************************************************/

void AccelStructCurveImpl::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	m_buildInputs.resize(buildInputs.size());
	m_vertBuffers.resize(buildInputs.size());
	m_widthBuffers.resize(buildInputs.size());
	m_numSbtRecords = static_cast<uint32_t>(buildInputs.size());

	for (size_t i = 0; i < buildInputs.size(); i++)
	{
		m_vertBuffers[i]										= (CUdeviceptr)buildInputs[i].vertexBuffer.data();
		m_widthBuffers[i]										= (CUdeviceptr)buildInputs[i].widthBuffer.data();

		m_buildInputs[i]										= OptixBuildInputCurveArray{};
	#if OPTIX_VERSION >= 70400
		m_buildInputs[i].endcapFlags							= OPTIX_CURVE_ENDCAP_DEFAULT;
	#endif
	#if OPTIX_VERSION >= 70100
		m_buildInputs[i].flag									= buildInputs[i].flags;
		m_buildInputs[i].curveType								= static_cast<OptixPrimitiveType>(buildInputs[i].curveType);
		m_buildInputs[i].numVertices							= buildInputs[i].numVertices;
		m_buildInputs[i].numPrimitives							= buildInputs[i].numPrimitives;
		m_buildInputs[i].primitiveIndexOffset					= buildInputs[i].primitiveIndexOffset;
		m_buildInputs[i].vertexBuffers							= &m_vertBuffers[i];
		m_buildInputs[i].vertexStrideInBytes					= sizeof(ns::float3_16a);
		m_buildInputs[i].indexBuffer							= (CUdeviceptr)buildInputs[i].indexBuffer.data();
		m_buildInputs[i].indexStrideInBytes						= sizeof(uint32_t);
		m_buildInputs[i].widthBuffers							= &m_widthBuffers[i];
		m_buildInputs[i].widthStrideInBytes						= sizeof(float);
		m_buildInputs[i].normalBuffers							= nullptr;
		m_buildInputs[i].normalStrideInBytes					= 0;
	#endif
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

	AccelStructBase::build(stream, allocator, makeOptixBuildInputs(), buildOptions, headerSize);
}


std::vector<OptixBuildInput> AccelStructCurveImpl::makeOptixBuildInputs() const
{
	std::vector<OptixBuildInput> result(m_buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
		result[i].type			= OPTIX_BUILD_INPUT_TYPE_CURVES;
		result[i].curveArray	= m_buildInputs[i];
	}

	return result;
}

/*********************************************************************************
**************************    AccelStructSphereImpl    ***************************
*********************************************************************************/

void AccelStructSphereImpl::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	m_numSbtRecords = 0;
	m_geomFlags.resize(buildInputs.size());
	m_buildInputs.resize(buildInputs.size());
	m_vertBuffers.resize(buildInputs.size());
	m_radiusBuffers.resize(buildInputs.size());

	for (size_t i = 0; i < buildInputs.size(); i++)
	{
		if (buildInputs[i].perSbtRecordFlags.empty())
		{
			m_geomFlags[i].assign(buildInputs[i].numSbtRecords, OPTIX_GEOMETRY_FLAG_NONE);
		}
		else if (buildInputs[i].perSbtRecordFlags.size() == buildInputs[i].numSbtRecords)
		{
			m_geomFlags[i].resize(buildInputs[i].perSbtRecordFlags.size());

			std::memcpy(m_geomFlags[i].data(),buildInputs[i].perSbtRecordFlags.data(), sizeof(GeomFlags) * buildInputs[i].perSbtRecordFlags.size());
		}
		else
		{
			NS_ASSERT_LOG_IF(buildInputs[i].perSbtRecordFlags.size() != buildInputs[i].numSbtRecords, "Geometry flags does not match with numSbtRecords!");

			return;
		}

		m_vertBuffers[i]											= (CUdeviceptr)buildInputs[i].vertexBuffer.data();
		m_radiusBuffers[i]											= (CUdeviceptr)buildInputs[i].radiusBuffer.data();
		m_numSbtRecords												+= buildInputs[i].numSbtRecords;
	#if OPTIX_VERSION >= 70500
		m_buildInputs[i]											= OptixBuildInputSphereArray{};
		m_buildInputs[i].flags										= m_geomFlags[i].data();
		m_buildInputs[i].numVertices								= buildInputs[i].numVertices;
		m_buildInputs[i].vertexBuffers								= &m_vertBuffers[i];
		m_buildInputs[i].vertexStrideInBytes						= sizeof(ns::float3_16a);
		m_buildInputs[i].radiusBuffers								= &m_radiusBuffers[i];
		m_buildInputs[i].radiusStrideInBytes						= sizeof(float);
		m_buildInputs[i].singleRadius								= buildInputs[i].singleRadius;
		m_buildInputs[i].numSbtRecords								= buildInputs[i].numSbtRecords;
		m_buildInputs[i].primitiveIndexOffset						= buildInputs[i].primitiveIndexOffset;
		m_buildInputs[i].sbtIndexOffsetBuffer						= (CUdeviceptr)buildInputs[i].sbtIndexOffsetBuffer.data();
		m_buildInputs[i].sbtIndexOffsetSizeInBytes					= sizeof(uint32_t);
		m_buildInputs[i].sbtIndexOffsetStrideInBytes				= sizeof(uint32_t);
	#endif
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

	AccelStructBase::build(stream, allocator, makeOptixBuildInputs(), buildOptions, headerSize);
}


std::vector<OptixBuildInput> AccelStructSphereImpl::makeOptixBuildInputs() const
{
	std::vector<OptixBuildInput> result(m_buildInputs.size());

	for (size_t i = 0; i < m_buildInputs.size(); i++)
	{
		result[i].type				= OPTIX_BUILD_INPUT_TYPE_SPHERES;
		result[i].sphereArray		= m_buildInputs[i];
	}

	return result;
}

/*********************************************************************************
***************************    InstAccelStructImpl    ****************************
*********************************************************************************/

namespace kernels
{
	__global__ void AssignInstanceTransforms(dev::Ptr<OptixInstance> pInstances, dev::Ptr<const dev::Ptr<const Mat4x4>> ppTransforms, unsigned int numInstances)
	{
		CUDA_for(i, numInstances);

		Mat4x4 transform = {};
		transform.rows[0] = ns::float4{ 1, 0, 0, 0 };
		transform.rows[1] = ns::float4{ 0, 1, 0, 0 };
		transform.rows[2] = ns::float4{ 0, 0, 1, 0 };
		transform.rows[3] = ns::float4{ 0, 0, 0, 1 };

		if (ppTransforms[i])
		{
			transform = *ppTransforms[i];
		}

		ns::float4 * pAddressBegin = reinterpret_cast<ns::float4*>(pInstances[i].transform);

		pAddressBegin[0] = transform.rows[0];
		pAddressBegin[1] = transform.rows[1];
		pAddressBegin[2] = transform.rows[2];
	}
}

void InstAccelStructImpl::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<BuildInput> buildInputs, bool preferFastTrace, bool allowUpdate)
{
	m_hostInstances.resize(buildInputs.size());
	m_geomStructs.resize(buildInputs.size());
	m_instances.resize(allocator, buildInputs.size());
	m_transforms.resize(allocator, buildInputs.size());

	std::vector<ns::dev::Ptr<const Mat4x4>>		pTransforms(buildInputs.size());

	for (size_t i = 0; i < buildInputs.size(); i++)
	{
		m_hostInstances[i]					= OptixInstance{};
		m_hostInstances[i].traversableHandle	= buildInputs[i].geomAccelStruct->handle();
		m_hostInstances[i].visibilityMask		= buildInputs[i].visibilityMask;
		m_hostInstances[i].instanceId			= buildInputs[i].instanceId;
		m_hostInstances[i].sbtOffset			= buildInputs[i].sbtOffset;
		m_hostInstances[i].flags				= buildInputs[i].flags;
		m_geomStructs[i]					= buildInputs[i].geomAccelStruct;
		pTransforms[i]						= buildInputs[i].transform;
	}

	stream.memcpy(m_instances.data(), m_hostInstances.data(), m_hostInstances.size());
	stream.memcpy(m_transforms.data(), pTransforms.data(), pTransforms.size());
	stream.launch(kernels::AssignInstanceTransforms, ns::ceil_div(m_instances.size(), 128), 128)(m_instances, m_transforms, static_cast<uint32_t>(m_instances.size()));

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

	AccelStructBase::build(stream, allocator, makeOptixBuildInputs(), buildOptions, 0);
}


void InstAccelStructImpl::rebuild(ns::Stream & stream)
{
	stream.launch(kernels::AssignInstanceTransforms, ns::ceil_div(m_instances.size(), 128), 128)(m_instances, m_transforms, static_cast<uint32_t>(m_instances.size()));

	AccelStructBase::rebuild(stream);
}


void InstAccelStructImpl::refit(ns::Stream & stream)
{
	stream.launch(kernels::AssignInstanceTransforms, ns::ceil_div(m_instances.size(), 128), 128)(m_instances, m_transforms, static_cast<uint32_t>(m_instances.size()));

	AccelStructBase::refit(stream);
}


std::vector<OptixBuildInput> InstAccelStructImpl::makeOptixBuildInputs() const
{
	OptixBuildInput		optixBuildInput					= {};
	optixBuildInput.type								= OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	optixBuildInput.instanceArray.instances				= (CUdeviceptr)m_instances.data();
#if OPTIX_VERSION >= 70600
	optixBuildInput.instanceArray.instanceStride		= sizeof(OptixInstance);
#endif
	optixBuildInput.instanceArray.numInstances			= static_cast<uint32_t>(m_instances.size());

	return { optixBuildInput };
}