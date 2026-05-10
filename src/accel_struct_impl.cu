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
#include <stdexcept>

PHOTON_USING_NAMESPACE

namespace
{
	OptixAccelBuildOptions toOptixBuildOptions(const AccelBuildOptions & buildOptions)
	{
		OptixAccelBuildOptions optixBuildOptions = {};
		optixBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
		optixBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
		optixBuildOptions.buildFlags |= buildOptions.preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
		optixBuildOptions.buildFlags |= buildOptions.allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0;
		optixBuildOptions.buildFlags |= buildOptions.allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0;
		optixBuildOptions.motionOptions = buildOptions.motionOptions;
		return optixBuildOptions;
	}

	void validateBuildOptions(AccelStruct::SubType subType, const AccelBuildOptions & buildOptions)
	{
		if ((subType == AccelStruct::Instance) && (buildOptions.headerSize != 0))
			throw std::invalid_argument("IAS build options cannot request a header buffer.");

		if (buildOptions.motionOptions.numKeys == 0)
		{
			if ((buildOptions.motionOptions.timeBegin != 0.0f)
			 || (buildOptions.motionOptions.timeEnd != 0.0f)
			 || (buildOptions.motionOptions.flags != OPTIX_MOTION_FLAG_NONE))
			{
				throw std::invalid_argument("Motion options must stay zeroed when numKeys is 0.");
			}
		}
	}

	template<typename T>
	void assignBuildSources(std::vector<T> & dst, ns::ArrayProxy<T> src)
	{
		dst.assign(src.data(), src.data() + src.size());
	}

	template<typename TBuildInput, typename TSetter>
	void prepareBuildInputs(std::vector<OptixBuildInput> & dst, const std::vector<TBuildInput> & src, OptixBuildInputType type, TSetter setter)
	{
		if (src.empty())
			throw std::invalid_argument("Acceleration structure build inputs cannot be empty.");

		dst.resize(src.size());

		for (size_t i = 0; i < src.size(); ++i)
		{
			dst[i] = {};
			dst[i].type = type;
			setter(dst[i], src[i]);
		}
	}

	bool isZeroTransform(const OptixInstance & instance)
	{
		for (float element : instance.transform)
		{
			if (element != 0.0f)
				return false;
		}

		return true;
	}

	void validateBuildInputs(ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs)
	{
		if (buildInputs.empty())
			throw std::invalid_argument("Triangle build inputs cannot be empty.");

		for (const auto & buildInput : buildInputs)
		{
			if ((buildInput.numVertices == 0) || (buildInput.vertexBuffers == nullptr))
				throw std::invalid_argument("Triangle build inputs require vertex buffers and a non-zero vertex count.");

			if ((buildInput.numIndexTriplets != 0) && (buildInput.indexBuffer == 0))
				throw std::invalid_argument("Indexed triangle build inputs require an index buffer.");

			if ((buildInput.numSbtRecords == 0) || (buildInput.flags == nullptr))
				throw std::invalid_argument("Triangle build inputs require SBT flags and at least one record.");
		}
	}

	void validateBuildInputs(ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs)
	{
		if (buildInputs.empty())
			throw std::invalid_argument("Custom primitive build inputs cannot be empty.");

		for (const auto & buildInput : buildInputs)
		{
			if ((buildInput.numPrimitives == 0) || (buildInput.aabbBuffers == nullptr))
				throw std::invalid_argument("Custom primitive build inputs require AABB buffers and a non-zero primitive count.");

			if ((buildInput.numSbtRecords == 0) || (buildInput.flags == nullptr))
				throw std::invalid_argument("Custom primitive build inputs require SBT flags and at least one record.");
		}
	}

#if OPTIX_VERSION >= 70100
	void validateBuildInputs(ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs)
	{
		if (buildInputs.empty())
			throw std::invalid_argument("Curve build inputs cannot be empty.");

		for (const auto & buildInput : buildInputs)
		{
			if ((buildInput.numPrimitives == 0) || (buildInput.vertexBuffers == nullptr) || (buildInput.widthBuffers == nullptr))
				throw std::invalid_argument("Curve build inputs require vertex and width buffers and a non-zero primitive count.");

			if ((buildInput.numSbtRecords == 0) || (buildInput.flags == nullptr))
				throw std::invalid_argument("Curve build inputs require SBT flags and at least one record.");
		}
	}
#endif

#if OPTIX_VERSION >= 70500
	void validateBuildInputs(ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs)
	{
		if (buildInputs.empty())
			throw std::invalid_argument("Sphere build inputs cannot be empty.");

		for (const auto & buildInput : buildInputs)
		{
			if ((buildInput.numVertices == 0) || (buildInput.vertexBuffers == nullptr) || (buildInput.radiusBuffers == nullptr))
				throw std::invalid_argument("Sphere build inputs require center and radius buffers and a non-zero sphere count.");

			if ((buildInput.numSbtRecords == 0) || (buildInput.flags == nullptr))
				throw std::invalid_argument("Sphere build inputs require SBT flags and at least one record.");
		}
	}
#endif

	void validateBuildInputs(ns::ArrayProxy<OptixInstance> buildInputs)
	{
		if (buildInputs.empty())
			throw std::invalid_argument("Instance build inputs cannot be empty.");

		for (const auto & buildInput : buildInputs)
		{
			if (buildInput.traversableHandle == 0)
				throw std::invalid_argument("Instance build inputs require valid traversable handles.");

			if (isZeroTransform(buildInput))
				throw std::invalid_argument("Instance build inputs require a valid 3x4 affine transform.");
		}
	}

	void validateRefitInputs(const std::vector<OptixBuildInputTriangleArray> & current, ns::ArrayProxy<OptixBuildInputTriangleArray> next)
	{
		if (current.size() != next.size())
			throw std::invalid_argument("Triangle refit cannot change the number of build inputs.");

		for (size_t i = 0; i < current.size(); ++i)
		{
			if ((current[i].numVertices != next[i].numVertices)
			 || (current[i].numIndexTriplets != next[i].numIndexTriplets)
			 || (current[i].numSbtRecords != next[i].numSbtRecords))
			{
				throw std::invalid_argument("Triangle refit cannot change topology or SBT layout.");
			}
		}
	}

	void validateRefitInputs(const std::vector<OptixBuildInputCustomPrimitiveArray> & current, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> next)
	{
		if (current.size() != next.size())
			throw std::invalid_argument("Custom primitive refit cannot change the number of build inputs.");

		for (size_t i = 0; i < current.size(); ++i)
		{
			if ((current[i].numPrimitives != next[i].numPrimitives)
			 || (current[i].numSbtRecords != next[i].numSbtRecords))
			{
				throw std::invalid_argument("Custom primitive refit cannot change primitive counts or SBT layout.");
			}
		}
	}

#if OPTIX_VERSION >= 70100
	void validateRefitInputs(const std::vector<OptixBuildInputCurveArray> & current, ns::ArrayProxy<OptixBuildInputCurveArray> next)
	{
		if (current.size() != next.size())
			throw std::invalid_argument("Curve refit cannot change the number of build inputs.");

		for (size_t i = 0; i < current.size(); ++i)
		{
			if ((current[i].numPrimitives != next[i].numPrimitives)
			 || (current[i].numSbtRecords != next[i].numSbtRecords))
			{
				throw std::invalid_argument("Curve refit cannot change primitive counts or SBT layout.");
			}
		}
	}
#endif

#if OPTIX_VERSION >= 70500
	void validateRefitInputs(const std::vector<OptixBuildInputSphereArray> & current, ns::ArrayProxy<OptixBuildInputSphereArray> next)
	{
		if (current.size() != next.size())
			throw std::invalid_argument("Sphere refit cannot change the number of build inputs.");

		for (size_t i = 0; i < current.size(); ++i)
		{
			if ((current[i].numVertices != next[i].numVertices)
			 || (current[i].numSbtRecords != next[i].numSbtRecords))
			{
				throw std::invalid_argument("Sphere refit cannot change sphere counts or SBT layout.");
			}
		}
	}
#endif

	void validateRefitInputs(const std::vector<OptixInstance> & current, ns::ArrayProxy<OptixInstance> next)
	{
		if (current.size() != next.size())
			throw std::invalid_argument("Instance refit cannot change the number of instances.");
	}
}

/*********************************************************************************
********************************    AccelStruct    ********************************
*********************************************************************************/

AccelStruct::AccelStruct(std::shared_ptr<DeviceContext> deviceContext) : m_deviceContext(deviceContext), m_hTraversable(0)
{
	m_optixBuildOptions = OptixAccelBuildOptions{};
}


void AccelStruct::buildPrepared(ns::Stream & stream, ns::AllocPtr allocator, const AccelBuildOptions & buildOptions)
{
	validateBuildOptions(this->subType(), buildOptions);
	this->prepareBuildInputs(stream);
	this->buildBase(stream, allocator, m_cachedBuildInputs, toOptixBuildOptions(buildOptions), buildOptions.headerSize);
}


void AccelStruct::buildBase(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, OptixAccelBuildOptions buildOptions, size_t headerSize)
{
	OptixAccelBufferSizes accelBufferSizes = {};
	const size_t requestedHeaderSize = headerSize;

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

	m_optixBuildOptions = buildOptions;
	m_buildDesc.headerSize = requestedHeaderSize;
	m_buildDesc.preferFastTrace = (buildOptions.buildFlags & OPTIX_BUILD_FLAG_PREFER_FAST_TRACE) != 0;
	m_buildDesc.allowUpdate = (buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;
	m_buildDesc.allowCompaction = (buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
	m_buildDesc.motionOptions = buildOptions.motionOptions;
	m_bufferLayout.headerSize = requestedHeaderSize;
	m_bufferLayout.accelBufferOffset = headerSize;
}


void AccelStruct::rebuild(ns::Stream & stream)
{
	if (m_hTraversable != 0)
	{
		OptixResult err = OPTIX_SUCCESS;

		OptixTraversableHandle outputHandle = 0;

		this->prepareBuildInputs(stream);
		m_optixBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		if (this->allowCompaction())
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_optixBuildOptions, m_cachedBuildInputs.data(), (uint32_t)m_cachedBuildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), (CUdeviceptr)m_outputBuffer.data(), m_outputBuffer.bytes(), &outputHandle, nullptr, 0);

			if (err == OPTIX_SUCCESS)
			{
				err = optixAccelCompact(m_deviceContext->handle(), stream.handle(), outputHandle,
										CUdeviceptr(m_compactedBuffer.data() + m_bufferLayout.accelBufferOffset), m_compactedBuffer.bytes() - m_bufferLayout.accelBufferOffset, &m_hTraversable);
			}
		}
		else
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_optixBuildOptions, m_cachedBuildInputs.data(), (uint32_t)m_cachedBuildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), CUdeviceptr(m_outputBuffer.data() + m_bufferLayout.accelBufferOffset),
								  m_outputBuffer.bytes() - m_bufferLayout.accelBufferOffset, &outputHandle, nullptr, 0);

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
		this->prepareBuildInputs(stream);
		m_optixBuildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

		if (this->allowCompaction())
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_optixBuildOptions, m_cachedBuildInputs.data(), (uint32_t)m_cachedBuildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), CUdeviceptr(m_compactedBuffer.data() + m_bufferLayout.accelBufferOffset),
								  m_compactedBuffer.bytes() - m_bufferLayout.accelBufferOffset, &m_hTraversable, nullptr, 0);
		}
		else
		{
			err = optixAccelBuild(m_deviceContext->handle(), stream.handle(), &m_optixBuildOptions, m_cachedBuildInputs.data(), (uint32_t)m_cachedBuildInputs.size(),
								  (CUdeviceptr)m_tempBuffer.data(), m_tempBuffer.bytes(), CUdeviceptr(m_outputBuffer.data() + m_bufferLayout.accelBufferOffset),
								  m_outputBuffer.bytes() - m_bufferLayout.accelBufferOffset, &m_hTraversable, nullptr, 0);
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


void AccelStructTriangle::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs, const AccelBuildOptions & buildOptions)
{
	validateBuildInputs(buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::buildPrepared(stream, allocator, buildOptions);
}


void AccelStructTriangle::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	AccelBuildOptions buildOptions = {};
	buildOptions.headerSize = headerSize;
	buildOptions.preferFastTrace = preferFastTrace;
	buildOptions.allowUpdate = allowUpdate;
	this->build(stream, allocator, buildInputs, buildOptions);
}


void AccelStructTriangle::rebuild(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs)
{
	validateBuildInputs(buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::buildPrepared(stream, allocator, this->buildOptions());
}


void AccelStructTriangle::refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs)
{
	validateBuildInputs(buildInputs);
	validateRefitInputs(m_buildSources, buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::refit(stream);
}


void AccelStructTriangle::prepareBuildInputs(ns::Stream & stream)
{
	(void)stream;
	prepareBuildInputs(m_cachedBuildInputs, m_buildSources, OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
					   [](OptixBuildInput & buildInput, const OptixBuildInputTriangleArray & source) { buildInput.triangleArray = source; });
}


/*********************************************************************************
******************************    AccelStructAabb    *****************************
*********************************************************************************/

AccelStructAabb::AccelStructAabb(std::shared_ptr<DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext))
{

}


void AccelStructAabb::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs, const AccelBuildOptions & buildOptions)
{
	validateBuildInputs(buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::buildPrepared(stream, allocator, buildOptions);
}


void AccelStructAabb::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	AccelBuildOptions buildOptions = {};
	buildOptions.headerSize = headerSize;
	buildOptions.preferFastTrace = preferFastTrace;
	buildOptions.allowUpdate = allowUpdate;
	this->build(stream, allocator, buildInputs, buildOptions);
}


void AccelStructAabb::rebuild(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs)
{
	validateBuildInputs(buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::buildPrepared(stream, allocator, this->buildOptions());
}


void AccelStructAabb::refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs)
{
	validateBuildInputs(buildInputs);
	validateRefitInputs(m_buildSources, buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::refit(stream);
}


void AccelStructAabb::prepareBuildInputs(ns::Stream & stream)
{
	(void)stream;
	prepareBuildInputs(m_cachedBuildInputs, m_buildSources, OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
					   [](OptixBuildInput & buildInput, const OptixBuildInputCustomPrimitiveArray & source)
					   {
						#if OPTIX_VERSION >= 70100
							buildInput.customPrimitiveArray = source;
						#else
							buildInput.aabbArray = source;
						#endif
					   });
}

/*********************************************************************************
*****************************    AccelStructCurve    *****************************
*********************************************************************************/

#if OPTIX_VERSION >= 70100

AccelStructCurve::AccelStructCurve(std::shared_ptr<DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext))
{

}


void AccelStructCurve::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs, const AccelBuildOptions & buildOptions)
{
	validateBuildInputs(buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::buildPrepared(stream, allocator, buildOptions);
}


void AccelStructCurve::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	AccelBuildOptions buildOptions = {};
	buildOptions.headerSize = headerSize;
	buildOptions.preferFastTrace = preferFastTrace;
	buildOptions.allowUpdate = allowUpdate;
	this->build(stream, allocator, buildInputs, buildOptions);
}


void AccelStructCurve::rebuild(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs)
{
	validateBuildInputs(buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::buildPrepared(stream, allocator, this->buildOptions());
}


void AccelStructCurve::refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs)
{
	validateBuildInputs(buildInputs);
	validateRefitInputs(m_buildSources, buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::refit(stream);
}


void AccelStructCurve::prepareBuildInputs(ns::Stream & stream)
{
	(void)stream;
	prepareBuildInputs(m_cachedBuildInputs, m_buildSources, OPTIX_BUILD_INPUT_TYPE_CURVES,
					   [](OptixBuildInput & buildInput, const OptixBuildInputCurveArray & source) { buildInput.curveArray = source; });
}

#endif	//	OPTIX_VERSION >= 70100

/*********************************************************************************
****************************    AccelStructSphere    *****************************
*********************************************************************************/

#if OPTIX_VERSION >= 70500

AccelStructSphere::AccelStructSphere(std::shared_ptr<DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext))
{

}


void AccelStructSphere::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs, const AccelBuildOptions & buildOptions)
{
	validateBuildInputs(buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::buildPrepared(stream, allocator, buildOptions);
}


void AccelStructSphere::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate)
{
	AccelBuildOptions buildOptions = {};
	buildOptions.headerSize = headerSize;
	buildOptions.preferFastTrace = preferFastTrace;
	buildOptions.allowUpdate = allowUpdate;
	this->build(stream, allocator, buildInputs, buildOptions);
}


void AccelStructSphere::rebuild(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs)
{
	validateBuildInputs(buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::buildPrepared(stream, allocator, this->buildOptions());
}


void AccelStructSphere::refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs)
{
	validateBuildInputs(buildInputs);
	validateRefitInputs(m_buildSources, buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::refit(stream);
}


void AccelStructSphere::prepareBuildInputs(ns::Stream & stream)
{
	(void)stream;
	prepareBuildInputs(m_cachedBuildInputs, m_buildSources, OPTIX_BUILD_INPUT_TYPE_SPHERES,
					   [](OptixBuildInput & buildInput, const OptixBuildInputSphereArray & source) { buildInput.sphereArray = source; });
}

#endif	//	OPTIX_VERSION >= 70500

/*********************************************************************************
******************************    InstAccelStruct    *****************************
*********************************************************************************/

InstAccelStruct::InstAccelStruct(std::shared_ptr<DeviceContext> deviceContext) : AccelStruct(std::move(deviceContext))
{

}


void InstAccelStruct::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixInstance> buildInputs, const AccelBuildOptions & buildOptions)
{
	validateBuildInputs(buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	m_instanceAllocator = allocator;
	AccelStruct::buildPrepared(stream, allocator, buildOptions);
}


void InstAccelStruct::build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixInstance> buildInputs, bool preferFastTrace, bool allowUpdate)
{
	AccelBuildOptions buildOptions = {};
	buildOptions.preferFastTrace = preferFastTrace;
	buildOptions.allowUpdate = allowUpdate;
	this->build(stream, allocator, buildInputs, buildOptions);
}


void InstAccelStruct::rebuild(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixInstance> buildInputs)
{
	validateBuildInputs(buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	m_instanceAllocator = allocator;
	AccelStruct::buildPrepared(stream, allocator, this->buildOptions());
}


void InstAccelStruct::refit(ns::Stream & stream, ns::ArrayProxy<OptixInstance> buildInputs)
{
	validateBuildInputs(buildInputs);
	validateRefitInputs(m_buildSources, buildInputs);
	assignBuildSources(m_buildSources, buildInputs);
	AccelStruct::refit(stream);
}


void InstAccelStruct::prepareBuildInputs(ns::Stream & stream)
{
	if (m_buildSources.empty())
		throw std::invalid_argument("Instance build inputs cannot be empty.");

	m_instances.resize(m_instanceAllocator, m_buildSources.size());
	stream.memcpy(m_instances.data(), m_buildSources.data(), m_buildSources.size());

	m_cachedBuildInputs.resize(1);
	m_cachedBuildInputs[0] = {};
	m_cachedBuildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	m_cachedBuildInputs[0].instanceArray.instances = (CUdeviceptr)m_instances.data();
#if OPTIX_VERSION >= 70600
	m_cachedBuildInputs[0].instanceArray.instanceStride = sizeof(OptixInstance);
#endif
	m_cachedBuildInputs[0].instanceArray.numInstances = static_cast<uint32_t>(m_instances.size());
}


void InstAccelStruct::rebuild(ns::Stream & stream)
{
	AccelStruct::rebuild(stream);
}


void InstAccelStruct::refit(ns::Stream & stream)
{
	AccelStruct::refit(stream);
}
