/**
 *Copyright (c) 2025 Wenchao Huang <physhuangwenchao@gmail.com>
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in all
 *copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *SOFTWARE.
 */

#include "accel_struct_builder_impl.h"
#include "device_context.h"
#include <nucleus/launch_utils.cuh>
#include <optix_stubs.h>

PHOTON_USING_NAMESPACE

void AccelStructBuilder::build(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, AccelStruct::BuildOptions buildOptions)
{
OptixAccelBufferSizes accelBufferSizes = {};
OptixAccelBuildOptions optixBuildOptions = {};

optixBuildOptions.buildFlags = buildOptions.buildFlags;
optixBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
optixBuildOptions.motionOptions = buildOptions.motionOptions;

OptixResult err = optixAccelComputeMemoryUsage(m_state.deviceContext->handle(), &optixBuildOptions, buildInputs.data(), (uint32_t)buildInputs.size(), &accelBufferSizes);

if (err == OPTIX_SUCCESS)
{
buildOptions.headerSize = ns::align_up(buildOptions.headerSize, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);

//! Last aligned 8-bytes for storing compacted size.
m_state.tempBuffer.resize(allocator, ns::align_up(NS_MAX(accelBufferSizes.tempSizeInBytes, accelBufferSizes.tempUpdateSizeInBytes), alignof(uint64_t)) + sizeof(uint64_t));

if (buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION)
{
m_state.outputBuffer.resize(allocator, accelBufferSizes.outputSizeInBytes);

OptixAccelEmitDesc emittedProp = {};
OptixTraversableHandle outputHandle = 0;
emittedProp.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
emittedProp.result = CUdeviceptr(m_state.tempBuffer.data() + m_state.tempBuffer.size() - sizeof(uint64_t));

err = optixAccelBuild(m_state.deviceContext->handle(), stream.handle(), &optixBuildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
(CUdeviceptr)m_state.tempBuffer.data(), m_state.tempBuffer.bytes(), (CUdeviceptr)m_state.outputBuffer.data(),
m_state.outputBuffer.bytes(), &outputHandle, &emittedProp, 1);

if (err == OPTIX_SUCCESS)
{
uint64_t compactedSize = 0;

stream.memcpy<uint64_t>(&compactedSize, (const uint64_t*)emittedProp.result, 1).sync();

//! First \p headerSize bytes for storing user data.
m_state.compactedBuffer.resize(allocator, buildOptions.headerSize + compactedSize);

err = optixAccelCompact(m_state.deviceContext->handle(), stream.handle(), outputHandle, CUdeviceptr(m_state.compactedBuffer.data() + buildOptions.headerSize),
m_state.compactedBuffer.bytes() - buildOptions.headerSize, &m_state.traversable);
}
}
else
{
//! First \p headerSize bytes for storing user data.
m_state.outputBuffer.resize(allocator, buildOptions.headerSize + accelBufferSizes.outputSizeInBytes);

err = optixAccelBuild(m_state.deviceContext->handle(), stream.handle(), &optixBuildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
(CUdeviceptr)m_state.tempBuffer.data(), m_state.tempBuffer.bytes(), CUdeviceptr(m_state.outputBuffer.data() + buildOptions.headerSize),
m_state.outputBuffer.bytes() - buildOptions.headerSize, &m_state.traversable, nullptr, 0);
}
}

if (err != OPTIX_SUCCESS)
{
NS_ERROR_LOG("Failed to build acceleration structure: %s.", optixGetErrorString(err));

throw err;
}

m_state.headerSize = buildOptions.headerSize;
m_state.cachedBuildOptions = optixBuildOptions;
}


void AccelStructBuilder::rebuild(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs)
{
if (m_state.traversable != 0)
{
OptixResult err = OPTIX_SUCCESS;
OptixTraversableHandle outputHandle = 0;

m_state.cachedBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

if (this->allowCompaction())
{
err = optixAccelBuild(m_state.deviceContext->handle(), stream.handle(), &m_state.cachedBuildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
(CUdeviceptr)m_state.tempBuffer.data(), m_state.tempBuffer.bytes(), (CUdeviceptr)m_state.outputBuffer.data(),
m_state.outputBuffer.bytes(), &outputHandle, nullptr, 0);

if (err == OPTIX_SUCCESS)
{
err = optixAccelCompact(m_state.deviceContext->handle(), stream.handle(), outputHandle, CUdeviceptr(m_state.compactedBuffer.data() + m_state.headerSize),
m_state.compactedBuffer.bytes() - m_state.headerSize, &m_state.traversable);
}
}
else
{
err = optixAccelBuild(m_state.deviceContext->handle(), stream.handle(), &m_state.cachedBuildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
(CUdeviceptr)m_state.tempBuffer.data(), m_state.tempBuffer.bytes(), CUdeviceptr(m_state.outputBuffer.data() + m_state.headerSize),
m_state.outputBuffer.bytes() - m_state.headerSize, &outputHandle, nullptr, 0);

m_state.traversable = outputHandle;
}

if (err != OPTIX_SUCCESS)
{
NS_ERROR_LOG("Failed to rebuild acceleration structure: %s.", optixGetErrorString(err));

throw err;
}
}
}


void AccelStructBuilder::refit(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs)
{
OptixResult err = OPTIX_SUCCESS;

if (this->allowUpdate() && (m_state.traversable != 0))
{
m_state.cachedBuildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

if (this->allowCompaction())
{
err = optixAccelBuild(m_state.deviceContext->handle(), stream.handle(), &m_state.cachedBuildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
(CUdeviceptr)m_state.tempBuffer.data(), m_state.tempBuffer.bytes(), CUdeviceptr(m_state.compactedBuffer.data() + m_state.headerSize),
m_state.compactedBuffer.bytes() - m_state.headerSize, &m_state.traversable, nullptr, 0);
}
else
{
err = optixAccelBuild(m_state.deviceContext->handle(), stream.handle(), &m_state.cachedBuildOptions, buildInputs.data(), (uint32_t)buildInputs.size(),
(CUdeviceptr)m_state.tempBuffer.data(), m_state.tempBuffer.bytes(), CUdeviceptr(m_state.outputBuffer.data() + m_state.headerSize),
m_state.outputBuffer.bytes() - m_state.headerSize, &m_state.traversable, nullptr, 0);
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
