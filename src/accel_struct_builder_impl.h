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
#pragma once

#include "accel_struct.h"

namespace PHOTON_NAMESPACE
{
struct AccelStructBuildState
{
size_t & headerSize;
ns::Array<unsigned char> & tempBuffer;
ns::Array<unsigned char> & outputBuffer;
ns::Array<unsigned char> & compactedBuffer;
OptixTraversableHandle & traversable;
OptixAccelBuildOptions & cachedBuildOptions;
const std::shared_ptr<class DeviceContext> & deviceContext;
};

class AccelStructBuilder
{
public:

explicit AccelStructBuilder(AccelStructBuildState state) : m_state(state) {}

void build(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, AccelStruct::BuildOptions buildOptions);

void rebuild(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs);

void refit(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs);

private:

bool allowUpdate() const { return (m_state.cachedBuildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0; }

bool allowCompaction() const { return (m_state.cachedBuildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0; }

private:

AccelStructBuildState m_state;
};
}
