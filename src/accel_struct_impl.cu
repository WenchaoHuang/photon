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

#include "accel_struct_builder_impl.h"
#include "accel_struct.h"

PHOTON_USING_NAMESPACE

/*********************************************************************************
*******************************    AccelStruct    ********************************
*********************************************************************************/

AccelStruct::AccelStruct(std::shared_ptr<DeviceContext> deviceContext) : m_deviceContext(deviceContext), m_hTraversable(0), m_headerSize(0)
{
	m_buildOptions = OptixAccelBuildOptions{};
}


void AccelStruct::build(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, BuildOptions buildOptions)
{
	AccelStructBuilder builder({ m_headerSize, m_tempBuffer, m_outputBuffer, m_compactedBuffer, m_hTraversable, m_buildOptions, m_deviceContext });
	builder.build(stream, allocator, buildInputs, buildOptions);
}


void AccelStruct::rebuild(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs)
{
	AccelStructBuilder builder({ m_headerSize, m_tempBuffer, m_outputBuffer, m_compactedBuffer, m_hTraversable, m_buildOptions, m_deviceContext });
	builder.rebuild(stream, buildInputs);
}


void AccelStruct::refit(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs)
{
	AccelStructBuilder builder({ m_headerSize, m_tempBuffer, m_outputBuffer, m_compactedBuffer, m_hTraversable, m_buildOptions, m_deviceContext });
	builder.refit(stream, buildInputs);
}