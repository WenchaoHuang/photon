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
#pragma once

#include "pipeline.h"
#include <optix.h>

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	*****************************    ProgramType    ******************************
	*****************************************************************************/

	/**
	 *	@brief		Supported OptiX program types.
	 */
	enum ProgramType
	{
		Miss,						//	prefix "__miss__"
		AnyHit,						//	prefix "__anyhit__"
		Raygen,						//	prefix "__raygen__"
		Exception,					//	prefix "__exception__"
		ClosestHit,					//	prefix "__closesthit__"
		Intersection,				//	prefix "__intersection__"
		DirectCallable,				//	prefix "__direct_callable__"
		ContinuationCallable,		//	prefix "__continuation_callable__"
		BuiltinIntersection,
		CallableGroup,
		HitGroup,
		Unknow,
	};

	/*****************************************************************************
	******************************    ModuleImpl    ******************************
	*****************************************************************************/

	class ModuleImpl : public Module, public std::enable_shared_from_this<ModuleImpl>
	{

	public:

		ModuleImpl(std::shared_ptr<DeviceContext> deviceContext, OptixModule hModule);

		~ModuleImpl();

	public:

		OptixModule handle() const { return m_hModule; }

		virtual ProgramEntry entry(const std::string & funcName) override;

		std::shared_ptr<DeviceContext> deviceContext() const { return m_deviceContext; }

	private:

		const std::shared_ptr<DeviceContext>		m_deviceContext;
		const OptixModule							m_hModule;
	};

	/*****************************************************************************
	*****************************    ProgramImpl    ******************************
	*****************************************************************************/

	class ProgramImpl : public Program
	{

	public:

		ProgramImpl(std::shared_ptr<DeviceContext> context, OptixProgramGroup hProgramGroup, ProgramType type);

		~ProgramImpl();

	public:

		virtual const SbtHeader & header() const override { return m_header; }

		OptixProgramGroup handle() { return m_hProgramGroup; }

	private:

		const OptixProgramGroup						m_hProgramGroup;
		const std::shared_ptr<DeviceContext>		m_deviceContext;
		const ProgramType							m_progType;
		SbtHeader									m_header;
	};
}