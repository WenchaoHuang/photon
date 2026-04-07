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

#include "fwd.h"
#include "sbt_record.h"
#include <nucleus/array_proxy.h>
#include <nucleus/device_pointer.h>
#include <optix.h>
#include <string>

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	********************************    Module    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Abstract interface for an OptiX module.
	 *	@note		Represents a compiled OptiX module that contains one or more
	 *				program entry points.
	 */
	class Module
	{

	public:

		//!	Virtual destructor.
		virtual ~Module() {}

		/**
		 *	@brief		Get a lightweight entry reference by function name.
		 *	@note		Does not create any OptiX resources. The returned ProgramEntry should
		 *				be passed to Program factory methods (raygen, miss, hitgroup, etc.).
		 *	@param[in]	funcName - The PTX function entry name (e.g. "__raygen__", "__closesthit__xxx").
		 *	@return		A valid ProgramEntry on success, or an invalid (empty) ProgramEntry on failure.
		 */
		virtual ProgramEntry entry(const std::string & funcName) = 0;
	};

	/*****************************************************************************
	*******************************    Program    ********************************
	*****************************************************************************/

	/**
	 *	@brief		Represents an OptiX program group.
	 *	@note		Each Program owns an OptixProgramGroup and its SBT header.
	 *				Create via static factory methods: raygen(), miss(), exception(), hitgroup(), callables().
	 */
	class Program
	{

	public:

		/**
		 *	@brief	Supported OptiX program types.
		 */
		enum Type
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

		//!	Virtual destructor.
		virtual ~Program() {}

		//!	Get the type of this program.
		virtual Type type() const = 0;

		//!	Get the SBT header for this program.
		virtual const SbtHeader & header() const = 0;

		//!	@brief		Create a miss program group.
		PHOTON_API static std::shared_ptr<Program> miss(const ProgramEntry & entry);

		//!	@brief		Create a raygen program group.
		PHOTON_API static std::shared_ptr<Program> raygen(const ProgramEntry & entry);

		//!	@brief		Create an exception program group.
		PHOTON_API static std::shared_ptr<Program> exception(const ProgramEntry & entry);

		//!	@brief		Create a callables program group. Either entry can be empty.
		PHOTON_API static std::shared_ptr<Program> callables(const ProgramEntry & dc, const ProgramEntry & cc);

		//!	@brief		Create a hitgroup program group. Any entry can be empty (default ProgramEntry{}).
		PHOTON_API static std::shared_ptr<Program> hitgroup(const ProgramEntry & is, const ProgramEntry & ch, const ProgramEntry & ah);
	};

	/*****************************************************************************
	*****************************    ProgramEntry    *****************************
	*****************************************************************************/

	/**
	 *	@brief		Lightweight reference to a shader entry point.
	 *	@note		Does not create any OptiX resources. Pass to Program factory methods to create program groups.
	 */
	class ProgramEntry
	{
		friend class Program;

	public:

		//!	@brief	Default constructor creates an invalid entry.
		ProgramEntry() = default;

		//!	@brief	Construct a ProgramEntry with the given module, entry name, and program type.
		explicit ProgramEntry(std::shared_ptr<Module> module, Program::Type type, std::string entryName)
			: m_module(std::move(module)), m_entryName(std::move(entryName)), m_type(type) {}

	public:

		//!	@brief		Check if this entry is valid.
		bool valid() const { return m_module != nullptr; }

		//!	@brief		Return the program type.
		Program::Type type() const { return m_type; }

	private:

		const std::shared_ptr<Module>		m_module;
		const std::string					m_entryName;	//!	for debug
		const Program::Type					m_type = Program::Unknow;
	};

	/*****************************************************************************
	*******************************    Pipeline    *******************************
	*****************************************************************************/

	/**
	 *	@brief		Wrapper for an OptiX pipeline.
	 *	@note		A pipeline represents the compiled set of program groups
	 *				(raygen, miss, hit, callable, etc.) and encapsulates the
	 *				execution configuration for launching OptiX kernels.
	 */
	class Pipeline
	{
		NS_NONCOPYABLE(Pipeline)

	public:

		//!	@brief	
		PHOTON_API explicit Pipeline(SharedContext context,
									 ns::ArrayProxy<std::shared_ptr<Program>> programs,
									 const OptixPipelineCompileOptions & pipelineCompileOptions = OptixPipelineCompileOptions{},
									 const OptixPipelineLinkOptions & pipelineLinkOptions = OptixPipelineLinkOptions{});

		//!	@brief	Destructor.
		PHOTON_API ~Pipeline();

	public:

		/**
		 *	@brief		Launch the pipeline with the given parameters.
		 *	@tparam		Type - Type of the pipeline parameter structure.
		 *	@param[in]	stream - CUDA stream to enqueue the launch on.
		 *	@param[in]	pipelineParams - Pointer to the pipeline parameter structure.
		 *	@param[in]	sbt - Shader Binding Table that defines program associations (raygen, miss, hit, callable, etc.).
		 *	@param[in]	width - Launch width in threads.
		 *	@param[in]	height - Launch height in threads (default = 1).
		 *	@param[in]	depth - Launch depth in threads (default = 1).
		 *	@retval		ns::Stream& - Reference to this stream (enables method chaining).
		 *	@note		This function dispatches ray generation and subsequent OptiX programs across the specified launch dimensions.
		 *	@note		Multiple launches may be issued in parallel from multiple threads as long as they target different CUDA streams.
		 *	@warning	The stream and pipeline must belong to the same device context.
		 */
		template<typename Type> ns::Stream & launch(ns::Stream & stream, ns::dev::Ptr<const Type> pipelineParams, const OptixShaderBindingTable & sbt, size_t width, size_t height = 1, size_t depth = 1)
		{
			this->doLaunch(stream, pipelineParams.data(), sizeof(Type), sbt, static_cast<unsigned int>(width), static_cast<unsigned int>(height), static_cast<unsigned int>(depth));

			return stream;
		}

	private:

		/**
		 *	@brief		Internal implementation of pipeline launch.
		 *	@note		This is the low-level entry point that forwards the call to the OptiX API with untyped pipeline parameters.
		 */
		PHOTON_API void doLaunch(ns::Stream & stream, const void * pipelineParams, size_t pipelineParamsSize, const OptixShaderBindingTable & sbt, unsigned int width, unsigned int height, unsigned int depth);

	private:

		const SharedContext 	m_context;

		OptixPipeline			m_hPipeline;
	};
}