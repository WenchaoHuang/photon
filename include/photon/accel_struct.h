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
#include <nucleus/array_1d.h>
#include <nucleus/array_proxy.h>
#include <nucleus/vector_types.h>
#include <nucleus/device_pointer.h>
#include <optix.h>

//	Backward compatibility: OptixBuildInputAabbArray was renamed to
//	OptixBuildInputCustomPrimitiveArray in OptiX 7.1.
#if OPTIX_VERSION < 70100
	typedef OptixBuildInputAabbArray		OptixBuildInputCustomPrimitiveArray;
#endif

namespace PHOTON_NAMESPACE
{
	/*****************************************************************************
	*****************************    AccelStruct    ******************************
	*****************************************************************************/

	//	Base class for acceleration structures. Holds all common GPU buffer state
	//	and implements rebuild/refit logic shared by all concrete subclasses.
	class AccelStruct
	{
		NS_NONCOPYABLE(AccelStruct)

	public:

		//	Construct with the device context that owns this structure.
		PHOTON_API explicit AccelStruct(std::shared_ptr<class DeviceContext> deviceContext);

		//	Virtual destructor.
		virtual ~AccelStruct() {}

	public:

		//	Enum defining different subtypes of acceleration structures.
		enum SubType
		{
			Geometry,		//	Geometry acceleration structure (GAS).
			Instance,		//	Instance acceleration structure (IAS).
		};

		//	Struct encapsulating build options for acceleration structure construction.
		struct BuildOptions
		{
			size_t					headerSize;
			OptixBuildFlags			buildFlags;
			OptixMotionOptions		motionOptions;
		};


		//	Returns the subtype (Geometry or Instance) of this acceleration structure.
		virtual SubType subType() const = 0;

		//	Returns true if the acceleration structure has not been built yet.
		bool empty() const { return m_hTraversable == 0; }

		//	Returns the native OptiX traversable handle.
		OptixTraversableHandle handle() const { return m_hTraversable; }

		//	Returns the device context associated with this acceleration structure.
		std::shared_ptr<class DeviceContext> deviceContext() const { return m_deviceContext; }

		//	Returns true if the acceleration structure was built with update support.
		bool allowUpdate() const { return (m_buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0; }

	protected:

		//	Returns true if compaction was enabled when the structure was built.
		bool allowCompaction() const { return (m_buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0; }

		/**
		 *	@brief		Returns a device pointer to the user header buffer prepended to the GAS output.
		 *	@details	Returns nullptr if no header size was requested at build time.
		 */
		dev::Ptr<unsigned char> gasHeaderBuffer()
		{
			if ((m_headerSize != 0) && this->allowCompaction())
				return dev::Ptr<unsigned char>(m_compactedBuffer.data(), m_headerSize);
			else if (m_headerSize != 0)
				return dev::Ptr<unsigned char>(m_outputBuffer.data(), m_headerSize);
			else
				return dev::Ptr<unsigned char>(nullptr);
		}

		//	Internal build helper called by each concrete subclass build() method.
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, BuildOptions buildOptions);

		//	Rebuilds the acceleration structure in-place using the last build inputs.
		PHOTON_API void rebuild(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs);

		/**
		 *	@brief		Refits the acceleration structure without changing topology.
		 *	@warning	Only the device pointers and/or their buffer content may be changed.
		 */
		PHOTON_API void refit(ns::Stream & stream, const std::vector<OptixBuildInput> & buildInputs);

		size_t										m_headerSize;
		std::vector<OptixBuildInput>				m_cachedBuildInputs;

	private:

		ns::Array<unsigned char>					m_tempBuffer;
		ns::Array<unsigned char>					m_outputBuffer;
		ns::Array<unsigned char>					m_compactedBuffer;
		OptixTraversableHandle						m_hTraversable;
		OptixAccelBuildOptions						m_buildOptions;
		const std::shared_ptr<class DeviceContext>	m_deviceContext;
	};

	/*****************************************************************************
	***************************    GeomAccelStruct    ****************************
	*****************************************************************************/

	//	Base class for geometry acceleration structures (GAS).
	class GeomAccelStruct : public AccelStruct
	{

	public:

		//!	Enum defining different types of primitives supported by the acceleration structure.
		enum PrimitiveType
		{
			Triangle,		//!	Geometry acceleration structure containing built-in triangles.
			Sphere,			//!	Geometry acceleration structure containing built-in spheres.
			Curve,			//!	Geometry acceleration structure containing built-in curve primitives.
			AABB,			//!	Geometry acceleration structure containing custom primitives.
		};

		//!	Returns the primitive type of this GAS.
		virtual PrimitiveType primitiveType() const = 0;

		//!	Returns SubType::Geometry.
		SubType subType() const override final { return SubType::Geometry; }

		//	Construct with the device context that owns this GAS.
		explicit GeomAccelStruct(std::shared_ptr<class DeviceContext> deviceContext) : AccelStruct(std::move(deviceContext)) {}

	public:

		/**
		 *	@brief		Device pointer to the header buffer of a GAS (Geometry Acceleration Structure).
		 * 
		 *	@details	This returns a pointer to the metadata header associated with the GAS.
		 *				The header region is placed before the OptiX acceleration data and is designed
		 *				for user-defined information such as primitive offsets, material indices,
		 *				or any custom per-GAS data layout.
		 *
		 *	@details	Memory layout example (including alignment considerations):
		 *				|------ user header buffer ------|--------- GAS output buffer --------- ... |
		 *				       ^ returned pointer                   ^BVH and geometry data
		 *
		 *	@note		The size of the header region will be aligned up to `OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT`.
		 *				That means the actual offset to the GAS acceleration data may be larger than the
		 *				raw header struct size and should be computed accordingly.
		 * 
		 *	@note		This header is **intended to be accessed and interpreted by the application**,
		 *				and its structure is defined by the user. It can be used directly on device-side
		 *				kernels without copying back to host.
		 *
		 *	@example	auto geoBuffer = reinterpret_cast<const GeoBuffer*>(optixGetGASPointerFromHandle(optixGetGASTraversableHandle()) - headerSizeInBytes);
		 *	@note		API `optixGetGASPointerFromHandle` requires `OPTIX_VERSION >= 8.1.0`.
		 *
		 */
		dev::Ptr<unsigned char> headerBuffer() { return this->gasHeaderBuffer(); }

		//!	Returns the size of the header region in bytes (as passed to build()).
		size_t headerSize() const { return m_headerSize; }
	};

	/*****************************************************************************
	*************************    AccelStructTriangle    **************************
	*****************************************************************************/

	//!	@brief		Concrete GAS for built-in triangle primitives.
	class AccelStructTriangle : public GeomAccelStruct
	{

	public:

		//!	@brief	Constructs an empty triangle GAS associated with the given device context.
		explicit AccelStructTriangle(std::shared_ptr<class DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext)) {}

		//!	@brief	Builds the triangle GAS from the supplied build inputs.
		void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs, BuildOptions buildOptions)
		{
			AccelStruct::build(stream, allocator, this->makeBuildInput(buildInputs), buildOptions);
		}

		//!	@brief	Rebuilds the triangle GAS from the supplied build inputs.
		void rebuild(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs)
		{
			AccelStruct::rebuild(stream, this->makeBuildInput(buildInputs));
		}

		//!	@brief	Refits the triangle GAS from the supplied build inputs.
		void refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs)
		{
			AccelStruct::refit(stream, this->makeBuildInput(buildInputs));
		}

		//!	@brief	Returns PrimitiveType::Triangle.
		PrimitiveType primitiveType() const override final { return PrimitiveType::Triangle; }

	private:

		//!	@brief	Helper function to convert triangle build inputs to the generic `OptixBuildInput` format.
		const std::vector<OptixBuildInput> & makeBuildInput(ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs)
		{
			m_cachedBuildInputs.resize(buildInputs.size());

			for (size_t i = 0; i < m_cachedBuildInputs.size(); i++)
			{
				m_cachedBuildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
				m_cachedBuildInputs[i].triangleArray = buildInputs[i];
			}
			return m_cachedBuildInputs;
		}
	};

	/*****************************************************************************
	***************************    AccelStructAabb    ****************************
	*****************************************************************************/

	//	Concrete GAS for custom AABB (user-defined intersection) primitives.
	class AccelStructAabb : public GeomAccelStruct
	{

	public:

		//!	@brief	Constructs an empty AABB GAS associated with the given device context.
		explicit AccelStructAabb(std::shared_ptr<class DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext)) {}

		//!	@brief	Builds the AABB GAS from the supplied build inputs.
		void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs, BuildOptions buildOptions)
		{
			AccelStruct::build(stream, allocator, this->makeBuildInput(buildInputs), buildOptions);
		}

		//!	@brief	Rebuilds the AABB GAS from the supplied build inputs.
		void rebuild(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs)
		{
			AccelStruct::rebuild(stream, this->makeBuildInput(buildInputs));
		}

		//!	@brief	Refits the AABB GAS from the supplied build inputs.
		void refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs)
		{
			AccelStruct::refit(stream, this->makeBuildInput(buildInputs));
		}

		//!	@brief	Returns PrimitiveType::AABB.
		PrimitiveType primitiveType() const override final { return PrimitiveType::AABB; }

	private:

		//!	@brief	Helper function to convert custom primitive build inputs to the generic `OptixBuildInput` format.
		const std::vector<OptixBuildInput> & makeBuildInput(ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs)
		{
			m_cachedBuildInputs.resize(buildInputs.size());

			for (size_t i = 0; i < m_cachedBuildInputs.size(); i++)
			{
				m_cachedBuildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
			#if OPTIX_VERSION >= 70100
				m_cachedBuildInputs[i].customPrimitiveArray = buildInputs[i];
			#else
				m_cachedBuildInputs[i].aabbArray = buildInputs[i];
			#endif
			}
			return m_cachedBuildInputs;
		}
	};

	/*****************************************************************************
	***************************    AccelStructCurve    ***************************
	*****************************************************************************/
	
#if OPTIX_VERSION >= 70100
	/**
	 *	@note		Requires Optix version >= 7.1.0
	 */
	//	Concrete GAS for built-in curve primitives.
	class AccelStructCurve : public GeomAccelStruct
	{

	public:

		//!	@brief	Constructs an empty curve GAS associated with the given device context.
		explicit AccelStructCurve(std::shared_ptr<class DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext)) {}

		//!	@brief	Builds the curve GAS from the supplied build inputs.
		void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs, BuildOptions buildOptions)
		{
			AccelStruct::build(stream, allocator, this->makeBuildInput(buildInputs), buildOptions);
		}

		//!	@brief	Rebuilds the curve GAS from the supplied build inputs.
		void rebuild(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs)
		{
			AccelStruct::rebuild(stream, this->makeBuildInput(buildInputs));
		}

		//!	@brief	Refits the curve GAS from the supplied build inputs.
		void refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs)
		{
			AccelStruct::refit(stream, this->makeBuildInput(buildInputs));
		}

		//!	@brief	Returns PrimitiveType::Curve.
		PrimitiveType primitiveType() const override final { return PrimitiveType::Curve; }

	private:

		//!	@brief	Helper function to convert curve build inputs to the generic `OptixBuildInput` format.
		const std::vector<OptixBuildInput> & makeBuildInput(ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs)
		{
			m_cachedBuildInputs.resize(buildInputs.size());

			for (size_t i = 0; i < m_cachedBuildInputs.size(); i++)
			{
				m_cachedBuildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_CURVES;
				m_cachedBuildInputs[i].curveArray = buildInputs[i];
			}
			return m_cachedBuildInputs;
		}
	};
#endif

	/*****************************************************************************
	**************************    AccelStructSphere    ***************************
	*****************************************************************************/

#if OPTIX_VERSION >= 70500
	/**
	 *	@note		Requires Optix version >= 7.5.0
	 */
	//	Concrete GAS for built-in sphere primitives.
	class AccelStructSphere : public GeomAccelStruct
	{

	public:

		//!	@brief	Constructs an empty sphere GAS associated with the given device context.
		explicit AccelStructSphere(std::shared_ptr<class DeviceContext> deviceContext) : GeomAccelStruct(std::move(deviceContext)) {}

		//!	@brief	Builds the sphere GAS from the supplied build inputs.
		void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs, BuildOptions buildOptions)
		{
			AccelStruct::build(stream, allocator, this->makeBuildInput(buildInputs), buildOptions);
		}

		//!	@brief	Rebuilds the sphere GAS from the supplied build inputs.
		void rebuild(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs)
		{
			AccelStruct::rebuild(stream, this->makeBuildInput(buildInputs));
		}

		//!	@brief	Refits the sphere GAS from the supplied build inputs.
		void refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs)
		{
			AccelStruct::refit(stream, this->makeBuildInput(buildInputs));
		}

		//!	@brief	Returns PrimitiveType::Sphere.
		PrimitiveType primitiveType() const override final { return PrimitiveType::Sphere; }

	private:

		//!	@brief	Helper function to convert sphere build inputs to the generic `OptixBuildInput` format.
		const std::vector<OptixBuildInput> & makeBuildInput(ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs)
		{
			m_cachedBuildInputs.resize(buildInputs.size());

			for (size_t i = 0; i < m_cachedBuildInputs.size(); i++)
			{
				m_cachedBuildInputs[i].type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
				m_cachedBuildInputs[i].sphereArray = buildInputs[i];
			}
			return m_cachedBuildInputs;
		}
	};
#endif

	/*****************************************************************************
	***************************    InstAccelStruct    ****************************
	*****************************************************************************/

	//	Concrete IAS (Instance Acceleration Structure).
	class InstAccelStruct : public AccelStruct
	{

	public:

		//!	@brief	Constructs an empty IAS associated with the given device context.
		explicit InstAccelStruct(std::shared_ptr<class DeviceContext> deviceContext) : AccelStruct(std::move(deviceContext)) {}

		//!	@brief	Uploads the current host-side instance data and then builds the IAS.
		void build(ns::Stream & stream, ns::AllocPtr allocator, OptixBuildInputInstanceArray buildInput, BuildOptions buildOptions)
		{
			AccelStruct::build(stream, allocator, this->makeBuildInput(buildInput), buildOptions);
		}

		//!	@brief	Uploads the current host-side instance data and then rebuilds the IAS.
		void rebuild(ns::Stream & stream, OptixBuildInputInstanceArray buildInput)
		{
			AccelStruct::rebuild(stream, this->makeBuildInput(buildInput));
		}

		//!	@brief	Uploads the current host-side instance data and then refits the IAS.
		void refit(ns::Stream & stream, OptixBuildInputInstanceArray buildInput)
		{
			AccelStruct::refit(stream, this->makeBuildInput(buildInput));
		}

		//!	@brief	Returns SubType::Instance.
		SubType subType() const override final { return SubType::Instance; }

	private:

		//!	@brief	Helper function to convert instance build input to the generic `OptixBuildInput` format.
		std::vector<OptixBuildInput> & makeBuildInput(OptixBuildInputInstanceArray buildInput)
		{
			m_cachedBuildInputs.resize(1);
			m_cachedBuildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			m_cachedBuildInputs[0].instanceArray = buildInput;
			return m_cachedBuildInputs;
		}
	};
}