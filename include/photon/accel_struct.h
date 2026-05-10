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
#include <vector>

//	Backward compatibility: OptixBuildInputAabbArray was renamed to
//	OptixBuildInputCustomPrimitiveArray in OptiX 7.1.
#if OPTIX_VERSION < 70100
	typedef OptixBuildInputAabbArray	OptixBuildInputCustomPrimitiveArray;
#endif

namespace PHOTON_NAMESPACE
{
	struct AccelBuildOptions
	{
		size_t				headerSize = 0;
		bool				preferFastTrace = true;
		bool				allowUpdate = false;
		bool				allowCompaction = false;
		OptixMotionOptions	motionOptions = {};
	};

	struct AccelBufferLayout
	{
		size_t	headerSize = 0;
		size_t	accelBufferOffset = 0;
	};

	/*****************************************************************************
	*****************************    AccelStruct    ******************************
	*****************************************************************************/

	//	Base class for acceleration structures. Holds all common GPU buffer state
	//	and implements rebuild/refit logic shared by all concrete subclasses.
	class AccelStruct
	{
		NS_NONCOPYABLE(AccelStruct)

	public:

		//	Enum defining different subtypes of acceleration structures.
		enum SubType
		{
			Geometry,		//	Geometry acceleration structure (GAS).
			Instance,		//	Instance acceleration structure (IAS).
		};

		//	Construct with the device context that owns this structure.
		PHOTON_API explicit AccelStruct(std::shared_ptr<class DeviceContext> deviceContext);

		//	Virtual destructor.
		PHOTON_API virtual ~AccelStruct();

	public:

		//	Returns the subtype (Geometry or Instance) of this acceleration structure.
		virtual SubType subType() const = 0;

		//	Returns true if the acceleration structure has not been built yet.
		bool empty() const { return m_hTraversable == 0; }

		//	Returns the native OptiX traversable handle.
		OptixTraversableHandle handle() const { return m_hTraversable; }

		//	Returns the device context associated with this acceleration structure.
		std::shared_ptr<class DeviceContext> deviceContext() const { return m_deviceContext; }

		//	Returns true if the acceleration structure was built with update support.
		bool allowUpdate() const { return (m_optixBuildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0; }

		//	Returns the high-level options used for the current build state.
		const AccelBuildOptions & buildOptions() const { return m_buildDesc; }

		//	Returns the user-header / acceleration-data layout of the output buffer.
		const AccelBufferLayout & bufferLayout() const { return m_bufferLayout; }

		//	Rebuilds the acceleration structure in-place using the last build inputs.
		PHOTON_API virtual void rebuild(ns::Stream & stream);

		/**
		 *	@brief		Refits the acceleration structure without changing topology.
		 *	@warning	Only the device pointers and/or their buffer content may be changed.
		 */
		PHOTON_API virtual void refit(ns::Stream & stream);

	protected:

		//	Returns true if compaction was enabled when the structure was built.
		bool allowCompaction() const { return (m_optixBuildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0; }

		/**
		 *	@brief		Returns a device pointer to the user header buffer prepended to the GAS output.
		 *	@details	Returns nullptr if no header size was requested at build time.
		 */
		dev::Ptr<unsigned char> gasHeaderBuffer()
		{
			if ((m_bufferLayout.headerSize != 0) && this->allowCompaction())
				return dev::Ptr<unsigned char>(m_compactedBuffer.data(), m_bufferLayout.headerSize);
			else if (m_bufferLayout.headerSize != 0)
				return dev::Ptr<unsigned char>(m_outputBuffer.data(), m_bufferLayout.headerSize);
			else
				return dev::Ptr<unsigned char>(nullptr);
		}

		//	Derived classes refresh cached OptiX build inputs from their stored sources.
		PHOTON_API virtual void prepareBuildInputs(ns::Stream & stream) = 0;

		//	Internal build helper used after the concrete source state has been stored.
		PHOTON_API void buildPrepared(ns::Stream & stream, ns::AllocPtr allocator, const AccelBuildOptions & buildOptions);

	private:

		//	Internal build helper called after build inputs have been prepared.
		PHOTON_API void buildBase(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, OptixAccelBuildOptions buildOptions, size_t headerSize);

		mutable std::vector<OptixBuildInput>		m_cachedBuildInputs;
		ns::Array<unsigned char>					m_tempBuffer;
		ns::Array<unsigned char>					m_outputBuffer;
		ns::Array<unsigned char>					m_compactedBuffer;
		OptixTraversableHandle						m_hTraversable;
		OptixAccelBuildOptions						m_optixBuildOptions;
		AccelBuildOptions							m_buildDesc;
		AccelBufferLayout							m_bufferLayout;
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

		//!	Returns the user-visible header size in bytes.
		size_t headerSize() const { return this->bufferLayout().headerSize; }

		//!	Returns the aligned offset from the output buffer start to the OptiX accel data.
		size_t accelBufferOffset() const { return this->bufferLayout().accelBufferOffset; }
	};

	/*****************************************************************************
	*************************    AccelStructTriangle    **************************
	*****************************************************************************/

	//	Concrete GAS for built-in triangle primitives.
	class AccelStructTriangle : public GeomAccelStruct
	{

	public:

		//	Returns PrimitiveType::Triangle.
		PrimitiveType primitiveType() const override final { return PrimitiveType::Triangle; }

		//	Constructs an empty triangle GAS associated with the given device context.
		PHOTON_API explicit AccelStructTriangle(std::shared_ptr<class DeviceContext> deviceContext);

		//	Builds the triangle GAS from the supplied build inputs.
		//	Callers should zero-initialize each OptixBuildInputTriangleArray first
		//	(e.g. OptixBuildInputTriangleArray{}), then populate all required OptiX
		//	fields before calling build(). In particular, initialize the vertex data
		//	(description pointers/count, vertexFormat, and vertexStrideInBytes) and,
		//	when indexed triangles are used, the index data (index buffer/count,
		//	indexFormat, and indexStrideInBytes). Also provide per-input flags /
		//	numSbtRecords as required by the OptiX build input contract.
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs, const AccelBuildOptions & buildOptions);
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate);
		PHOTON_API void rebuild(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs);
		PHOTON_API void refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs);

	protected:

		PHOTON_API void prepareBuildInputs(ns::Stream & stream) override;

	private:

		std::vector<OptixBuildInputTriangleArray>	m_buildSources;
	};

	/*****************************************************************************
	***************************    AccelStructAabb    ****************************
	*****************************************************************************/

	//	Concrete GAS for custom AABB (user-defined intersection) primitives.
	class AccelStructAabb : public GeomAccelStruct
	{

	public:

		//	Returns PrimitiveType::AABB.
		PrimitiveType primitiveType() const override final { return PrimitiveType::AABB; }

		//	Constructs an empty AABB GAS associated with the given device context.
		PHOTON_API explicit AccelStructAabb(std::shared_ptr<class DeviceContext> deviceContext);

		//	Builds the AABB GAS from the supplied build inputs.
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs, const AccelBuildOptions & buildOptions);
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate);
		PHOTON_API void rebuild(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs);
		PHOTON_API void refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs);

	protected:

		PHOTON_API void prepareBuildInputs(ns::Stream & stream) override;

	private:

		std::vector<OptixBuildInputCustomPrimitiveArray>	m_buildSources;
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

		//	Returns PrimitiveType::Curve.
		PrimitiveType primitiveType() const override final { return PrimitiveType::Curve; }

		//	Constructs an empty curve GAS associated with the given device context.
		PHOTON_API explicit AccelStructCurve(std::shared_ptr<class DeviceContext> deviceContext);

		//	Builds the curve GAS from the supplied build inputs.
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs, const AccelBuildOptions & buildOptions);
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate);
		PHOTON_API void rebuild(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs);
		PHOTON_API void refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs);

	protected:

		PHOTON_API void prepareBuildInputs(ns::Stream & stream) override;

	private:

		std::vector<OptixBuildInputCurveArray>	m_buildSources;
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

		//	Returns PrimitiveType::Sphere.
		PrimitiveType primitiveType() const override final { return PrimitiveType::Sphere; }

		//	Constructs an empty sphere GAS associated with the given device context.
		PHOTON_API explicit AccelStructSphere(std::shared_ptr<class DeviceContext> deviceContext);

		//	Builds the sphere GAS from the supplied build inputs.
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs, const AccelBuildOptions & buildOptions);
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate);
		PHOTON_API void rebuild(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs);
		PHOTON_API void refit(ns::Stream & stream, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs);

	protected:

		PHOTON_API void prepareBuildInputs(ns::Stream & stream) override;

	private:

		std::vector<OptixBuildInputSphereArray>	m_buildSources;
	};
#endif

	/*****************************************************************************
	***************************    InstAccelStruct    ****************************
	*****************************************************************************/

	//	Concrete IAS (Instance Acceleration Structure).
	class InstAccelStruct : public AccelStruct
	{

	public:

		//	Constructs an empty IAS associated with the given device context.
		PHOTON_API explicit InstAccelStruct(std::shared_ptr<class DeviceContext> deviceContext);

		//	Returns SubType::Instance.
		SubType subType() const override final { return SubType::Instance; }

		/**
		 *	Builds the IAS from the supplied instance descriptors.
		 *
		 *	The supplied OptixInstance values are passed through directly and must already
		 *	satisfy OptiX requirements. In particular:
		 *	- traversableHandle must be valid,
		 *	- transform must be a valid row-major 3x4 affine transform array,
		 *	  typically identity if no transform is intended,
		 *	- visibilityMask should be set intentionally (commonly 255 for full visibility),
		 *	- sbtOffset should be set to the intended SBT record offset (commonly 0 when
		 *	  no per-instance offset is needed),
		 *	- flags should be set to the intended OptixInstanceFlags value(s), or 0 when
		 *	  no special instance flags are required.
		 *
		 *	Passing zero-initialized OptixInstance values is unsafe because the transform
		 *	field would not describe a valid identity transform.
		 */
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixInstance> buildInputs, const AccelBuildOptions & buildOptions);
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixInstance> buildInputs, bool preferFastTrace, bool allowUpdate);
		PHOTON_API void rebuild(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixInstance> buildInputs);
		PHOTON_API void refit(ns::Stream & stream, ns::ArrayProxy<OptixInstance> buildInputs);

		//	Uploads the current host-side instance data and then rebuilds the IAS.
		PHOTON_API void rebuild(ns::Stream & stream) override;

		//	Uploads the current host-side instance data and then refits the IAS.
		PHOTON_API void refit(ns::Stream & stream) override;

	private:

		PHOTON_API void prepareBuildInputs(ns::Stream & stream) override;

		std::vector<OptixInstance>		m_buildSources;
		ns::AllocPtr					m_instanceAllocator;
		ns::Array<OptixInstance>		m_instances;
	};
}
