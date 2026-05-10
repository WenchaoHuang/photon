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
	typedef OptixBuildInputAabbArray	OptixBuildInputCustomPrimitiveArray;
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

		//	Returns true if the acceleration structure has not been built yet.
		bool empty() const { return m_hTraversable == 0; }

		//	Returns true if the acceleration structure was built with update support.
		bool allowUpdate() const { return (m_buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0; }

		//	Returns the native OptiX traversable handle.
		OptixTraversableHandle handle() const { return m_hTraversable; }

		//	Returns the device context associated with this acceleration structure.
		std::shared_ptr<class DeviceContext> deviceContext() const { return m_deviceContext; }

		//	Returns the subtype (Geometry or Instance) of this acceleration structure.
		virtual SubType subType() const = 0;

		//	Rebuilds the acceleration structure in-place using the last build inputs.
		PHOTON_API virtual void rebuild(ns::Stream & stream);

		/**
		 *	@brief		Refits the acceleration structure without changing topology.
		 *	@warning	Only the device pointers and/or their buffer content may be changed.
		 */
		PHOTON_API virtual void refit(ns::Stream & stream);

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
		PHOTON_API void buildBase(ns::Stream & stream, ns::AllocPtr allocator, const std::vector<OptixBuildInput> & buildInputs, OptixAccelBuildOptions buildOptions, size_t headerSize);

		size_t										m_headerSize;
		unsigned int								m_numSbtRecords;
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

		//!	Geometry flags describing the primitive behavior.
		enum GeomFlags
		{
			None							= 0,			//!	No flags set.
			DisableAnyhit					= 1u << 0,		//!	Disables the invocation of the anyhit program. Can be overridden by OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT and OPTIX_RAY_FLAG_ENFORCE_ANYHIT.
			RequireSingleAnyhitCall			= 1u << 1,		//!	If set, an intersection with the primitive will trigger one and only one invocation of the anyhit program. Otherwise, the anyhit program may be invoked more than once.
		#if OPTIX_VERSION >= 70500
			DisableTriangleFaceCulling		= 1u << 2,		//!	Prevent triangles from getting culled due to their orientation. Effectively ignores ray flags OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES and OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES.
		#endif
		};

		//	Construct with the device context that owns this GAS.
		explicit GeomAccelStruct(std::shared_ptr<class DeviceContext> deviceContext) : AccelStruct(std::move(deviceContext)) {}

		//!	Returns SubType::Geometry.
		SubType subType() const override final { return SubType::Geometry; }

		//!	Returns the primitive type of this GAS.
		virtual PrimitiveType primitiveType() const = 0;

		//	Returns the total number of SBT records across all build inputs.
		unsigned int numSbtRecords() const { return m_numSbtRecords; }

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

	//	Concrete GAS for built-in triangle primitives.
	class AccelStructTriangle : public GeomAccelStruct
	{

	public:

		//	Constructs an empty triangle GAS associated with the given device context.
		PHOTON_API explicit AccelStructTriangle(std::shared_ptr<class DeviceContext> deviceContext);

		//	Returns PrimitiveType::Triangle.
		PrimitiveType primitiveType() const override final { return PrimitiveType::Triangle; }

		//	Returns a constant reference to the cached triangle build inputs.
		const std::vector<OptixBuildInputTriangleArray> & buildInputs() const { return m_buildInputs; }

		//	Builds the triangle GAS from the supplied build inputs.
		//	Callers should zero-initialize each OptixBuildInputTriangleArray first
		//	(e.g. OptixBuildInputTriangleArray{}), then populate all required OptiX
		//	fields before calling build(). In particular, initialize the vertex data
		//	(description pointers/count, vertexFormat, and vertexStrideInBytes) and,
		//	when indexed triangles are used, the index data (index buffer/count,
		//	indexFormat, and indexStrideInBytes). Also provide per-input flags /
		//	numSbtRecords as required by the OptiX build input contract.
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate);

	private:

		std::vector<OptixBuildInputTriangleArray>	m_buildInputs;
	};

	/*****************************************************************************
	***************************    AccelStructAabb    ****************************
	*****************************************************************************/

	//	Concrete GAS for custom AABB (user-defined intersection) primitives.
	class AccelStructAabb : public GeomAccelStruct
	{

	public:

		//	Constructs an empty AABB GAS associated with the given device context.
		PHOTON_API explicit AccelStructAabb(std::shared_ptr<class DeviceContext> deviceContext);

		//	Returns PrimitiveType::AABB.
		PrimitiveType primitiveType() const override final { return PrimitiveType::AABB; }

		//	Returns a constant reference to the cached AABB build inputs.
		const std::vector<OptixBuildInputCustomPrimitiveArray> & buildInputs() const { return m_buildInputs; }

		//	Builds the AABB GAS from the supplied build inputs.
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate);

	private:

		std::vector<OptixBuildInputCustomPrimitiveArray>	m_buildInputs;
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

		//	Enum defining different curve types.
		enum CurveType
		{
			RoundLinear				= 0x2503,		//	Piecewise linear curve with circular cross-section.
			RoundCubicBSpline		= 0x2502,		//	B-spline curve of degree 3 with circular cross-section.
			RoundQuadraticBSpline	= 0x2501,		//	B-spline curve of degree 2 with circular cross-section.
		#if OPTIX_VERSION >= 70700
			RoundCubicBezier		= 0x2507,		//	Bezier curve of degree 3 with circular cross-section.
			FlatQuadraticBSpline	= 0x2505,		//	B-spline curve of degree 2 with oriented, flat cross-section.
		#endif
		#if OPTIX_VERSION >= 70400
			RoundCatmullRom			= 0x2504,		//	CatmullRom curve with circular cross-section.
		#endif
		};

		//	Constructs an empty curve GAS associated with the given device context.
		PHOTON_API explicit AccelStructCurve(std::shared_ptr<class DeviceContext> deviceContext);

		//	Returns PrimitiveType::Curve.
		PrimitiveType primitiveType() const override final { return PrimitiveType::Curve; }

		//	Returns a constant reference to the cached curve build inputs.
		const std::vector<OptixBuildInputCurveArray> & buildInputs() const { return m_buildInputs; }

		//	Builds the curve GAS from the supplied build inputs.
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate);

	private:

		std::vector<OptixBuildInputCurveArray>	m_buildInputs;
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

		//	Constructs an empty sphere GAS associated with the given device context.
		PHOTON_API explicit AccelStructSphere(std::shared_ptr<class DeviceContext> deviceContext);

		//	Returns PrimitiveType::Sphere.
		PrimitiveType primitiveType() const override final { return PrimitiveType::Sphere; }

		//	Returns a constant reference to the cached sphere build inputs.
		const std::vector<OptixBuildInputSphereArray> & buildInputs() const { return m_buildInputs; }

		//	Builds the sphere GAS from the supplied build inputs.
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate);

	private:

		std::vector<OptixBuildInputSphereArray>	m_buildInputs;
	};
#endif

	/*****************************************************************************
	***************************    InstAccelStruct    ****************************
	*****************************************************************************/

	//	Concrete IAS (Instance Acceleration Structure).
	class InstAccelStruct : public AccelStruct
	{

	public:

		//	Flags set on the InstDesc::flags. These can be or'ed together to combine multiple flags.
		enum InstFlags : unsigned int
		{
			None								= 0,			//	No special flag set.
			DisableTriangleFaceCulling			= 1u << 0,		//	Prevent triangles from getting culled due to their orientation.
			FlipTriangleFacing					= 1u << 1,		//	Flip triangle orientation. This affects front/backface culling as well as the reported face in case of a hit.
			DisableAnyhit						= 1u << 2,		//	Disable anyhit programs for all geometries of the instance. Can be overridden by Flags::EnforceAnyhit. This flag is mutually exclusive with Flags::EnforceAnyhit.
			EnforceAnyhit						= 1u << 3,		//	Enables anyhit programs for all geometries of the instance. Overrides OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT. Can be overridden by Flags::eDisableAnyhit.
		#if OPTIX_VERSION >= 70600
			ForceOpacityMicromapAsTwoState		= 1u << 4,		//	Force 4-state opacity micromaps to behave as 2-state opacity micromaps during traversal.
			DisableOpacityMicromaps				= 1u << 5,		//	Don't perform opacity micromap query for this instance. GAS must be built with ALLOW_DISABLE_OPACITY_MICROMAPS for this to be valid.
		#endif
		};

		//	Constructs an empty IAS associated with the given device context.
		PHOTON_API explicit InstAccelStruct(std::shared_ptr<class DeviceContext> deviceContext);

		//	Returns SubType::Instance.
		SubType subType() const override final { return SubType::Instance; }

		/**
		 *	Returns a constant reference to the host-side copy of OptixInstance descriptors.
		 *
		 *	Each OptixInstance stored here is expected to be fully initialized for OptiX.
		 *	At minimum, callers should ensure:
		 *	- traversableHandle references a valid child GAS/IAS,
		 *	- transform contains a valid row-major 3x4 affine transform,
		 *	- instanceId, visibilityMask, sbtOffset, and flags are explicitly set.
		 *
		 *	When no instance transform is desired, use the identity transform:
		 *	{ 1, 0, 0, 0,
		 *	  0, 1, 0, 0,
		 *	  0, 0, 1, 0 }.
		 *
		 *	Do not rely on zero-initialization for transform, as an all-zero matrix is not
		 *	a valid default "no transform" affine transform.
		 */
		const std::vector<OptixInstance> & buildInputs() const { return m_hostInstances; }

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
		PHOTON_API void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixInstance> buildInputs, bool preferFastTrace, bool allowUpdate);

		//	Uploads the current host-side instance data and then rebuilds the IAS.
		PHOTON_API void rebuild(ns::Stream & stream) override;

		//	Uploads the current host-side instance data and then refits the IAS.
		PHOTON_API void refit(ns::Stream & stream) override;

	private:

		std::vector<OptixInstance>		m_hostInstances;
		ns::Array<OptixInstance>		m_instances;
	};
}