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
#include <nucleus/array_proxy.h>
#include <nucleus/vector_types.h>
#include <nucleus/device_pointer.h>
#include <optix.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

//	Backward compatibility: OptixBuildInputAabbArray was renamed to
//	OptixBuildInputCustomPrimitiveArray in OptiX 7.1.
#if OPTIX_VERSION < 70100
	typedef OptixBuildInputAabbArray	OptixBuildInputCustomPrimitiveArray;
#endif

namespace PHOTON_NAMESPACE
{
	namespace detail
	{
		template<typename T>
		using RemoveCVRef = std::remove_cv_t<std::remove_reference_t<T>>;

		template<typename T>
		inline constexpr bool AlwaysFalse = false;

		template<typename T>
		struct StdArrayTraits
		{
			using ValueType = void;
			static constexpr bool Valid = false;
			static constexpr size_t Count = 0;
		};

		template<typename Type, size_t Size>
		struct StdArrayTraits<std::array<Type, Size>>
		{
			using ValueType = Type;
			static constexpr bool Valid = true;
			static constexpr size_t Count = Size;
		};

		template<typename T>
		inline constexpr bool HasXYZMembers = requires(T value)
		{
			value.x;
			value.y;
			value.z;
		};

		template<typename T>
		inline constexpr bool HasXYMembers = requires(T value)
		{
			value.x;
			value.y;
		};

		template<typename T>
		constexpr OptixVertexFormat optixVertexFormat()
		{
			using Type = RemoveCVRef<T>;
			if constexpr (HasXYZMembers<Type>
				&& std::is_same_v<RemoveCVRef<decltype(std::declval<Type &>().x)>, float>
				&& std::is_same_v<RemoveCVRef<decltype(std::declval<Type &>().y)>, float>
				&& std::is_same_v<RemoveCVRef<decltype(std::declval<Type &>().z)>, float>)
			{
				return OPTIX_VERTEX_FORMAT_FLOAT3;
			}
			else if constexpr (HasXYMembers<Type>
				&& std::is_same_v<RemoveCVRef<decltype(std::declval<Type &>().x)>, float>
				&& std::is_same_v<RemoveCVRef<decltype(std::declval<Type &>().y)>, float>)
			{
				return OPTIX_VERTEX_FORMAT_FLOAT2;
			}
			else if constexpr (StdArrayTraits<Type>::Valid
				&& std::is_same_v<typename StdArrayTraits<Type>::ValueType, float>
				&& (StdArrayTraits<Type>::Count == 2 || StdArrayTraits<Type>::Count == 3))
			{
				if constexpr (StdArrayTraits<Type>::Count == 3)
					return OPTIX_VERTEX_FORMAT_FLOAT3;
				else
					return OPTIX_VERTEX_FORMAT_FLOAT2;
			}
			else if constexpr (std::is_array_v<Type>
				&& std::is_same_v<std::remove_all_extents_t<Type>, float>
				&& (std::extent_v<Type> == 2 || std::extent_v<Type> == 3))
			{
				if constexpr (std::extent_v<Type> == 3)
					return OPTIX_VERTEX_FORMAT_FLOAT3;
				else
					return OPTIX_VERTEX_FORMAT_FLOAT2;
			}
			else
			{
				static_assert(AlwaysFalse<Type>, "Unsupported vertex type. Use float2/float3-like types or set vertexFormat manually.");
			}
		}

		template<typename T>
		constexpr OptixIndicesFormat optixIndicesFormat()
		{
			using Type = RemoveCVRef<T>;
			if constexpr (HasXYZMembers<Type>
				&& std::is_unsigned_v<RemoveCVRef<decltype(std::declval<Type &>().x)>>
				&& std::is_same_v<RemoveCVRef<decltype(std::declval<Type &>().x)>, RemoveCVRef<decltype(std::declval<Type &>().y)>>
				&& std::is_same_v<RemoveCVRef<decltype(std::declval<Type &>().x)>, RemoveCVRef<decltype(std::declval<Type &>().z)>>)
			{
				using ValueType = RemoveCVRef<decltype(std::declval<Type &>().x)>;
				if constexpr (sizeof(ValueType) == sizeof(uint16_t))
					return OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
				else if constexpr (sizeof(ValueType) == sizeof(uint32_t))
					return OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				else
					static_assert(AlwaysFalse<Type>, "Unsupported index element type. Use unsigned short3-like or unsigned int3-like types.");
			}
			else if constexpr (StdArrayTraits<Type>::Valid
				&& std::is_unsigned_v<typename StdArrayTraits<Type>::ValueType>
				&& StdArrayTraits<Type>::Count == 3)
			{
				using ValueType = typename StdArrayTraits<Type>::ValueType;
				if constexpr (sizeof(ValueType) == sizeof(uint16_t))
					return OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
				else if constexpr (sizeof(ValueType) == sizeof(uint32_t))
					return OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				else
					static_assert(AlwaysFalse<Type>, "Unsupported index element type. Use unsigned short[3] or unsigned int[3]-like types.");
			}
			else if constexpr (std::is_array_v<Type>
				&& std::is_unsigned_v<std::remove_all_extents_t<Type>>
				&& std::extent_v<Type> == 3)
			{
				using ValueType = std::remove_all_extents_t<Type>;
				if constexpr (sizeof(ValueType) == sizeof(uint16_t))
					return OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
				else if constexpr (sizeof(ValueType) == sizeof(uint32_t))
					return OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				else
					static_assert(AlwaysFalse<Type>, "Unsupported index element type. Use unsigned short[3] or unsigned int[3]-like types.");
			}
			else
			{
				static_assert(AlwaysFalse<Type>, "Unsupported index triplet type. Use uint3/ushort3-like types or set indexFormat manually.");
			}
		}
	}

	class BuildInputTriangles : public OptixBuildInputTriangleArray
	{

	public:

		BuildInputTriangles()
		{
			static_cast<OptixBuildInputTriangleArray &>(*this) = {};
		}

		BuildInputTriangles & setVertexBuffers(const CUdeviceptr * buffers)
		{
			vertexBuffers = buffers;
			return *this;
		}

		template<typename Vertex>
		BuildInputTriangles & setVertexBuffers(const CUdeviceptr * buffers)
		{
			vertexBuffers = buffers;
			vertexFormat = detail::optixVertexFormat<Vertex>();
			vertexStrideInBytes = sizeof(Vertex);
			return *this;
		}

		BuildInputTriangles & setNumVertices(unsigned int value)
		{
			numVertices = value;
			return *this;
		}

		BuildInputTriangles & setVertexFormat(OptixVertexFormat value)
		{
			vertexFormat = value;
			return *this;
		}

		BuildInputTriangles & setVertexStrideInBytes(unsigned int value)
		{
			vertexStrideInBytes = value;
			return *this;
		}

		BuildInputTriangles & setIndexBuffer(CUdeviceptr value)
		{
			indexBuffer = value;
			return *this;
		}

		template<typename Index>
		BuildInputTriangles & setIndexBuffer(const Index * buffer)
		{
			indexBuffer = reinterpret_cast<CUdeviceptr>(buffer);
			indexFormat = detail::optixIndicesFormat<Index>();
			indexStrideInBytes = sizeof(Index);
			return *this;
		}

		BuildInputTriangles & setNumIndexTriplets(unsigned int value)
		{
			numIndexTriplets = value;
			return *this;
		}

		BuildInputTriangles & setIndexFormat(OptixIndicesFormat value)
		{
			indexFormat = value;
			return *this;
		}

		BuildInputTriangles & setIndexStrideInBytes(unsigned int value)
		{
			indexStrideInBytes = value;
			return *this;
		}

		BuildInputTriangles & setPreTransform(CUdeviceptr value)
		{
			preTransform = value;
			transformFormat = (value == 0) ? OPTIX_TRANSFORM_FORMAT_NONE : OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
			return *this;
		}

		BuildInputTriangles & setFlags(const unsigned int * value)
		{
			flags = value;
			return *this;
		}

		BuildInputTriangles & setNumSbtRecords(unsigned int value)
		{
			numSbtRecords = value;
			return *this;
		}

		BuildInputTriangles & setSbtIndexOffsetBuffer(CUdeviceptr buffer)
		{
			sbtIndexOffsetBuffer = buffer;
			return *this;
		}

		template<typename Offset>
		BuildInputTriangles & setSbtIndexOffsetBuffer(const Offset * buffer)
		{
			sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(buffer);
			sbtIndexOffsetSizeInBytes = sizeof(Offset);
			sbtIndexOffsetStrideInBytes = sizeof(Offset);
			return *this;
		}

		BuildInputTriangles & setSbtIndexOffsetSizeInBytes(unsigned int value)
		{
			sbtIndexOffsetSizeInBytes = value;
			return *this;
		}

		BuildInputTriangles & setSbtIndexOffsetStrideInBytes(unsigned int value)
		{
			sbtIndexOffsetStrideInBytes = value;
			return *this;
		}

		BuildInputTriangles & setPrimitiveIndexOffset(unsigned int value)
		{
			primitiveIndexOffset = value;
			return *this;
		}

		BuildInputTriangles & setTransformFormat(OptixTransformFormat value)
		{
			transformFormat = value;
			return *this;
		}
	};

	class BuildInputAabb : public OptixBuildInputCustomPrimitiveArray
	{

	public:

		BuildInputAabb()
		{
			static_cast<OptixBuildInputCustomPrimitiveArray &>(*this) = {};
		}

		BuildInputAabb & setAabbBuffers(const CUdeviceptr * buffers)
		{
			aabbBuffers = buffers;
			return *this;
		}

		template<typename AabbType>
		BuildInputAabb & setAabbBuffers(const CUdeviceptr * buffers)
		{
			aabbBuffers = buffers;
			strideInBytes = sizeof(AabbType);
			return *this;
		}

		BuildInputAabb & setNumPrimitives(unsigned int value)
		{
			numPrimitives = value;
			return *this;
		}

		BuildInputAabb & setStrideInBytes(unsigned int value)
		{
			strideInBytes = value;
			return *this;
		}

		BuildInputAabb & setFlags(const unsigned int * value)
		{
			flags = value;
			return *this;
		}

		BuildInputAabb & setNumSbtRecords(unsigned int value)
		{
			numSbtRecords = value;
			return *this;
		}

		BuildInputAabb & setSbtIndexOffsetBuffer(CUdeviceptr buffer)
		{
			sbtIndexOffsetBuffer = buffer;
			return *this;
		}

		template<typename Offset>
		BuildInputAabb & setSbtIndexOffsetBuffer(const Offset * buffer)
		{
			sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(buffer);
			sbtIndexOffsetSizeInBytes = sizeof(Offset);
			sbtIndexOffsetStrideInBytes = sizeof(Offset);
			return *this;
		}

		BuildInputAabb & setSbtIndexOffsetSizeInBytes(unsigned int value)
		{
			sbtIndexOffsetSizeInBytes = value;
			return *this;
		}

		BuildInputAabb & setSbtIndexOffsetStrideInBytes(unsigned int value)
		{
			sbtIndexOffsetStrideInBytes = value;
			return *this;
		}

		BuildInputAabb & setPrimitiveIndexOffset(unsigned int value)
		{
			primitiveIndexOffset = value;
			return *this;
		}
	};

#if OPTIX_VERSION >= 70100
	class BuildInputCurves : public OptixBuildInputCurveArray
	{

	public:

		BuildInputCurves()
		{
			static_cast<OptixBuildInputCurveArray &>(*this) = {};
		}

		BuildInputCurves & setCurveType(OptixPrimitiveType value)
		{
			curveType = value;
			return *this;
		}

		BuildInputCurves & setNumPrimitives(unsigned int value)
		{
			numPrimitives = value;
			return *this;
		}

		BuildInputCurves & setVertexBuffers(const CUdeviceptr * buffers)
		{
			vertexBuffers = buffers;
			return *this;
		}

		template<typename Vertex>
		BuildInputCurves & setVertexBuffers(const CUdeviceptr * buffers)
		{
			vertexBuffers = buffers;
			vertexStrideInBytes = sizeof(Vertex);
			return *this;
		}

		BuildInputCurves & setNumVertices(unsigned int value)
		{
			numVertices = value;
			return *this;
		}

		BuildInputCurves & setVertexStrideInBytes(unsigned int value)
		{
			vertexStrideInBytes = value;
			return *this;
		}

		BuildInputCurves & setWidthBuffers(const CUdeviceptr * buffers)
		{
			widthBuffers = buffers;
			return *this;
		}

		template<typename Width>
		BuildInputCurves & setWidthBuffers(const CUdeviceptr * buffers)
		{
			widthBuffers = buffers;
			widthStrideInBytes = sizeof(Width);
			return *this;
		}

		BuildInputCurves & setWidthStrideInBytes(unsigned int value)
		{
			widthStrideInBytes = value;
			return *this;
		}

		BuildInputCurves & setNormalBuffers(const CUdeviceptr * buffers)
		{
			normalBuffers = buffers;
			return *this;
		}

		template<typename Normal>
		BuildInputCurves & setNormalBuffers(const CUdeviceptr * buffers)
		{
			normalBuffers = buffers;
			normalStrideInBytes = sizeof(Normal);
			return *this;
		}

		BuildInputCurves & setNormalStrideInBytes(unsigned int value)
		{
			normalStrideInBytes = value;
			return *this;
		}

		BuildInputCurves & setIndexBuffer(CUdeviceptr value)
		{
			indexBuffer = value;
			return *this;
		}

		template<typename Index>
		BuildInputCurves & setIndexBuffer(const Index * buffer)
		{
			static_assert(std::is_unsigned_v<detail::RemoveCVRef<Index>>, "Curve index buffer must use an unsigned integer type.");
			indexBuffer = reinterpret_cast<CUdeviceptr>(buffer);
			indexStrideInBytes = sizeof(Index);
			return *this;
		}

		BuildInputCurves & setIndexStrideInBytes(unsigned int value)
		{
			indexStrideInBytes = value;
			return *this;
		}

		BuildInputCurves & setFlag(unsigned int value)
		{
			flag = value;
			return *this;
		}

		BuildInputCurves & setPrimitiveIndexOffset(unsigned int value)
		{
			primitiveIndexOffset = value;
			return *this;
		}

		BuildInputCurves & setEndcapFlags(unsigned int value)
		{
			endcapFlags = value;
			return *this;
		}
	};
#endif

#if OPTIX_VERSION >= 70500
	class BuildInputSphere : public OptixBuildInputSphereArray
	{

	public:

		BuildInputSphere()
		{
			static_cast<OptixBuildInputSphereArray &>(*this) = {};
		}

		BuildInputSphere & setVertexBuffers(const CUdeviceptr * buffers)
		{
			vertexBuffers = buffers;
			return *this;
		}

		template<typename Vertex>
		BuildInputSphere & setVertexBuffers(const CUdeviceptr * buffers)
		{
			vertexBuffers = buffers;
			vertexStrideInBytes = sizeof(Vertex);
			return *this;
		}

		BuildInputSphere & setVertexStrideInBytes(unsigned int value)
		{
			vertexStrideInBytes = value;
			return *this;
		}

		BuildInputSphere & setNumVertices(unsigned int value)
		{
			numVertices = value;
			return *this;
		}

		BuildInputSphere & setRadiusBuffers(const CUdeviceptr * buffers)
		{
			radiusBuffers = buffers;
			return *this;
		}

		template<typename Radius>
		BuildInputSphere & setRadiusBuffers(const CUdeviceptr * buffers)
		{
			radiusBuffers = buffers;
			radiusStrideInBytes = sizeof(Radius);
			return *this;
		}

		BuildInputSphere & setRadiusStrideInBytes(unsigned int value)
		{
			radiusStrideInBytes = value;
			return *this;
		}

		BuildInputSphere & setSingleRadius(bool value)
		{
			singleRadius = value ? 1 : 0;
			return *this;
		}

		BuildInputSphere & setFlags(const unsigned int * value)
		{
			flags = value;
			return *this;
		}

		BuildInputSphere & setNumSbtRecords(unsigned int value)
		{
			numSbtRecords = value;
			return *this;
		}

		BuildInputSphere & setSbtIndexOffsetBuffer(CUdeviceptr buffer)
		{
			sbtIndexOffsetBuffer = buffer;
			return *this;
		}

		template<typename Offset>
		BuildInputSphere & setSbtIndexOffsetBuffer(const Offset * buffer)
		{
			sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(buffer);
			sbtIndexOffsetSizeInBytes = sizeof(Offset);
			sbtIndexOffsetStrideInBytes = sizeof(Offset);
			return *this;
		}

		BuildInputSphere & setSbtIndexOffsetSizeInBytes(unsigned int value)
		{
			sbtIndexOffsetSizeInBytes = value;
			return *this;
		}

		BuildInputSphere & setSbtIndexOffsetStrideInBytes(unsigned int value)
		{
			sbtIndexOffsetStrideInBytes = value;
			return *this;
		}

		BuildInputSphere & setPrimitiveIndexOffset(unsigned int value)
		{
			primitiveIndexOffset = value;
			return *this;
		}
	};
#endif

	class BuildInputInstance : public OptixInstance
	{

	public:

		static constexpr unsigned int FullVisibilityMask = 255u;

		BuildInputInstance()
		{
			static_cast<OptixInstance &>(*this) = {};
			this->setIdentityTransform();
			visibilityMask = FullVisibilityMask;
		}

		BuildInputInstance & setTransform(const float * value)
		{
			std::memcpy(transform, value, sizeof(transform));
			return *this;
		}

		BuildInputInstance & setTransform(const float (&value)[12])
		{
			std::memcpy(transform, value, sizeof(transform));
			return *this;
		}

		BuildInputInstance & setIdentityTransform()
		{
			const float identity[12] = { 1.0f, 0.0f, 0.0f, 0.0f,
										 0.0f, 1.0f, 0.0f, 0.0f,
										 0.0f, 0.0f, 1.0f, 0.0f };
			std::memcpy(transform, identity, sizeof(transform));
			return *this;
		}

		BuildInputInstance & setInstanceId(unsigned int value)
		{
			instanceId = value;
			return *this;
		}

		BuildInputInstance & setSbtOffset(unsigned int value)
		{
			sbtOffset = value;
			return *this;
		}

		BuildInputInstance & setVisibilityMask(unsigned int value)
		{
			visibilityMask = value;
			return *this;
		}

		BuildInputInstance & setFlags(unsigned int value)
		{
			flags = value;
			return *this;
		}

		BuildInputInstance & setTraversableHandle(OptixTraversableHandle value)
		{
			traversableHandle = value;
			return *this;
		}
	};

	/*****************************************************************************
	*****************************    AccelStruct    ******************************
	*****************************************************************************/

	//	An abstract class representing an acceleration structure.
	class AccelStruct
	{

	public:

		//	Default constructor.
		AccelStruct() {}

		//	Virtual destructor.
		virtual ~AccelStruct() {}

	public:

		//	Enum defining different subtypes of acceleration structures.
		enum SubType
		{
			Geometry,		//	Geometry acceleration structure (GAS).
			Instance,		//	Instance acceleration structure (IAS).
		};

		//	Pure virtual function to check if the acceleration structure is empty.
		virtual bool empty() const = 0;

		//	Pure virtual function to check if updates are allowed on the acceleration structure.
		virtual bool allowUpdate() const = 0;

		//	Pure virtual function to retrieve the subtype of the acceleration structure.
		virtual SubType subType() const = 0;

		//	Pure virtual function to retrieve the handle of Optix acceleration structure.
		virtual OptixTraversableHandle handle() const = 0;

		//	Pure virtual function to retrieve the context associated with the acceleration structure.
		virtual std::shared_ptr<class DeviceContext> deviceContext() const = 0;

		//	Pure virtual function to rebuild the acceleration structure.
		virtual void rebuild(ns::Stream & stream) = 0;

		/**
		 *	@brief		Pure virtual function to refit the acceleration structure.
		 *	@warning	Only the device pointers and/or their buffer content may be changed.
		 */
		virtual void refit(ns::Stream & stream) = 0;
	};

	/*****************************************************************************
	***************************    GeomAccelStruct    ****************************
	*****************************************************************************/

	//!	An abstract class representing an geometry acceleration structure.
	class GeomAccelStruct : virtual public AccelStruct
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

		//!	Function to retrieve the subtype of the acceleration structure, indicating it as a geometry type.
		virtual SubType subType() const final { return SubType::Geometry; }

		//!	Virtual function to retrieve the primitive type of the acceleration structure.
		virtual PrimitiveType primitiveType() const = 0;

		//	Pure virtual function to retrieve the total number of SbtRecords.
		virtual unsigned int numSbtRecords() const = 0;

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
		virtual dev::Ptr<unsigned char> headerBuffer() = 0;

		//!	Virtual function to retrieve the size of header in bytes.
		virtual size_t headerSize() const = 0;
	};

	/*****************************************************************************
	*************************    AccelStructTriangle    **************************
	*****************************************************************************/

	class AccelStructTriangle : public GeomAccelStruct
	{

	public:

		//	Returns a constant reference to a vector of OptixBuildInputTriangleArray structures.
		virtual const std::vector<OptixBuildInputTriangleArray> & buildInputs() const = 0;

		//	Function to retrieve the primitive type of the acceleration structure, indicating it as a triangle primitive.
		virtual PrimitiveType primitiveType() const final { return PrimitiveType::Triangle; }

		//	Abstract function to build the acceleration structure from input triangles.
		//	Callers should zero-initialize each OptixBuildInputTriangleArray first
		//	(e.g. OptixBuildInputTriangleArray{}), then populate all required OptiX
		//	fields before calling build(). In particular, initialize the vertex data
		//	(description pointers/count, vertexFormat, and vertexStrideInBytes) and,
		//	when indexed triangles are used, the index data (index buffer/count,
		//	indexFormat, and indexStrideInBytes). Also provide per-input flags /
		//	numSbtRecords as required by the OptiX build input contract.
		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputTriangleArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate) = 0;
	};

	/*****************************************************************************
	***************************    AccelStructAabb    ****************************
	*****************************************************************************/

	class AccelStructAabb : public GeomAccelStruct
	{

	public:

		//	Returns a constant reference to a vector of OptixBuildInputCustomPrimitiveArray structures.
		virtual const std::vector<OptixBuildInputCustomPrimitiveArray> & buildInputs() const = 0;

		//	Function to retrieve the primitive type of the acceleration structure, indicating it as a AABB primitive.
		virtual PrimitiveType primitiveType() const final { return PrimitiveType::AABB; }

		//	Abstract function to build the acceleration structure from input AABBs.
		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCustomPrimitiveArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate) = 0;
	};

	/*****************************************************************************
	***************************    AccelStructCurve    ***************************
	*****************************************************************************/

#if OPTIX_VERSION >= 70100
	/**
	 *	@note		Requires Optix version >= 7.1.0
	 */
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

		//	Returns a constant reference to a vector of OptixBuildInputCurveArray structures.
	#if OPTIX_VERSION >= 70100
		virtual const std::vector<OptixBuildInputCurveArray> & buildInputs() const = 0;
	#else
		#error "AccelStructCurve requires OptiX 7.1 or newer because OptixBuildInputCurveArray is unavailable before OPTIX_VERSION 70100."
	#endif

		//	Function to retrieve the primitive type of the acceleration structure, indicating it as a curve primitive.
		virtual PrimitiveType primitiveType() const final { return PrimitiveType::Curve; }

		//	Abstract function to build the acceleration structure from input curves.
		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputCurveArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate) = 0;
	};
#endif

	/*****************************************************************************
	**************************    AccelStructSphere    ***************************
	*****************************************************************************/

#if OPTIX_VERSION >= 70500
	/**
	 *	@note		Requires Optix version >= 7.5.0
	 */
	class AccelStructSphere : public GeomAccelStruct
	{

	public:

		//	Returns a constant reference to a vector of OptixBuildInputSphereArray structures.
		virtual const std::vector<OptixBuildInputSphereArray> & buildInputs() const = 0;

		//	Function to retrieve the primitive type of the acceleration structure, indicating it as a sphere primitive.
		virtual PrimitiveType primitiveType() const final { return PrimitiveType::Sphere; }

		//	Abstract function to build the acceleration structure from input spheres.
		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixBuildInputSphereArray> buildInputs, size_t headerSize, bool preferFastTrace, bool allowUpdate) = 0;
	};
#endif

	/*****************************************************************************
	***************************    InstAccelStruct    ****************************
	*****************************************************************************/

	class InstAccelStruct : virtual public AccelStruct
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

		/**
		 *	Returns a constant reference to a vector of OptixInstance structures.
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
		virtual const std::vector<OptixInstance> & buildInputs() const = 0;

		//	Function to retrieve the subtype of the acceleration structure, indicating it as a instance type.
		virtual SubType subType() const final { return SubType::Instance; }

		/**
		 *	Abstract function to build the acceleration structure from input instances.
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
		virtual void build(ns::Stream & stream, ns::AllocPtr allocator, ns::ArrayProxy<OptixInstance> buildInputs, bool preferFastTrace, bool allowUpdate) = 0;
	};
}
