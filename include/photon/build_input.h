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

		static constexpr unsigned int DefaultVisibilityMask = 255u;

		BuildInputInstance()
		{
			static_cast<OptixInstance &>(*this) = {};
			this->setIdentityTransform();
			visibilityMask = DefaultVisibilityMask;
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

}
