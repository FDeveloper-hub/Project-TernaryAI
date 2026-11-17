#ifndef TERNARY_STDVIDEO_SHIM_H_
#define TERNARY_STDVIDEO_SHIM_H_
/*
 * Este proyecto no usa ninguna extension de codec de video de Vulkan
 * (VK_KHR_video_encode... / VK_KHR_video_decode...) -- estos tipos
 * existen solo porque vulkan_core.h los referencia incondicionalmente
 * al declarar los structs de esas extensiones. Se dejan como
 * declaraciones minimas: los "Level"/"Profile" son enums reales en el
 * Vulkan Video SDK (por eso van como int32_t, mismo tamano que
 * cualquier enum de C), y el resto son structs que vulkan_core.h
 * SIEMPRE referencia por puntero (verificado con grep sobre el header
 * completo), asi que alcanza con dejarlos incompletos/opacos -- nunca
 * se instancian ni se acceden sus campos desde este proyecto.
 */
#include <stdint.h>

typedef int32_t StdVideoH264LevelIdc;
typedef int32_t StdVideoH264ProfileIdc;
typedef int32_t StdVideoH265LevelIdc;
typedef int32_t StdVideoH265ProfileIdc;
typedef int32_t StdVideoAV1Level;
typedef int32_t StdVideoAV1Profile;
typedef int32_t StdVideoVP9Level;
typedef int32_t StdVideoVP9Profile;

typedef struct StdVideoH264SequenceParameterSet StdVideoH264SequenceParameterSet;
typedef struct StdVideoH264PictureParameterSet StdVideoH264PictureParameterSet;
typedef struct StdVideoEncodeH264SliceHeader StdVideoEncodeH264SliceHeader;
typedef struct StdVideoEncodeH264PictureInfo StdVideoEncodeH264PictureInfo;
typedef struct StdVideoEncodeH264ReferenceInfo StdVideoEncodeH264ReferenceInfo;
typedef struct StdVideoDecodeH264PictureInfo StdVideoDecodeH264PictureInfo;
typedef struct StdVideoDecodeH264ReferenceInfo StdVideoDecodeH264ReferenceInfo;

typedef struct StdVideoH265VideoParameterSet StdVideoH265VideoParameterSet;
typedef struct StdVideoH265SequenceParameterSet StdVideoH265SequenceParameterSet;
typedef struct StdVideoH265PictureParameterSet StdVideoH265PictureParameterSet;
typedef struct StdVideoEncodeH265SliceSegmentHeader StdVideoEncodeH265SliceSegmentHeader;
typedef struct StdVideoEncodeH265PictureInfo StdVideoEncodeH265PictureInfo;
typedef struct StdVideoEncodeH265ReferenceInfo StdVideoEncodeH265ReferenceInfo;
typedef struct StdVideoDecodeH265PictureInfo StdVideoDecodeH265PictureInfo;
typedef struct StdVideoDecodeH265ReferenceInfo StdVideoDecodeH265ReferenceInfo;

typedef struct StdVideoAV1SequenceHeader StdVideoAV1SequenceHeader;
typedef struct StdVideoDecodeAV1PictureInfo StdVideoDecodeAV1PictureInfo;
typedef struct StdVideoDecodeAV1ReferenceInfo StdVideoDecodeAV1ReferenceInfo;
typedef struct StdVideoEncodeAV1DecoderModelInfo StdVideoEncodeAV1DecoderModelInfo;
typedef struct StdVideoEncodeAV1OperatingPointInfo StdVideoEncodeAV1OperatingPointInfo;
typedef struct StdVideoEncodeAV1PictureInfo StdVideoEncodeAV1PictureInfo;
typedef struct StdVideoEncodeAV1ReferenceInfo StdVideoEncodeAV1ReferenceInfo;

typedef struct StdVideoDecodeVP9PictureInfo StdVideoDecodeVP9PictureInfo;

#endif /* TERNARY_STDVIDEO_SHIM_H_ */
