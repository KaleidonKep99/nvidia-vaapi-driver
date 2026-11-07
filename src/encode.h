#ifndef NV_ENCODE_H
#define NV_ENCODE_H

#include <va/va.h>
#include <stdbool.h>
#include <stdint.h>

#include "vabackend.h"

//NVENC-backed VA-API encode support. The opaque NVEncoder type is forward
//declared in vabackend.h alongside the other internal object types.

//Driver level: load libnvidia-encode and probe what NVENC can do.
//Returns false when NVENC is unavailable (encode entrypoints are then not
//advertised and decode keeps working as before).
bool nvEncodeDriverInit(NVDriver *drv);
void nvEncodeDriverCleanup(NVDriver *drv);
//Whether nvEncodeDriverInit() found a working NVENC for this profile.
bool nvEncodeProfileSupported(NVDriver *drv, VAProfile profile);
//Answers encode-specific VAConfigAttrib queries. Returns false if the
//attribute is not understood (caller logs it).
bool nvEncodeGetConfigAttribute(NVDriver *drv, VAProfile profile, VAConfigAttrib *attrib);

//Context level.
NVEncoder *nvEncoderCreate(NVContext *ctx);
void nvEncoderDestroy(NVEncoder *encoder);

//Picture pipeline, called from the matching vabackend entrypoints when the
//context's entrypoint is an encode entrypoint.
void nvEncoderBeginPicture(NVEncoder *encoder, VASurfaceID target);
//Copy one rendered parameter buffer into the encoder's staging area.
VAStatus nvEncoderRenderBuffer(NVEncoder *encoder, NVBuffer *buf, VABufferID bufId);
//Record the creation of a coded buffer so it can be used as the fallback
//output target for pictures whose parameters don't name one.
void nvEncoderCodedBufferCreated(NVEncoder *encoder, VABufferID buf);
//Encode the staged picture into its coded buffer.
VAStatus nvEncoderEndPicture(NVEncoder *encoder);

#endif //NV_ENCODE_H
