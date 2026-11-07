//NVENC-backed encode support: implements the VA-API encode entrypoint
//(VAEntrypointEncSlice) for H.264, HEVC and AV1 on top of the NVIDIA Video
//Encoder SDK (libnvidia-encode), mirroring how the rest of this driver wraps
//NVDEC for decode.
//
//The VAAPI picture flow:
//  vaBeginPicture(target surface)
//  vaRenderPicture(seq/pic/slice/misc parameter buffers)
//  vaEndPicture  ->  copy the surface NV12/P010 planes into a pitched CUDA
//                    buffer, register+map it with NVENC, NvEncEncodePicture,
//                    lock the bitstream and write it into the picture's
//                    VAEncCodedBufferType buffer as one VACodedBufferSegment.
//
//Scope: "basic" encoding. One encode session per VA context, synchronous,
//no B-frames (so output order matches input and no drain is needed), I/P
//only, rate control/GOP/framedata mapped from the standard VA sequence,
//picture and misc parameter buffers.
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

#include <va/va.h>
#include <va/va_enc_h264.h>
#include <va/va_enc_hevc.h>
#include <va/va_enc_av1.h>
#include <ffnvcodec/nvEncodeAPI.h>

#include "encode.h"
#include "vabackend.h"

#define NV_ENCODE_LIB_NAME "libnvidia-encode.so.1"

//Categories of staged parameter records. Each record in the staging buffer is
//stored as [uint32_t size][payload of size bytes] so the parser can walk a
//list of them without knowing the codec up front.
enum EncParamKind {
    ENC_PARAM_SEQ,
    ENC_PARAM_PIC,
    ENC_PARAM_SLICE,
    ENC_PARAM_MISC,
    ENC_PARAM_COUNT,
};

//What an NVENC session needs to be (re)created with.
typedef struct {
    uint32_t width, height;
    NV_ENC_BUFFER_FORMAT format;
    uint32_t gopSize, bitRate, fpsNum, fpsDen, quality;
    int rcMode, qp;
    bool valid;
} EncSetup;

struct _NVEncodeLib {
    void *handle;
    NV_ENCODE_API_FUNCTION_LIST api;
    bool loaded;

    //Probed once at driver init with a throwaway NVENC session.
    bool hasH264;
    bool hasHevcMain;
    bool hasHevcMain10;
    bool hasAv1;
    uint32_t maxWidth;
    uint32_t maxHeight;
};

struct _NVEncoder {
    NVContext *ctx;
    NVDriver *drv;
    NVEncodeLib *lib;

    //Session
    void *session;              //NVENC encoder handle
    bool encoderReady;          //nvEncInitializeEncoder succeeded
    NV_ENC_CONFIG encodeConfig;

    //Input staging: NVENC-owned input buffer. Registering our own pitched
    //CUDADEVICEPTR made NVENC mis-derive the NV12 plane geometry (frames come
    //out vertically squashed on this driver line), so we let NVENC allocate
    //and describe the input, the way FFmpeg's nvenc wrapper does. The surface
    //planes are read back into hostStage (packed layout) and copied into the
    //locked input buffer honouring NVENC's reported pitch.
    NV_ENC_INPUT_PTR inputBuffer;
    unsigned char *hostStage;
    size_t hostStageSize;
    uint32_t inputPitch;        //bytes per luma row, reported by NVENC
    NV_ENC_BUFFER_FORMAT bufferFormat;
    uint32_t encWidth, encHeight;   //size the encoder is initialised at

    //Output
    NV_ENC_OUTPUT_PTR bitstreamBuffer;
    uint32_t frameCount;

    //Per-picture staged parameters
    AppendableBuffer params[ENC_PARAM_COUNT];
    bool pictureActive;
    VASurfaceID targetSurface;

    //Clients typically only send sequence and misc parameter buffers with the
    //first (IDR) picture of a GOP, so cache what they imply and reuse it for
    //later pictures in the same configuration.
    AppendableBuffer cachedSeq;
    bool haveCachedSeq;
    uint32_t cachedFpsNum, cachedFpsDen, cachedQuality, cachedBitrate;
    int cachedQp;
    bool haveMiscCache;

    //Most recent VAEncCodedBufferType buffer created for this context, used
    //when the picture parameter doesn't name its output buffer.
    VABufferID lastCodedBuffer;

    //Settings the current encoder initialisation was derived from, so a
    //picture that needs something different triggers a re-init.
    EncSetup current, wanted;
    bool haveCurrent;
};

// ---------------------------------------------------------------------------
// NVENC library loading and capability probing
// ---------------------------------------------------------------------------

static void nvEncodeLogError(NVEncodeLib *lib, void *session, NVENCSTATUS status, const char *what) {
    const char *detail = NULL;
    if (lib && lib->api.nvEncGetLastErrorString && session) {
        detail = lib->api.nvEncGetLastErrorString(session);
    }
    LOG("NVENC %s failed: %d (%s)", what, (int) status, detail ? detail : "");
}

static void freeAppendable(AppendableBuffer *ab) {
    free(ab->buf);
    ab->buf = NULL;
    ab->size = 0;
    ab->allocated = 0;
}

bool nvEncodeDriverInit(NVDriver *drv) {
    if (drv->encode && drv->encode->loaded) {
        return true;
    }

    NVEncodeLib *lib = drv->encode ? drv->encode : calloc(1, sizeof(NVEncodeLib));
    if (!lib) {
        return false;
    }
    drv->encode = lib;

    lib->handle = dlopen(NV_ENCODE_LIB_NAME, RTLD_NOW | RTLD_LOCAL);
    if (!lib->handle) {
        LOG("Unable to load " NV_ENCODE_LIB_NAME " (%s), encoding will be unavailable", dlerror());
        return false;
    }

    NVENCSTATUS NVENCAPI (*createInstance)(NV_ENCODE_API_FUNCTION_LIST *);
    NVENCSTATUS NVENCAPI (*getMaxVersion)(uint32_t *);
    createInstance = (NVENCSTATUS NVENCAPI (*)(NV_ENCODE_API_FUNCTION_LIST *)) dlsym(lib->handle, "NvEncodeAPICreateInstance");
    getMaxVersion = (NVENCSTATUS NVENCAPI (*)(uint32_t *)) dlsym(lib->handle, "NvEncodeAPIGetMaxSupportedVersion");
    if (!createInstance || !getMaxVersion) {
        LOG("Missing NvEncodeAPI entry points, encoding will be unavailable");
        return false;
    }

    uint32_t maxVersion = 0;
    //Both NVENCAPI_VERSION and NvEncodeAPIGetMaxSupportedVersion() pack major in
    //the low byte and minor in bits 24-31, so they compare directly.
    if (getMaxVersion(&maxVersion) != NV_ENC_SUCCESS || NVENCAPI_VERSION > maxVersion) {
        LOG("Driver NVENC API version %d.%d too old for %d.%d, encoding will be unavailable",
            (int) (maxVersion & 0xFF), (int) (maxVersion >> 24), NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION);
        return false;
    }

    lib->api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    if (createInstance(&lib->api) != NV_ENC_SUCCESS) {
        LOG("NvEncodeAPICreateInstance failed, encoding will be unavailable");
        return false;
    }
    lib->loaded = true;

    //Probe what this GPU's NVENC can do with a short-lived session. All the
    //probe does is enumerate codecs/profiles and query caps, then destroy the
    //session again.
    NV_ENCODE_API_FUNCTION_LIST *api = &lib->api;
    void *session = NULL;
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessionParams = {0};
    sessionParams.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    sessionParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    //For CUDA deviceType the device pointer *is* the CUcontext handle.
    sessionParams.device = drv->cudaContext;
    sessionParams.apiVersion = NVENCAPI_VERSION;

    CHECK_CUDA_RESULT_RETURN(drv->cu->cuCtxPushCurrent(drv->cudaContext), false);

    NVENCSTATUS status = api->nvEncOpenEncodeSessionEx(&sessionParams, &session);
    if (status != NV_ENC_SUCCESS) {
        nvEncodeLogError(lib, NULL, status, "NvEncOpenEncodeSessionEx (probe)");
        drv->cu->cuCtxPopCurrent(NULL);
        return false;
    }

    uint32_t guidCount = 0;
    if (api->nvEncGetEncodeGUIDCount(session, &guidCount) == NV_ENC_SUCCESS && guidCount > 0) {
        GUID *guids = calloc(guidCount, sizeof(GUID));
        uint32_t retrieved = 0;
        if (guids && api->nvEncGetEncodeGUIDs(session, guids, guidCount, &retrieved) == NV_ENC_SUCCESS) {
            for (uint32_t i = 0; i < retrieved; i++) {
                if (memcmp(&guids[i], &NV_ENC_CODEC_H264_GUID, sizeof(GUID)) == 0) {
                    lib->hasH264 = true;
                } else if (memcmp(&guids[i], &NV_ENC_CODEC_HEVC_GUID, sizeof(GUID)) == 0) {
                    lib->hasHevcMain = true;
                } else if (memcmp(&guids[i], &NV_ENC_CODEC_AV1_GUID, sizeof(GUID)) == 0) {
                    lib->hasAv1 = true;
                }
            }
        }
        free(guids);
    }

    //HEVC Main10 needs the 10-bit profile to be present; check the profile
    //list for the codecs we found.
    if (lib->hasHevcMain) {
        uint32_t profileCount = 0;
        if (api->nvEncGetEncodeProfileGUIDCount(session, NV_ENC_CODEC_HEVC_GUID, &profileCount) == NV_ENC_SUCCESS && profileCount > 0) {
            GUID *profiles = calloc(profileCount, sizeof(GUID));
            uint32_t retrieved = 0;
            if (profiles &&
                api->nvEncGetEncodeProfileGUIDs(session, NV_ENC_CODEC_HEVC_GUID, profiles, profileCount, &retrieved) == NV_ENC_SUCCESS) {
                for (uint32_t i = 0; i < retrieved; i++) {
                    if (memcmp(&profiles[i], &NV_ENC_HEVC_PROFILE_MAIN10_GUID, sizeof(GUID)) == 0) {
                        lib->hasHevcMain10 = true;
                    }
                }
            }
            free(profiles);
        }
    }

    GUID probeGuid = NV_ENC_CODEC_H264_GUID;
    if (!lib->hasH264 && lib->hasHevcMain) {
        probeGuid = NV_ENC_CODEC_HEVC_GUID;
    } else if (!lib->hasH264 && !lib->hasHevcMain && lib->hasAv1) {
        probeGuid = NV_ENC_CODEC_AV1_GUID;
    }
    if (lib->hasH264 || lib->hasHevcMain || lib->hasAv1) {
        NV_ENC_CAPS_PARAM capsParam = {0};
        capsParam.version = NV_ENC_CAPS_PARAM_VER;
        int capsValue = 0;
        capsParam.capsToQuery = NV_ENC_CAPS_WIDTH_MAX;
        if (api->nvEncGetEncodeCaps(session, probeGuid, &capsParam, &capsValue) == NV_ENC_SUCCESS && capsValue > 0) {
            lib->maxWidth = (uint32_t) capsValue;
        } else {
            lib->maxWidth = 4096;
        }
        capsValue = 0;
        capsParam.capsToQuery = NV_ENC_CAPS_HEIGHT_MAX;
        if (api->nvEncGetEncodeCaps(session, probeGuid, &capsParam, &capsValue) == NV_ENC_SUCCESS && capsValue > 0) {
            lib->maxHeight = (uint32_t) capsValue;
        } else {
            lib->maxHeight = 4096;
        }
    }

    api->nvEncDestroyEncoder(session);
    drv->cu->cuCtxPopCurrent(NULL);

    LOG("NVENC available: H264=%d HEVC=%d HEVC10=%d AV1=%d (max %ux%u)",
        lib->hasH264, lib->hasHevcMain, lib->hasHevcMain10, lib->hasAv1, lib->maxWidth, lib->maxHeight);
    return true;
}

void nvEncodeDriverCleanup(NVDriver *drv) {
    NVEncodeLib *lib = drv->encode;
    if (!lib) {
        return;
    }
    if (lib->handle) {
        dlclose(lib->handle);
    }
    free(lib);
    drv->encode = NULL;
}

//Codec identity + NVENC profile GUID for a VA profile we support encoding.
typedef struct {
    VAProfile profile;
    GUID codecGuid;
    GUID profileGuid;
    NV_ENC_BUFFER_FORMAT format;
} EncodeProfileInfo;

static bool encodeProfileInfo(VAProfile profile, EncodeProfileInfo *info) {
    memset(info, 0, sizeof(*info));
    switch (profile) {
    case VAProfileH264ConstrainedBaseline:
        info->codecGuid = NV_ENC_CODEC_H264_GUID;
        info->profileGuid = NV_ENC_H264_PROFILE_BASELINE_GUID;
        info->format = NV_ENC_BUFFER_FORMAT_NV12;
        return true;
    case VAProfileH264Main:
        info->codecGuid = NV_ENC_CODEC_H264_GUID;
        info->profileGuid = NV_ENC_H264_PROFILE_MAIN_GUID;
        info->format = NV_ENC_BUFFER_FORMAT_NV12;
        return true;
    case VAProfileH264High:
        info->codecGuid = NV_ENC_CODEC_H264_GUID;
        info->profileGuid = NV_ENC_H264_PROFILE_HIGH_GUID;
        info->format = NV_ENC_BUFFER_FORMAT_NV12;
        return true;
    case VAProfileHEVCMain:
        info->codecGuid = NV_ENC_CODEC_HEVC_GUID;
        info->profileGuid = NV_ENC_HEVC_PROFILE_MAIN_GUID;
        info->format = NV_ENC_BUFFER_FORMAT_NV12;
        return true;
    case VAProfileHEVCMain10:
        info->codecGuid = NV_ENC_CODEC_HEVC_GUID;
        info->profileGuid = NV_ENC_HEVC_PROFILE_MAIN10_GUID;
        info->format = NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
        return true;
    case VAProfileAV1Profile0:
        info->codecGuid = NV_ENC_CODEC_AV1_GUID;
        info->profileGuid = NV_ENC_AV1_PROFILE_MAIN_GUID;
        info->format = NV_ENC_BUFFER_FORMAT_NV12;
        return true;
    default:
        return false;
    }
}

bool nvEncodeProfileSupported(NVDriver *drv, VAProfile profile) {
    NVEncodeLib *lib = drv->encode;
    if (!lib || !lib->loaded) {
        return false;
    }
    EncodeProfileInfo info;
    if (!encodeProfileInfo(profile, &info)) {
        return false;
    }
    if (memcmp(&info.codecGuid, &NV_ENC_CODEC_H264_GUID, sizeof(GUID)) == 0) {
        return lib->hasH264;
    }
    if (memcmp(&info.codecGuid, &NV_ENC_CODEC_HEVC_GUID, sizeof(GUID)) == 0) {
        return profile == VAProfileHEVCMain10 ? lib->hasHevcMain10 : lib->hasHevcMain;
    }
    if (memcmp(&info.codecGuid, &NV_ENC_CODEC_AV1_GUID, sizeof(GUID)) == 0) {
        return lib->hasAv1;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Config attribute reporting
// ---------------------------------------------------------------------------

bool nvEncodeGetConfigAttribute(NVDriver *drv, VAProfile profile, VAConfigAttrib *attrib) {
    NVEncodeLib *lib = drv->encode;
    if (!lib || !lib->loaded) {
        return false;
    }

    switch (attrib->type) {
    case VAConfigAttribRTFormat: {
        //The encode input comes from the same NV12 surfaces the decoder
        //produces; 10-bit profiles are fed P010.
        uint32_t value = VA_RT_FORMAT_YUV420;
        if (profile == VAProfileHEVCMain10) {
            value |= VA_RT_FORMAT_YUV420_10;
        } else if (profile == VAProfileAV1Profile0 && drv->supports16BitSurface) {
            //AV1 Profile0 covers 8 and 10 bit 4:2:0.
            value |= VA_RT_FORMAT_YUV420_10;
        }
        attrib->value = value;
        return true;
    }
    case VAConfigAttribRateControl:
        //CONSTQP/CBR/VBR are the modes NVENC offers on every architecture.
        attrib->value = VA_RC_CQP | VA_RC_CBR | VA_RC_VBR;
        return true;
    case VAConfigAttribEncPackedHeaders:
        //Headers are produced in-band by NVENC; packed header delivery is not
        //implemented, so report none supported.
        attrib->value = 0;
        return true;
    case VAConfigAttribEncInterlaced:
        attrib->value = 0;
        return true;
    case VAConfigAttribEncMaxRefFrames:
        //NVENC takes reference-frame management off the client; advertise
        //basic unidirectional prediction support.
        attrib->value = 1 | (1 << 8);
        return true;
    case VAConfigAttribPredictionDirection:
        //I/P only, no B-frames: forward prediction into past references.
        attrib->value = VA_PREDICTION_DIRECTION_PREVIOUS;
        return true;
    case VAConfigAttribEncSliceStructure:
        //NVENC can emit one slice per row, arbitrary rows, or byte-targeted
        //slices; advertise what our single-slice-per-picture path uses.
        attrib->value = VA_ENC_SLICE_STRUCTURE_ARBITRARY_ROWS | VA_ENC_SLICE_STRUCTURE_EQUAL_MULTI_ROWS;
        return true;
    case VAConfigAttribEncMaxSlices:
        attrib->value = 32768;
        return true;
    case VAConfigAttribEncQualityRange:
        //NVENC presets P1..P7
        attrib->value = 7;
        return true;
    case VAConfigAttribMaxPictureWidth:
        attrib->value = lib->maxWidth ? lib->maxWidth : 4096;
        return true;
    case VAConfigAttribMaxPictureHeight:
        attrib->value = lib->maxHeight ? lib->maxHeight : 4096;
        return true;
    default:
        return false;
    }
}

// ---------------------------------------------------------------------------
// Encoder object lifecycle
// ---------------------------------------------------------------------------

NVEncoder *nvEncoderCreate(NVContext *ctx) {
    NVEncoder *enc = calloc(1, sizeof(NVEncoder));
    if (!enc) {
        return NULL;
    }
    enc->ctx = ctx;
    enc->drv = ctx->drv;
    enc->lib = ctx->drv->encode;
    enc->wanted.qp = -1;
    return enc;
}

void nvEncoderDestroy(NVEncoder *enc) {
    if (!enc) {
        return;
    }
    NV_ENCODE_API_FUNCTION_LIST *api = &enc->lib->api;
    NVDriver *drv = enc->drv;

    if (CHECK_CUDA_RESULT(drv->cu->cuCtxPushCurrent(drv->cudaContext))) {
        //Nothing can be torn down safely without the CUDA context; free the
        //wrapper at least so the objects themselves don't leak harder.
        for (int i = 0; i < ENC_PARAM_COUNT; i++) {
            freeAppendable(&enc->params[i]);
        }
        free(enc);
        return;
    }

    if (enc->inputBuffer) {
        api->nvEncDestroyInputBuffer(enc->session, enc->inputBuffer);
        enc->inputBuffer = NULL;
    }
    if (enc->bitstreamBuffer) {
        api->nvEncDestroyBitstreamBuffer(enc->session, enc->bitstreamBuffer);
        enc->bitstreamBuffer = NULL;
    }
    if (enc->session) {
        api->nvEncDestroyEncoder(enc->session);
        enc->session = NULL;
    }
    free(enc->hostStage);
    enc->hostStage = NULL;

    drv->cu->cuCtxPopCurrent(NULL);

    for (int i = 0; i < ENC_PARAM_COUNT; i++) {
        freeAppendable(&enc->params[i]);
    }
    freeAppendable(&enc->cachedSeq);
    free(enc);
}

void nvEncoderBeginPicture(NVEncoder *enc, VASurfaceID target) {
    for (int i = 0; i < ENC_PARAM_COUNT; i++) {
        enc->params[i].size = 0;
    }
    enc->targetSurface = target;
    enc->pictureActive = true;
}

//Track the coded buffers created for an encode context.
void nvEncoderCodedBufferCreated(NVEncoder *enc, VABufferID buf) {
    if (enc) {
        enc->lastCodedBuffer = buf;
    }
}

static void stageParam(NVEncoder *enc, enum EncParamKind kind, const void *data, uint32_t size) {
    AppendableBuffer *ab = &enc->params[kind];
    appendBuffer(ab, &size, sizeof(size));
    appendBuffer(ab, data, size);
}

VAStatus nvEncoderRenderBuffer(NVEncoder *enc, NVBuffer *buf, VABufferID bufId) {
    if (!enc->pictureActive) {
        return VA_STATUS_ERROR_INVALID_CONTEXT;
    }
    if (buf->ptr == NULL) {
        return VA_STATUS_SUCCESS;
    }

    if (buf->bufferType == VAEncCodedBufferType) {
        //Rendered output buffer: this picture's bitstream destination.
        enc->lastCodedBuffer = bufId;
        return VA_STATUS_SUCCESS;
    }

    const void *data = buf->ptr;
    size_t size = buf->size;

    switch (buf->bufferType) {
    case VAEncSequenceParameterBufferType:
        stageParam(enc, ENC_PARAM_SEQ, data, (uint32_t) size);
        //Keep the newest sequence parameters around: clients usually only
        //send them with the leading picture of each GOP. The record header is
        //a uint32_t byte-count, matching what seqRecord/firstRecord read back.
        {
            uint32_t seqSize = (uint32_t) size;
            freeAppendable(&enc->cachedSeq);
            appendBuffer(&enc->cachedSeq, &seqSize, sizeof(seqSize));
            appendBuffer(&enc->cachedSeq, data, seqSize);
            enc->haveCachedSeq = true;
        }
        break;
    case VAEncPictureParameterBufferType:
        stageParam(enc, ENC_PARAM_PIC, data, (uint32_t) size);
        break;
    case VAEncSliceParameterBufferType:
    case VAEncMacroblockParameterBufferType:
        stageParam(enc, ENC_PARAM_SLICE, data, (uint32_t) size);
        break;
    case VAEncMiscParameterBufferType:
        stageParam(enc, ENC_PARAM_MISC, data, (uint32_t) size);
        break;
    case VAEncPackedHeaderParameterBufferType:
    case VAEncPackedHeaderDataBufferType:
        LOG("Encode packed headers are not supported, ignoring buffer of type %d", buf->bufferType);
        break;
    case VAEncMacroblockMapBufferType:
    case VAEncQPBufferType:
        LOG("Encode buffer type %d not supported, ignoring", buf->bufferType);
        break;
    default:
        LOG("Unhandled encode buffer type: %d", buf->bufferType);
        break;
    }
    return VA_STATUS_SUCCESS;
}

//Pull the staged seq/pic/misc records into an EncSetup + keyframe decision.
typedef struct {
    struct _NVEncoder *enc;
    VAProfile profile;
    EncSetup setup;
    bool keyFrame;
} PictureParse;

static void visitMisc(AppendableBuffer *ab, PictureParse *pp) {
    const VAEncMiscParameterBuffer *misc;
    uint64_t offset = 0;
    while (offset + sizeof(uint32_t) <= ab->size) {
        uint32_t size;
        memcpy(&size, PTROFF(ab->buf, offset), sizeof(size));
        offset += sizeof(size);
        if (size < sizeof(VAEncMiscParameterBuffer) || offset + size > ab->size) {
            break;
        }
        misc = (const VAEncMiscParameterBuffer *) PTROFF(ab->buf, offset);
        offset += size;
        EncSetup *setup = &pp->setup;

        switch ((int) misc->type) {
        case VAEncMiscParameterTypeFrameRate:
            if (size >= sizeof(VAEncMiscParameterBuffer) + sizeof(VAEncMiscParameterFrameRate)) {
                const VAEncMiscParameterFrameRate *fr = (const VAEncMiscParameterFrameRate *) misc->data;
                setup->fpsNum = fr->framerate & 0xFFFF;
                setup->fpsDen = fr->framerate >> 16;
                if (setup->fpsDen == 0) {
                    setup->fpsDen = 1;
                }
                pp->enc->cachedFpsNum = setup->fpsNum;
                pp->enc->cachedFpsDen = setup->fpsDen;
                pp->enc->haveMiscCache = true;
            }
            break;
        case VAEncMiscParameterTypeRateControl:
            if (size >= sizeof(VAEncMiscParameterBuffer) + sizeof(VAEncMiscParameterRateControl)) {
                const VAEncMiscParameterRateControl *rc = (const VAEncMiscParameterRateControl *) misc->data;
                if (rc->bits_per_second) {
                    setup->bitRate = rc->bits_per_second;
                    pp->enc->cachedBitrate = rc->bits_per_second;
                }
                if (rc->initial_qp) {
                    setup->qp = (int) rc->initial_qp;
                    pp->enc->cachedQp = (int) rc->initial_qp;
                }
                pp->enc->haveMiscCache = true;
            }
            break;
        case VAEncMiscParameterTypeQualityLevel:
            if (size >= sizeof(VAEncMiscParameterBuffer) + sizeof(VAEncMiscParameterBufferQualityLevel)) {
                const VAEncMiscParameterBufferQualityLevel *q = (const VAEncMiscParameterBufferQualityLevel *) misc->data;
                setup->quality = q->quality_level;
                pp->enc->cachedQuality = q->quality_level;
                pp->enc->haveMiscCache = true;
            }
            break;
        default:
            LOG_DEBUG("Ignoring unsupported encode misc parameter type: %d", misc->type);
            break;
        }
    }
}

static const void *firstRecord(AppendableBuffer *ab, uint32_t *size) {
    if (ab->size >= sizeof(uint32_t)) {
        uint32_t recSize;
        memcpy(&recSize, ab->buf, sizeof(recSize));
        if (ab->size >= sizeof(uint32_t) + recSize) {
            *size = recSize;
            return PTROFF(ab->buf, sizeof(uint32_t));
        }
    }
    *size = 0;
    return NULL;
}

//Sequence parameters for this picture: freshly staged if the client sent
//them, otherwise the cached set from the start of the GOP.
static const void *seqRecord(NVEncoder *enc, uint32_t *size) {
    const void *rec = firstRecord(&enc->params[ENC_PARAM_SEQ], size);
    if (rec) {
        LOG_DEBUG("seqRecord: fresh staged %u bytes", *size);
        return rec;
    }
    if (enc->haveCachedSeq) {
        rec = firstRecord(&enc->cachedSeq, size);
        LOG_DEBUG("seqRecord: cached %u bytes", *size);
        return rec;
    }
    *size = 0;
    return NULL;
}

static bool setupFromH264(PictureParse *pp) {
    uint32_t seqSize = 0, picSize = 0;
    const void *seqV = seqRecord(pp->enc, &seqSize);
    const void *picV = firstRecord(&pp->enc->params[ENC_PARAM_PIC], &picSize);
    if (!seqV || !picV ||
        seqSize < sizeof(VAEncSequenceParameterBufferH264) ||
        picSize < sizeof(VAEncPictureParameterBufferH264)) {
        return false;
    }
    const VAEncSequenceParameterBufferH264 *seq = seqV;
    const VAEncPictureParameterBufferH264 *pic = picV;
    EncSetup *setup = &pp->setup;

    setup->width = (uint32_t) seq->picture_width_in_mbs * 16;
    setup->height = (uint32_t) seq->picture_height_in_mbs * 16;
    setup->gopSize = seq->intra_idr_period ? seq->intra_idr_period : seq->intra_period;
    if (setup->gopSize == 0 || setup->gopSize > NVENC_INFINITE_GOPLENGTH) {
        setup->gopSize = 250;
    }
    if (seq->bits_per_second && setup->bitRate == 0) {
        setup->bitRate = seq->bits_per_second;
    }
    setup->qp = pic->pic_init_qp;
    setup->format = NV_ENC_BUFFER_FORMAT_NV12;
    pp->keyFrame = pic->pic_fields.bits.idr_pic_flag != 0;
    return true;
}

static bool setupFromHevc(PictureParse *pp, bool is10Bit) {
    uint32_t seqSize = 0, picSize = 0;
    const void *seqV = seqRecord(pp->enc, &seqSize);
    const void *picV = firstRecord(&pp->enc->params[ENC_PARAM_PIC], &picSize);
    if (!seqV || !picV ||
        seqSize < sizeof(VAEncSequenceParameterBufferHEVC) ||
        picSize < sizeof(VAEncPictureParameterBufferHEVC)) {
        return false;
    }
    const VAEncSequenceParameterBufferHEVC *seq = seqV;
    const VAEncPictureParameterBufferHEVC *pic = picV;
    EncSetup *setup = &pp->setup;

    setup->width = seq->pic_width_in_luma_samples;
    setup->height = seq->pic_height_in_luma_samples;
    setup->gopSize = seq->intra_idr_period ? seq->intra_idr_period : seq->intra_period;
    if (setup->gopSize == 0 || setup->gopSize > NVENC_INFINITE_GOPLENGTH) {
        setup->gopSize = 250;
    }
    if (seq->bits_per_second && setup->bitRate == 0) {
        setup->bitRate = seq->bits_per_second;
    }
    setup->qp = pic->pic_init_qp;
    setup->format = is10Bit ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT : NV_ENC_BUFFER_FORMAT_NV12;
    pp->keyFrame = pic->pic_fields.bits.idr_pic_flag != 0;
    return true;
}

static bool setupFromAv1(PictureParse *pp) {
    uint32_t seqSize = 0, picSize = 0;
    const void *seqV = seqRecord(pp->enc, &seqSize);
    const void *picV = firstRecord(&pp->enc->params[ENC_PARAM_PIC], &picSize);
    if (!seqV || !picV ||
        seqSize < sizeof(VAEncSequenceParameterBufferAV1) ||
        picSize < sizeof(VAEncPictureParameterBufferAV1)) {
        return false;
    }
    const VAEncSequenceParameterBufferAV1 *seq = seqV;
    const VAEncPictureParameterBufferAV1 *pic = picV;
    EncSetup *setup = &pp->setup;

    //AV1 picture params carry the frame size; NVENC is initialised per picture.
    setup->width = (uint32_t) pic->frame_width_minus_1 + 1;
    setup->height = (uint32_t) pic->frame_height_minus_1 + 1;
    setup->gopSize = seq->intra_period;
    if (setup->gopSize == 0 || setup->gopSize > NVENC_INFINITE_GOPLENGTH) {
        setup->gopSize = 250;
    }
    if (seq->bits_per_second && setup->bitRate == 0) {
        setup->bitRate = seq->bits_per_second;
    }
    setup->format = seq->seq_fields.bits.bit_depth_minus8 ?
        NV_ENC_BUFFER_FORMAT_YUV420_10BIT : NV_ENC_BUFFER_FORMAT_NV12;
    pp->keyFrame = pic->picture_flags.bits.frame_type == 0;
    return true;
}

//Pull the staged seq/pic/misc records into an EncSetup + keyframe decision.
static bool parsePicture(NVEncoder *enc, PictureParse *pp) {
    memset(pp, 0, sizeof(*pp));
    pp->enc = enc;
    pp->profile = enc->ctx->profile;
    pp->setup.rcMode = enc->ctx->rcMode;
    pp->setup.qp = -1;

    bool ok = false;
    switch (pp->profile) {
    case VAProfileH264ConstrainedBaseline:
    case VAProfileH264Main:
    case VAProfileH264High:
        ok = setupFromH264(pp);
        break;
    case VAProfileHEVCMain:
    case VAProfileHEVCMain10:
        ok = setupFromHevc(pp, pp->profile == VAProfileHEVCMain10);
        break;
    case VAProfileAV1Profile0:
        ok = setupFromAv1(pp);
        break;
    default:
        break;
    }
    if (!ok) {
        LOG("Encode picture parameters are missing or do not match profile %d", pp->profile);
        return false;
    }
    LOG_DEBUG("parsePicture: %ux%u gop %u bitrate %u qp %d key %d",
              pp->setup.width, pp->setup.height, pp->setup.gopSize,
              pp->setup.bitRate, pp->setup.qp, pp->keyFrame);

    //Misc parameters can override seq-derived values. When this picture
    //carries no misc buffers (mid-GOP), reuse the cached settings so the
    //session configuration stays stable.
    if (enc->params[ENC_PARAM_MISC].size == 0 && enc->haveMiscCache) {
        if (enc->cachedFpsNum) {
            pp->setup.fpsNum = enc->cachedFpsNum;
            pp->setup.fpsDen = enc->cachedFpsDen;
        }
        if (enc->cachedBitrate) {
            pp->setup.bitRate = enc->cachedBitrate;
        }
        pp->setup.quality = enc->cachedQuality;
        if (enc->cachedQp > 0) {
            pp->setup.qp = enc->cachedQp;
        }
    } else {
        if (pp->setup.fpsNum == 0) {
            pp->setup.fpsNum = 30;
            pp->setup.fpsDen = 1;
        }
        visitMisc(&enc->params[ENC_PARAM_MISC], pp);
    }

    pp->setup.valid = true;
    return true;
}

static bool setupChanged(const EncSetup *a, const EncSetup *b) {
    return a->width != b->width ||
           a->height != b->height ||
           a->format != b->format ||
           a->gopSize != b->gopSize ||
           a->bitRate != b->bitRate ||
           a->fpsNum != b->fpsNum ||
           a->fpsDen != b->fpsDen ||
           a->quality != b->quality ||
           a->rcMode != b->rcMode ||
           a->qp != b->qp;
}

// ---------------------------------------------------------------------------
// NVENC session management
// ---------------------------------------------------------------------------

static NV_ENC_PARAMS_RC_MODE nvencRcMode(int vaRcMode) {
    switch (vaRcMode & (VA_RC_CQP | VA_RC_CBR | VA_RC_VBR)) {
    case VA_RC_CQP:
        return NV_ENC_PARAMS_RC_CONSTQP;
    case VA_RC_VBR:
        return NV_ENC_PARAMS_RC_VBR;
    default:
        return NV_ENC_PARAMS_RC_CBR;
    }
}

static const GUID *presetForQuality(uint32_t quality) {
    //VA quality levels run 0 (default) .. 7 (best); NVENC presets P1..P7
    //P1 fastest/lowest quality, P7 slowest/best.
    static const GUID *presets[] = {
        &NV_ENC_PRESET_P1_GUID, &NV_ENC_PRESET_P2_GUID, &NV_ENC_PRESET_P3_GUID,
        &NV_ENC_PRESET_P4_GUID, &NV_ENC_PRESET_P5_GUID, &NV_ENC_PRESET_P6_GUID,
        &NV_ENC_PRESET_P7_GUID,
    };
    if (quality == 0 || quality > ARRAY_SIZE(presets)) {
        return &NV_ENC_PRESET_P4_GUID;
    }
    return presets[quality - 1];
}

//Tear down the initialised encoder (keeping the session) so the next picture
//re-creates it with new settings.
static void encoderReset(NVEncoder *enc) {
    NV_ENCODE_API_FUNCTION_LIST *api = &enc->lib->api;
    if (enc->inputBuffer) {
        api->nvEncDestroyInputBuffer(enc->session, enc->inputBuffer);
        enc->inputBuffer = NULL;
    }
    if (enc->bitstreamBuffer) {
        api->nvEncDestroyBitstreamBuffer(enc->session, enc->bitstreamBuffer);
        enc->bitstreamBuffer = NULL;
    }
    free(enc->hostStage);
    enc->hostStage = NULL;
    enc->hostStageSize = 0;
    if (enc->session) {
        api->nvEncDestroyEncoder(enc->session);
        enc->session = NULL;
    }
    enc->encoderReady = false;
    enc->haveCurrent = false;
}

//Create the NVENC encoder for the parsed setup. Assumes the CUDA context is
//already pushed.
static bool encoderInitialise(NVEncoder *enc, const EncSetup *setup) {
    NV_ENCODE_API_FUNCTION_LIST *api = &enc->lib->api;
    NVEncodeLib *lib = enc->lib;
    NVDriver *drv = enc->drv;
    CudaFunctions *cu = drv->cu;

    EncodeProfileInfo pInfo;
    if (!encodeProfileInfo(enc->ctx->profile, &pInfo)) {
        return false;
    }
    NV_ENC_BUFFER_FORMAT format = setup->format;

    if (setup->width == 0 || setup->height == 0) {
        LOG("Encode picture has zero dimensions");
        return false;
    }

    if (!enc->session) {
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS params = {0};
        params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        params.device = drv->cudaContext;
        params.apiVersion = NVENCAPI_VERSION;
        NVENCSTATUS status = api->nvEncOpenEncodeSessionEx(&params, &enc->session);
        if (status != NV_ENC_SUCCESS) {
            nvEncodeLogError(lib, NULL, status, "NvEncOpenEncodeSessionEx");
            enc->session = NULL;
            return false;
        }
    }

    memset(&enc->encodeConfig, 0, sizeof(enc->encodeConfig));
    NV_ENC_CONFIG *cfg = &enc->encodeConfig;
    cfg->version = NV_ENC_CONFIG_VER;
    cfg->profileGUID = pInfo.profileGuid;
    cfg->gopLength = setup->gopSize;
    //Basic path: no B-frames, so output order matches input order and there
    //is never a reordering delay to flush.
    cfg->frameIntervalP = 1;

    cfg->rcParams.version = NV_ENC_RC_PARAMS_VER;
    cfg->rcParams.rateControlMode = nvencRcMode(setup->rcMode);
    if (setup->bitRate) {
        cfg->rcParams.averageBitRate = setup->bitRate;
        cfg->rcParams.maxBitRate = (setup->rcMode & VA_RC_VBR) ? setup->bitRate * 2 : setup->bitRate;
    } else {
        //No bitrate given (e.g. quality-driven CQP-less default): VBR with a
        //constant-ish quality target keeps NVENC happy.
        cfg->rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
        cfg->rcParams.targetQuality = 24;
    }
    if (cfg->rcParams.rateControlMode == NV_ENC_PARAMS_RC_CONSTQP) {
        int qp = setup->qp >= 0 ? setup->qp : 26;
        if (qp > 51) {
            qp = 51;
        }
        cfg->rcParams.constQP.qpInterP = qp;
        cfg->rcParams.constQP.qpInterB = qp;
        cfg->rcParams.constQP.qpIntra = qp;
    }

    NV_ENC_INITIALIZE_PARAMS initParams = {0};
    initParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
    initParams.encodeGUID = pInfo.codecGuid;
    initParams.presetGUID = *presetForQuality(setup->quality);
    initParams.encodeWidth = setup->width;
    initParams.encodeHeight = setup->height;
    initParams.darWidth = setup->width;
    initParams.darHeight = setup->height;
    initParams.frameRateNum = setup->fpsNum ? setup->fpsNum : 30;
    initParams.frameRateDen = setup->fpsDen ? setup->fpsDen : 1;
    initParams.enablePTD = 1;
    initParams.enableEncodeAsync = 0;
    initParams.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
    initParams.encodeConfig = cfg;

    //Codec-specific config.
    if (memcmp(&pInfo.codecGuid, &NV_ENC_CODEC_H264_GUID, sizeof(GUID)) == 0) {
        cfg->encodeCodecConfig.h264Config.idrPeriod = setup->gopSize;
        cfg->encodeCodecConfig.h264Config.repeatSPSPPS = 1;
        cfg->encodeCodecConfig.h264Config.level = NV_ENC_LEVEL_AUTOSELECT;
        cfg->encodeCodecConfig.h264Config.chromaFormatIDC = 1;
    } else if (memcmp(&pInfo.codecGuid, &NV_ENC_CODEC_HEVC_GUID, sizeof(GUID)) == 0) {
        cfg->encodeCodecConfig.hevcConfig.idrPeriod = setup->gopSize;
        cfg->encodeCodecConfig.hevcConfig.repeatSPSPPS = 1;
        cfg->encodeCodecConfig.hevcConfig.level = NV_ENC_LEVEL_AUTOSELECT;
        cfg->encodeCodecConfig.hevcConfig.chromaFormatIDC = 1;
        cfg->encodeCodecConfig.hevcConfig.minCUSize = NV_ENC_HEVC_CUSIZE_AUTOSELECT;
        cfg->encodeCodecConfig.hevcConfig.maxCUSize = NV_ENC_HEVC_CUSIZE_AUTOSELECT;
        cfg->encodeCodecConfig.hevcConfig.inputBitDepth = cfg->encodeCodecConfig.hevcConfig.outputBitDepth =
            format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT ? NV_ENC_BIT_DEPTH_10 : NV_ENC_BIT_DEPTH_8;
    } else {
        cfg->encodeCodecConfig.av1Config.idrPeriod = setup->gopSize;
        cfg->encodeCodecConfig.av1Config.repeatSeqHdr = 1;
        cfg->encodeCodecConfig.av1Config.outputAnnexBFormat = 1;
        cfg->encodeCodecConfig.av1Config.level = NV_ENC_LEVEL_AUTOSELECT;
        cfg->encodeCodecConfig.av1Config.chromaFormatIDC = 1;
        cfg->encodeCodecConfig.av1Config.minPartSize = NV_ENC_AV1_PART_SIZE_AUTOSELECT;
        cfg->encodeCodecConfig.av1Config.maxPartSize = NV_ENC_AV1_PART_SIZE_AUTOSELECT;
        cfg->encodeCodecConfig.av1Config.inputBitDepth = cfg->encodeCodecConfig.av1Config.outputBitDepth =
            format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT ? NV_ENC_BIT_DEPTH_10 : NV_ENC_BIT_DEPTH_8;
    }

    NVENCSTATUS status = api->nvEncInitializeEncoder(enc->session, &initParams);
    if (status != NV_ENC_SUCCESS) {
        nvEncodeLogError(lib, enc->session, status, "NvEncInitializeEncoder");
        goto fail;
    }
    enc->encoderReady = true;
    enc->bufferFormat = format;
    enc->encWidth = setup->width;
    enc->encHeight = setup->height;

    NV_ENC_CREATE_BITSTREAM_BUFFER bsBuf = {0};
    bsBuf.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    status = api->nvEncCreateBitstreamBuffer(enc->session, &bsBuf);
    if (status != NV_ENC_SUCCESS) {
        nvEncodeLogError(lib, enc->session, status, "NvEncCreateBitstreamBuffer");
        goto fail;
    }
    enc->bitstreamBuffer = bsBuf.bitstreamBuffer;

    //Input: NVENC-owned buffer. Registering a client pitched device pointer
    //made NVENC mis-derive the NV12 plane geometry (output came out vertically
    //squashed), so let NVENC allocate it, as FFmpeg's nvenc does.
    const uint32_t bpp = format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT ? 2 : 1;
    const uint32_t alignedHeight = (setup->height + 1u) & ~1u;
    NV_ENC_CREATE_INPUT_BUFFER inBuf = {0};
    inBuf.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
    inBuf.width = setup->width;
    inBuf.height = setup->height;
    inBuf.bufferFmt = format;
    status = api->nvEncCreateInputBuffer(enc->session, &inBuf);
    if (status != NV_ENC_SUCCESS) {
        nvEncodeLogError(lib, enc->session, status, "NvEncCreateInputBuffer");
        goto fail;
    }
    enc->inputBuffer = inBuf.inputBuffer;

    //Host staging buffer sized for the tightest packed layout (bpp*W rows).
    enc->hostStageSize = (size_t) setup->width * bpp * alignedHeight * 3 / 2;
    enc->hostStage = malloc(enc->hostStageSize);
    if (!enc->hostStage) {
        LOG("Unable to allocate encode host staging buffer");
        goto fail;
    }

    LOG("NVENC encoder ready: profile %d, %ux%u, format %d, gop %u, bitrate %u",
        (int) enc->ctx->profile, setup->width, setup->height, (int) format, setup->gopSize, setup->bitRate);
    enc->haveCurrent = true;
    enc->current = *setup;
    return true;

fail:
    encoderReset(enc);
    return false;
}

//Copy the target surface's planes into the host staging buffer (packed NV12/
//P010 layout: luma rows of width*bpp bytes, then chroma rows).
static bool encoderUploadSurface(NVEncoder *enc, BackingImage *img, uint32_t width, uint32_t height) {
    if (!img->arrays[0] || !img->arrays[1]) {
        LOG("Encode target surface backing image has no CUDA arrays");
        return false;
    }
    const bool is16Bit = enc->bufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
    if (is16Bit != (img->format == NV_FORMAT_P010 || img->format == NV_FORMAT_P012)) {
        LOG("Encode surface format %d does not match encoder format %d", (int) img->format, (int) enc->bufferFormat);
        return false;
    }
    const uint32_t bpp = is16Bit ? 2 : 1;
    if ((uint32_t) img->width < width || (uint32_t) img->height < height) {
        LOG("Encode surface is %ux%u, picture is %ux%u", img->width, img->height, width, height);
        return false;
    }

    CudaFunctions *cu = enc->drv->cu;
    const uint32_t stagePitch = width * bpp;
    CUDA_MEMCPY2D yCpy = {
        .srcMemoryType = CU_MEMORYTYPE_ARRAY,
        .srcArray = img->arrays[0],
        .dstMemoryType = CU_MEMORYTYPE_HOST,
        .dstHost = enc->hostStage,
        .dstPitch = stagePitch,
        .WidthInBytes = stagePitch,
        .Height = height,
    };
    CUDA_MEMCPY2D uvCpy = {
        .srcMemoryType = CU_MEMORYTYPE_ARRAY,
        .srcArray = img->arrays[1],
        .dstMemoryType = CU_MEMORYTYPE_HOST,
        .dstHost = enc->hostStage + (size_t) stagePitch * height,
        .dstPitch = ((width + 1u) & ~1u) * bpp,
        .WidthInBytes = ((width + 1u) & ~1u) * bpp,
        .Height = (height + 1u) / 2,
    };

    if (CHECK_CUDA_RESULT(cu->cuMemcpy2DAsync(&yCpy, 0)) ||
        CHECK_CUDA_RESULT(cu->cuMemcpy2DAsync(&uvCpy, 0)) ||
        CHECK_CUDA_RESULT(cu->cuStreamSynchronize(0))) {
        return false;
    }

    if (getenv("NVD_DUMP_INPUT")) {
        static FILE *dump = NULL;
        if (!dump) dump = fopen("/tmp/encode_input.nv12", "wb");
        if (dump) {
            fwrite(enc->hostStage, 1, (size_t) stagePitch * height * 3 / 2, dump);
            fflush(dump);
        }
    }
    return true;
}

//Copy the staged frame into NVENC's locked input buffer, honouring the pitch
//NVENC reports for each plane.
static bool encoderFillInputBuffer(NVEncoder *enc, uint32_t width, uint32_t height) {
    NV_ENCODE_API_FUNCTION_LIST *api = &enc->lib->api;
    const bool is16Bit = enc->bufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
    const uint32_t bpp = is16Bit ? 2 : 1;
    const uint32_t stagePitch = width * bpp;

    NV_ENC_LOCK_INPUT_BUFFER lock = {0};
    lock.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
    lock.inputBuffer = enc->inputBuffer;
    NVENCSTATUS status = api->nvEncLockInputBuffer(enc->session, &lock);
    if (status != NV_ENC_SUCCESS) {
        nvEncodeLogError(enc->lib, enc->session, status, "NvEncLockInputBuffer");
        return false;
    }

    unsigned char *dst = lock.bufferDataPtr;
    const uint32_t dstPitch = lock.pitch;
    const unsigned char *src = enc->hostStage;
    for (uint32_t y = 0; y < height; y++) {
        memcpy(dst + (size_t) y * dstPitch, src + (size_t) y * stagePitch, stagePitch);
    }
    const uint32_t chromaW = ((width + 1u) & ~1u) * bpp;
    const unsigned char *srcUV = src + (size_t) stagePitch * height;
    unsigned char *dstUV = dst + (size_t) dstPitch * height;
    for (uint32_t y = 0; y < (height + 1u) / 2; y++) {
        memcpy(dstUV + (size_t) y * dstPitch, srcUV + (size_t) y * chromaW, chromaW);
    }

    api->nvEncUnlockInputBuffer(enc->session, enc->inputBuffer);
    return true;
}

//Write the encoded bitstream into a VA coded buffer (a VACodedBufferSegment
//linked list) so the client can pick it up through vaMapBuffer().
static bool writeCodedBuffer(NVBuffer *coded, const uint8_t *data, uint32_t size, VAStatus status) {
    VACodedBufferSegment *seg = (VACodedBufferSegment *) coded->ptr;
    if (!seg) {
        return false;
    }
    if (size > coded->capacity) {
        void *grown = realloc(seg->buf, size ? size : 1);
        if (!grown) {
            LOG("Unable to grow coded buffer to %u bytes", size);
            size = 0;
            status = VA_CODED_BUF_STATUS_BAD_BITSTREAM;
        } else {
            seg->buf = grown;
            coded->capacity = size;
        }
    }
    if (size) {
        memcpy(seg->buf, data, size);
    }
    seg->size = size;
    seg->bit_offset = 0;
    seg->status = status;
    seg->next = NULL;
    return true;
}

static VAStatus encoderEncodePicture(NVEncoder *enc, PictureParse *pp, VABufferID codedBufId) {
    NV_ENCODE_API_FUNCTION_LIST *api = &enc->lib->api;
    NVEncodeLib *lib = enc->lib;
    VAStatus ret = VA_STATUS_ERROR_OPERATION_FAILED;
    bool bsLocked = false;

    NVSurface *surface = nvGetSurface(enc->drv, enc->targetSurface);
    if (!surface) {
        LOG("Encode target surface is gone");
        return VA_STATUS_ERROR_INVALID_SURFACE;
    }

    NVBuffer *coded = nvGetBuffer(enc->drv, codedBufId);
    if (!coded || coded->bufferType != VAEncCodedBufferType) {
        LOG("No VAEncCodedBufferType buffer for encode picture");
        return VA_STATUS_ERROR_INVALID_BUFFER;
    }

    //The surface must hold decoded/VPP'd/uploaded data before we read it.
    nvSurfaceWaitResolved(surface);

    pthread_mutex_lock(&enc->drv->objectCreationMutex);
    BackingImage *img = surface->backingImage;
    if (img == NULL) {
        pthread_mutex_unlock(&enc->drv->objectCreationMutex);
        LOG("Encode target surface has no backing image");
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }

    //Sequence parameters report the coded size, which codecs round up to
    //their block size (e.g. 720 -> 736 for HEVC CTUs). The surface only holds
    //the real frame, so encode at the surface dimensions; NVENC re-pads and
    //signals the display size itself.
    if (pp->setup.width > (uint32_t) img->width) {
        pp->setup.width = img->width;
    }
    if (pp->setup.height > (uint32_t) img->height) {
        pp->setup.height = img->height;
    }

    const EncSetup *setup = &pp->setup;
    if (!enc->encoderReady || !enc->haveCurrent || setupChanged(&enc->current, setup)) {
        if (enc->encoderReady) {
            LOG("Encoder settings changed, re-initialising");
            encoderReset(enc);
        }
        if (!encoderInitialise(enc, setup)) {
            pthread_mutex_unlock(&enc->drv->objectCreationMutex);
            return VA_STATUS_ERROR_OPERATION_FAILED;
        }
    }

    bool uploaded = encoderUploadSurface(enc, img, setup->width, setup->height);
    pthread_mutex_unlock(&enc->drv->objectCreationMutex);
    if (!uploaded) {
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }

    if (!encoderFillInputBuffer(enc, setup->width, setup->height)) {
        return ret;
    }

    NV_ENC_PIC_PARAMS picParams = {0};
    picParams.version = NV_ENC_PIC_PARAMS_VER;
    picParams.inputWidth = setup->width;
    picParams.inputHeight = setup->height;
    //Pitch is implicit for an NVENC-owned input buffer.
    picParams.inputPitch = 0;
    picParams.inputBuffer = enc->inputBuffer;
    picParams.outputBitstream = enc->bitstreamBuffer;
    picParams.bufferFmt = enc->bufferFormat;
    picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    picParams.frameIdx = enc->frameCount;
    picParams.inputTimeStamp = enc->frameCount;
    if (pp->keyFrame) {
        picParams.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
    }

    NVENCSTATUS status = api->nvEncEncodePicture(enc->session, &picParams);
    if (status != NV_ENC_SUCCESS) {
        nvEncodeLogError(lib, enc->session, status, "NvEncEncodePicture");
        goto cleanup;
    }
    enc->frameCount++;

    NV_ENC_LOCK_BITSTREAM lock = {0};
    lock.version = NV_ENC_LOCK_BITSTREAM_VER;
    lock.outputBitstream = enc->bitstreamBuffer;
    lock.doNotWait = 0;
    status = api->nvEncLockBitstream(enc->session, &lock);
    if (status != NV_ENC_SUCCESS) {
        nvEncodeLogError(lib, enc->session, status, "NvEncLockBitstream");
        goto cleanup;
    }
    bsLocked = true;

    //hwEncodeStatus is unreliable in sync mode: on this driver it reports a
    //non-zero status even while producing a valid, decodable bitstream. Only
    //treat a picture as failed when no bytes came out.
    if (lock.hwEncodeStatus != NV_ENC_SUCCESS) {
        LOG_DEBUG("NVENC reported hwEncodeStatus %u for frame %u (%u bytes produced)",
                  lock.hwEncodeStatus, lock.frameIdx, lock.bitstreamSizeInBytes);
    }

    if (lock.bitstreamSizeInBytes == 0 || lock.bitstreamBufferPtr == NULL) {
        LOG("NVENC produced an empty bitstream for frame %u (hwEncodeStatus %u)",
            lock.frameIdx, lock.hwEncodeStatus);
        writeCodedBuffer(coded, NULL, 0, VA_CODED_BUF_STATUS_BAD_BITSTREAM);
        goto cleanup;
    }

    if (!writeCodedBuffer(coded, (const uint8_t *) lock.bitstreamBufferPtr, lock.bitstreamSizeInBytes, 0)) {
        goto cleanup;
    }
    LOG_DEBUG("Encoded frame %u: %u bytes, type %d", enc->frameCount - 1,
              lock.bitstreamSizeInBytes, (int) lock.pictureType);

    ret = VA_STATUS_SUCCESS;

cleanup:
    if (bsLocked) {
        api->nvEncUnlockBitstream(enc->session, enc->bitstreamBuffer);
    }
    return ret;
}

VAStatus nvEncoderEndPicture(NVEncoder *enc) {
    if (!enc->pictureActive) {
        return VA_STATUS_ERROR_INVALID_CONTEXT;
    }
    enc->pictureActive = false;

    PictureParse pp;
    if (!parsePicture(enc, &pp)) {
        return VA_STATUS_ERROR_INVALID_PARAMETER;
    }

    //Where does the bitstream go? The picture parameter names a coded buffer;
    //fall back to the most recently created coded buffer of the context.
    VABufferID codedBufId = enc->lastCodedBuffer;
    uint32_t picSize = 0;
    const void *pic = firstRecord(&enc->params[ENC_PARAM_PIC], &picSize);
    switch (enc->ctx->profile) {
    case VAProfileH264ConstrainedBaseline:
    case VAProfileH264Main:
    case VAProfileH264High:
        if (picSize >= sizeof(VAEncPictureParameterBufferH264)) {
            codedBufId = ((const VAEncPictureParameterBufferH264 *) pic)->coded_buf;
        }
        break;
    case VAProfileHEVCMain:
    case VAProfileHEVCMain10:
        if (picSize >= sizeof(VAEncPictureParameterBufferHEVC)) {
            codedBufId = ((const VAEncPictureParameterBufferHEVC *) pic)->coded_buf;
        }
        break;
    case VAProfileAV1Profile0:
        if (picSize >= sizeof(VAEncPictureParameterBufferAV1)) {
            codedBufId = ((const VAEncPictureParameterBufferAV1 *) pic)->coded_buf;
        }
        break;
    default:
        break;
    }

    CHECK_CUDA_RESULT_RETURN(enc->drv->cu->cuCtxPushCurrent(enc->drv->cudaContext), VA_STATUS_ERROR_OPERATION_FAILED);
    VAStatus ret = encoderEncodePicture(enc, &pp, codedBufId);
    enc->drv->cu->cuCtxPopCurrent(NULL);
    return ret;
}
