/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_v2_kernel_arch22.h
 * \brief
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_KERNEL_H
#define QUANT_LIGHTNING_INDEXER_V2_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common_arch22.h"
#include "quant_lightning_indexer_v2_service_vector_arch22.h"
#include "quant_lightning_indexer_v2_service_cube_arch22.h"
#include "../quant_lightning_indexer_v2_metadata.h"
#include "quant_lightning_indexer_v2_candidate_schedule.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
using namespace QLIV2ServiceVec;
using namespace matmul;
using namespace optiling::detail;
using namespace optiling;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息
// 同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;  // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;   // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL; // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    uint32_t actS2SizeOrig = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    bool isNeedLD = false;                // 该基本块是否需要LD
    uint32_t actMBaseSize = 0U;           // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;         // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;        // S2方向循环的尾基本块大小
    uint32_t validS2Len = 0U;
};

template <typename QLIV2T>
class QLIV2Preload {
public:
    __aicore__ inline QLIV2Preload(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale, __gm__ uint8_t *cuSeqlensQ,
                                __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK,
                                __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *outputIdxOffset, __gm__ uint8_t *metadata,
                                __gm__ uint8_t *candidateTopkIndex, __gm__ uint8_t *sparseIndices,
                                __gm__ uint8_t *sparseValues, __gm__ uint8_t *candidateTopkIndexOut,
                                __gm__ uint8_t *workspace, const QLIV2TilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename QLIV2T::queryType;
    using K_T = typename QLIV2T::keyType;
    using OUT_T = typename QLIV2T::outputType;
    static constexpr bool PAGE_ATTENTION = QLIV2T::pageAttention;
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIV2T::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIV2T::keyLayout;

    using MM1_OUT_T = float;

    QLIV2Matmul<QLIV2T> matmulService;
    QLIV2Vector<QLIV2T> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    // 参照 v1: s1BaseSize 固定, mBaseSize = s1BaseSize * gSize (支持 gSize=32/64)
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t S2_BASE_SIZE = 2048;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t LI_QUANT_PRELOAD_TASK_CACHE_SIZE = 2;

    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;
    static constexpr uint32_t ELE_NUM_PER_BLOCK = 16;

    static constexpr int64_t LD_PREFETCH_LEN = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t keyScaleCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    uint64_t candidateOutCoreOffset = 0ULL;
    uint64_t outputIdxOffsetCoreOffset = 0ULL;  // A15
    uint32_t coreZeroEnable = 1U;

    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<half> weightsGm;
    GlobalTensor<uint32_t> metadataGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> candidateTopkIndexInGm;
    GlobalTensor<int32_t> candidateTopkIndexOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> outputIdxOffsetGm;  // A15: 每行输出索引偏移

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    GlobalTensor<uint32_t> cmpResidualKGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;

    QLIV2Common::ConstInfo constInfo{};
    QLIV2Common::LdSplitCoreInfo ldInfo{};
    TempLoopInfo tempLoopInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const QLIV2TilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK,
                                            __gm__ uint8_t *cmpResidualK);
    // ================================Split Core================================
    __aicore__ inline void SplitCore();
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig,
                                                       uint32_t &validS2Len);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessCandidate();
    __aicore__ inline void ProcessGroupedCandidate();
    __aicore__ inline void ProcessPreparedBlock(uint32_t loop,
        QLIV2Common::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE]);
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                            QLIV2Common::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE]);
    __aicore__ inline void ProcessInvalid();
    __aicore__ inline void ProcessDecode();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline uint32_t GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, uint32_t cmpResiduaKLenDims,
                                                  bool isAccumSeq, GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                  GlobalTensor<uint32_t> &cmpResidualKGm, uint32_t defaultSeqLen,
                                                  uint32_t cmpRatio);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size,
                                               uint32_t &actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::InitTilingData(const QLIV2TilingData *__restrict tilingData)
{
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.keyStride0 = tilingData->keyStride0;                     // A11: key 0轴非连续
    constInfo.keyDequantScaleStride0 = tilingData->keyDequantScaleStride0;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.cmpRatio = tilingData->cmpRatio;
    constInfo.outputLayout = Q_LAYOUT_T; // 输出和输入形状一致
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    constInfo.s1BaseSize = S1_BASE_SIZE;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;

    // candidate (two-level topk)
    constInfo.candidateMode = tilingData->candidateMode;
    constInfo.candidateTopkBlocks = tilingData->candidateTopkBlocks;
    constInfo.candidateBlockSize = tilingData->candidateBlockSize;
    if (false /* retired experimental path */) {
        constInfo.s1BaseSize = 1;
        constInfo.mBaseSize = constInfo.gSize;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::InitActualSeqLen(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                                              __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK,
                                                              __gm__ uint8_t *cmpResidualK)
{
    // Q side: cu_seqlens (TND) or seqused (non-TND)
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
        if (cuSeqlensQ != nullptr) {
            constInfo.actualLenQDims = constInfo.batchSize + 1;
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)cuSeqlensQ, constInfo.actualLenQDims);
        } else {
            constInfo.actualLenQDims = 0;
        }
    } else {
        if (sequsedQ != nullptr) {
            constInfo.actualLenQDims = constInfo.batchSize;
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)sequsedQ, constInfo.actualLenQDims);
        } else {
            constInfo.actualLenQDims = 0;
        }
    }
    // K side: cu_seqlens (TND) or seqused (non-TND)
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) {
        if (cuSeqlensK != nullptr) {
            constInfo.actualLenDims = constInfo.batchSize + 1;
            actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)cuSeqlensK, constInfo.actualLenDims);
        } else {
            constInfo.actualLenDims = 0;
        }
    } else {
        if (sequsedK != nullptr) {
            constInfo.actualLenDims = constInfo.batchSize;
            actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)sequsedK, constInfo.actualLenDims);
        } else {
            constInfo.actualLenDims = 0;
        }
    }
    // cmpResidualK获取
    if (cmpResidualK != nullptr) {
        constInfo.cmpResiduaKLenDims = constInfo.batchSize;
        cmpResidualKGm.SetGlobalBuffer((__gm__ uint32_t *)cmpResidualK, constInfo.batchSize);
    } else {
        constInfo.cmpResiduaKLenDims = 0;
    }
}

template <typename QLIV2T>
__aicore__ inline uint32_t QLIV2Preload<QLIV2T>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                                 GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                                 uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq) {
        // TND with cu_seqlens: length[i] = cu_seqlens[i+1] - cu_seqlens[i]
        return actualSeqLengthsGm.GetValue(bIdx + 1) - actualSeqLengthsGm.GetValue(bIdx);
    } else {
        // non-TND with seqused: length[i] = seqused[i]
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename QLIV2T>
__aicore__ inline uint32_t QLIV2Preload<QLIV2T>::GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims,
                                                                    uint32_t cmpResiduaKLenDims, bool isAccumSeq,
                                                                    GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                                    GlobalTensor<uint32_t> &cmpResidualKGm,
                                                                    uint32_t defaultSeqLen, uint32_t cmpRatio)
{
    uint32_t cmpResidualK; // 当前bidx对应的cmpResidualK
    if (cmpResiduaKLenDims == 0) {
        cmpResidualK = 0;
    } else {
        cmpResidualK = cmpResidualKGm.GetValue(bIdx);
    }
    if (actualLenDims == 0) {
        return defaultSeqLen * cmpRatio + cmpResidualK;
    } else if (isAccumSeq) {
        // TND with cu_seqlens: length[i] = cu_seqlens[i+1] - cu_seqlens[i]
        return (actualSeqLengthsGm.GetValue(bIdx + 1) - actualSeqLengthsGm.GetValue(bIdx)) * cmpRatio + cmpResidualK;
    } else {
        return (actualSeqLengthsGm.GetValue(bIdx)) * cmpRatio + cmpResidualK;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size,
                                                                 uint32_t &actS2Size, uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2SizeOrig = GetActualSeqLenKey(bIdx, constInfo.actualLenDims, constInfo.cmpResiduaKLenDims,
                                       constInfo.isAccumSeqS2, actualSeqLengthsGm, cmpResidualKGm, constInfo.kSeqSize,
                                       constInfo.cmpRatio); // 压缩前的actS2Size
    actS2Size = actS2SizeOrig / constInfo.cmpRatio;         // 真实使用的压缩后S2长度
}

template <typename QLIV2T>
__aicore__ inline uint32_t QLIV2Preload<QLIV2T>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                         uint32_t actS2SizeOrig, uint32_t &validS2Len)
{
    if (actS2SizeOrig / constInfo.cmpRatio == 0) {
        validS2Len = 0;
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase =
        static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size); // 压缩前的validS2LenBase
    validS2Len = (static_cast<int32_t>(s1Offset) + validS2LenBase + static_cast<int32_t>(constInfo.s1BaseSize)) /
                 static_cast<int32_t>(constInfo.cmpRatio);
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / constInfo.cmpRatio);
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename QLIV2T>
__aicore__ inline uint32_t QLIV2Preload<QLIV2T>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum;
    uint32_t validS2Len = 0;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig, validS2Len);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间。基本原则：计算每个核最少处理的块数,
// 剩余的部分前面的核每个核多处理一块
template <typename QLIV2T>
__aicore__ void inline QLIV2Preload<QLIV2T>::SplitCore()
{
    constInfo.coreEnable = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_CORE_ENABLE_INDEX, false));
    if (aiCoreIdx != 0) {
        constInfo.bN2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_BN2_START_INDEX, false));
        constInfo.gS1Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_M_START_INDEX, false));
        constInfo.s2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_S2_START_INDEX, false));
    }
    constInfo.bN2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_BN2_END_INDEX, false));
    constInfo.gS1End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_M_END_INDEX, false));
    constInfo.s2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_S2_END_INDEX, false));

    // 如果0核都没有启动，说明所有核都没启动
    coreZeroEnable = metadataGm.GetValue(GetAttrAbsIndex(0, QLI_V2_CORE_ENABLE_INDEX, false));

    // LD 第一个workspace的索引
    uint32_t ldFirstWorkSpaceIndex = GetAttrAbsIndex(aiCoreIdx, QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX, false);
    ldInfo.saveWorkSpaceIdx = metadataGm.GetValue(ldFirstWorkSpaceIndex);
    if ASCEND_IS_AIV {
        uint32_t vecCoreIdx = tmpBlockIdx; // 此时代表vectoe核数
        uint32_t ldCoreEnableIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_CORE_ENABLE_INDEX, true);
        ldInfo.isLdCoreEnable = metadataGm.GetValue(ldCoreEnableIndex);

        if (!ldInfo.isLdCoreEnable) {
            return;
        }

        // LD 参数信息
        uint32_t ldBn2IdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_BN2_IDX_INDEX, true);
        uint32_t ldMIdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_IDX_INDEX, true);
        uint32_t ldWorkspaceIdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_WORKSPACE_IDX_INDEX, true);
        uint32_t ldWorkspaceNumINDEX = GetAttrAbsIndex(vecCoreIdx, QLD_V2_WORKSPACE_NUM_INDEX, true);
        uint32_t ldMstartIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_START_INDEX, true);
        uint32_t ldMNumIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_NUM_INDEX, true);

        ldInfo.bn2Idx = metadataGm.GetValue(ldBn2IdxIndex);
        ldInfo.bIdx = ldInfo.bn2Idx / constInfo.kHeadNum;
        ldInfo.n2Idx = ldInfo.bn2Idx % constInfo.kHeadNum;
        ldInfo.mIdx = metadataGm.GetValue(ldMIdxIndex);
        ldInfo.workspaceIdx = metadataGm.GetValue(ldWorkspaceIdxIndex);
        ldInfo.workspaceNum = metadataGm.GetValue(ldWorkspaceNumINDEX);
        ldInfo.mStart = metadataGm.GetValue(ldMstartIndex);
        ldInfo.mNum = metadataGm.GetValue(ldMNumIndex);
        uint64_t actualSeqQPrefixSum = 0;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (ldInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(ldInfo.bIdx);
        } else { // BSND
            actualSeqQPrefixSum = (ldInfo.bIdx <= 0) ? 0 : ldInfo.bIdx * constInfo.qSeqSize;
        }
        // 搬出Topk的初始偏移地址
        ldInfo.indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount +
            static_cast<uint64_t>(ldInfo.n2Idx) * constInfo.sparseCount +
            static_cast<uint64_t>(ldInfo.mIdx) * constInfo.s1BaseSize * constInfo.kHeadNum * constInfo.sparseCount;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_LAYOUT::TND) {
            uint32_t tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize);
            uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx);
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount + // T轴、s1轴偏移
                    n2Idx * constInfo.sparseCount;                                 // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset =
                    static_cast<uint64_t>(bIdx) * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount +
                    static_cast<uint64_t>(s1Idx) * constInfo.kHeadNum * constInfo.sparseCount + // B轴、S1轴偏移
                    static_cast<uint64_t>(n2Idx) * constInfo.sparseCount;                       // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights, __gm__ uint8_t *queryScale,
    __gm__ uint8_t *keyScale, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedK, __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *blockTable, __gm__ uint8_t *outputIdxOffset,
    __gm__ uint8_t *metadata, __gm__ uint8_t *candidateTopkIndex, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *sparseValues, __gm__ uint8_t *candidateTopkIndexOut, __gm__ uint8_t *workspace,
    const QLIV2TilingData *__restrict tiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK);

    // Benchmark only: mode 4 always uses direct candidate K, including prefill.

    if (false /* retired experimental path */) {
        uint32_t rows = actualSeqLengthsGmQ.GetValue(constInfo.batchSize);
        QLIV2Candidate::Schedule schedule(rows, constInfo.candidateTopkBlocks, GetBlockNum());
        constInfo.coreEnable = aiCoreIdx / schedule.parts < rows;
        if ASCEND_IS_AIV {
            ldInfo.isLdCoreEnable = schedule.parts > 1 && tmpBlockIdx < rows;
            ldInfo.workspaceIdx = tmpBlockIdx * schedule.parts;
            ldInfo.workspaceNum = schedule.parts;
            ldInfo.mNum = 1;
            ldInfo.indiceOutCoreOffset = static_cast<uint64_t>(tmpBlockIdx) * constInfo.sparseCount;
        }
    } else if ((false /* retired experimental path */ || constInfo.candidateMode == CANDIDATE_MODE_PACKED)) {
        constInfo.coreEnable = true;
        ldInfo.isLdCoreEnable = false;
        candidateTopkIndexInGm.SetGlobalBuffer((__gm__ int32_t *)candidateTopkIndex);
        uint32_t groups = candidateTopkIndexInGm.GetValue(6);
        uint32_t parts = Max(1U, GetBlockNum() / groups);
        if ASCEND_IS_AIV {
            uint32_t group = tmpBlockIdx / 2;
            if (parts > 1 && group < groups) {
                uint64_t base = static_cast<uint64_t>(group) * QLIV2Candidate::GROUP_STRIDE;
                uint32_t queries = candidateTopkIndexInGm.GetValue(base + 1);
                ldInfo.mStart = (tmpBlockIdx % 2) * 2;
                if (queries > ldInfo.mStart) {
                    ldInfo.isLdCoreEnable = true;
                    ldInfo.mNum = Min(2U, queries - ldInfo.mStart);
                    ldInfo.workspaceIdx = group * parts;
                    ldInfo.workspaceNum = Min(parts, static_cast<uint32_t>(candidateTopkIndexInGm.GetValue(base + 4)) / 64);
                    ldInfo.indiceOutCoreOffset = static_cast<uint64_t>(candidateTopkIndexInGm.GetValue(base)) * constInfo.sparseCount;
                }
            }
        }
    } else if (metadata != nullptr) {
        metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);
        // 计算分核
        SplitCore();
    }

    pipe = tPipe;
    // workspace 内存排布
    // |mm1ResGm(存S)
    uint64_t offset = 0;

    // mm1开DoubleBuffer
    GlobalTensor<MM1_OUT_T> mm1ResGm; // 存放S
    uint64_t singleCoreMm1ResSize = WS_DOUBLE * constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    GlobalTensor<half> weightWorkspaceGm; // v1阶段处理w*scale后的结果
    uint64_t weightMemSize = BLOCK_CUBE * constInfo.mBaseSize * WS_DOUBLE * sizeof(half);
    weightWorkspaceGm.SetGlobalBuffer((__gm__ half *)(workspace + offset + aiCoreIdx * weightMemSize));
    offset += GetBlockNum() * weightMemSize;

    // ld流程需要ws大小: [aicnum, 2, s1BaseSize, topkOut_*2]
    // (aic, 8, 2, 2, 2048)
    // (aic, s1_cube, 头尾, idx/value, K)
    GlobalTensor<float> vec1ResGm; // 存放TopK计算中间结果
    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * WS_DOUBLE * BASE_TOPK * sizeof(float);

    if (false /* retired experimental path */ ||
        (false /* retired experimental path */ || constInfo.candidateMode == CANDIDATE_MODE_PACKED)) {
        candidateTopkIndexInGm.SetGlobalBuffer((__gm__ int32_t *)candidateTopkIndex);
    }
    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, ldInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        weightsGm.SetGlobalBuffer((__gm__ half *)weights);
        qScaleGm.SetGlobalBuffer((__gm__ half *)queryScale);
        kScaleGm.SetGlobalBuffer((__gm__ half *)keyScale);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        if (((constInfo.candidateMode == CANDIDATE_MODE_CONSUMER || constInfo.candidateMode == CANDIDATE_MODE_LOCAL_CAP || constInfo.candidateMode == CANDIDATE_MODE_CACHED_MASK) ||
             false /* retired experimental path */) && candidateTopkIndex != nullptr) {
            candidateTopkIndexInGm.SetGlobalBuffer((__gm__ int32_t *)candidateTopkIndex);
        }
        if ((constInfo.candidateMode == CANDIDATE_MODE_SOURCE || constInfo.candidateMode == CANDIDATE_MODE_SOURCE_TOP512) && candidateTopkIndexOut != nullptr) {
            candidateTopkIndexOutGm.SetGlobalBuffer((__gm__ int32_t *)candidateTopkIndexOut);
        }
        if (outputIdxOffset != nullptr) {  // A15: 使能 output_idx_offset (参照 arch35)
            outputIdxOffsetGm.SetGlobalBuffer((__gm__ int32_t *)outputIdxOffset);
        }
        // A15: 有效标志须与 tensor 一并传给 vector (InitParams 已先值拷贝 constInfo, 此处再改标志不生效)
        bool outputIdxOffsetValid = (outputIdxOffset != nullptr);
        vectorService.InitVecInputTensor(weightsGm, qScaleGm, kScaleGm, indiceOutGm, blockTableGm);
        vectorService.InitVecCandidateTensor(candidateTopkIndexInGm, candidateTopkIndexOutGm, outputIdxOffsetGm,
                                             outputIdxOffsetValid);
        vectorService.InitVecWorkspaceTensor(weightWorkspaceGm, mm1ResGm, vec1ResGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm, weightWorkspaceGm);
        matmulService.InitCandidateTensor(candidateTopkIndexInGm);
    }
    InitBuffers();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx + 1 == constInfo.bN2End) && (gS1LoopIdx + 1 == tempLoopInfo.gS1LoopEnd);
    uint32_t s2BlockNum;
    uint32_t validS2Len = 0;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig,
                                             tempLoopInfo.validS2Len);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
        tempLoopInfo.validS2Len = tempLoopInfo.actS2Size;
    }
    tempLoopInfo.s2LoopEnd = (isEnd && constInfo.s2End != 0) ? constInfo.s2End : s2BlockNum;
    if (constInfo.s2Start > 0 || tempLoopInfo.s2LoopEnd < s2BlockNum) {
        tempLoopInfo.isNeedLD = true;
    } else {
        tempLoopInfo.isNeedLD = false;
    }
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.validS2Len % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd =
        (bN2LoopIdx + 1 == constInfo.bN2End && constInfo.gS1End != 0) ? constInfo.gS1End : gS1SplitNum;
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx,
                                                         QLIV2Common::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx < tempLoopInfo.s2LoopEnd;
    runInfo.isNeedLD = tempLoopInfo.isNeedLD;
    if (runInfo.isNeedLD && s2LoopIdx + 1 == tempLoopInfo.s2LoopEnd) {
        runInfo.saveWorkSpaceIdx = ldInfo.saveWorkSpaceIdx;
        ldInfo.saveWorkSpaceIdx++;
    }

    if (!runInfo.isValid) {
        return;
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign = QLIV2Common::Align((uint32_t)runInfo.actualSingleProcessSInnerSize,
                                                                    QLIV2Common::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == constInfo.s2Start;
    runInfo.isLastS2InnerLoop = (s2LoopIdx + 1 == tempLoopInfo.s2LoopEnd);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx);
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
        candidateOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.candidateTopkBlocks +
                                 runInfo.n2Idx * constInfo.candidateTopkBlocks;
        // A15: output_idx_offset 布局 [行, N2] (每行每 key 头一个 int32), batch 级前缀
        outputIdxOffsetCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum + runInfo.n2Idx;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) { // T N2 D, cu_seqlens_k
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx);
    } else {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
    }
    uint64_t tndBIdxOffsetForK = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
    keyCoreOffset = tndBIdxOffsetForK + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    keyScaleCoreOffset = (actualSeqKPrefixSum + runInfo.s2Idx * constInfo.s2BaseSize) * constInfo.kHeadNum;
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset;
    runInfo.tensorKeyScaleOffset = keyScaleCoreOffset;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
    runInfo.candidateOutOffset = candidateOutCoreOffset;
    runInfo.outputIdxOffsetCoreOffset = outputIdxOffsetCoreOffset;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::Process()
{
    if ((false /* retired experimental path */ || constInfo.candidateMode == CANDIDATE_MODE_PACKED)) {
        ProcessGroupedCandidate();
        return;
    }
    if (false /* retired experimental path */) {
        ProcessCandidate();
        return;
    }
    // 没有计算任务，直接清理输出
    if (coreZeroEnable == 0) {
        ProcessInvalid();
        return;
    }
    ProcessMain();

    ProcessDecode();
    if ASCEND_IS_AIV {
        if (false /* retired experimental path */ || constInfo.candidateMode == CANDIDATE_MODE_CACHED_MASK) {
            uint32_t rows = actualSeqLengthsGmQ.GetValue(constInfo.batchSize);
            for (uint32_t row = rows + tmpBlockIdx; row < constInfo.qSeqSize; row += GetBlockNum() * 2) {
                GlobalTensor<int32_t> output = indiceOutGm[static_cast<uint64_t>(row) * constInfo.sparseCount];
                InitGlobalMemory(output, constInfo.sparseCount, -1);
            }
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            QLIV2Common::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessMain()
{
    // 无任务核直接返回
    if (constInfo.coreEnable == 0) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
        CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
    }

    QLIV2Common::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE];

    // 适配左闭右开
    if (constInfo.bN2Start == constInfo.bN2End) {
        if (constInfo.gS1Start != constInfo.gS1End || constInfo.s2Start != constInfo.s2End) {
            constInfo.bN2End += 1;
            if (constInfo.s2End != 0) {
                constInfo.gS1End += 1;
            }
        }
    } else if ((constInfo.gS1End != 0) || (constInfo.s2End != 0)) {
        constInfo.bN2End += 1;
        if (constInfo.s2End != 0) {
            constInfo.gS1End += 1;
        }
    }

    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = constInfo.bN2Start; bN2LoopIdx < constInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);

            if ASCEND_IS_AIV {
                if (bN2LoopIdx + 1 == constInfo.bN2End && gloop > 0) {
                    CrossCoreWaitFlag(constInfo.syncC1V1);
                    vectorService.ProcessVec1(runInfo[1 - gloop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE]);
                    CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(
                        constInfo.syncV1C1); // 反向同步 1
                }
            }
            continue;
        }
        for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx < tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            bool isEnd = (bN2LoopIdx + 1 == constInfo.bN2End) && (gS1LoopIdx + 1 == tempLoopInfo.gS1LoopEnd);
            uint32_t extraLoop = isEnd ? LI_QUANT_PRELOAD_TASK_CACHE_SIZE - 1 : 0; // 只preload一轮

            for (uint32_t s2LoopIdx = constInfo.s2Start; s2LoopIdx < (tempLoopInfo.s2LoopEnd + extraLoop);
                 s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            constInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        constInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncC1V0);
        CrossCoreWaitFlag(constInfo.syncC1V0);
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessBaseBlock(
    uint32_t loop, uint64_t s2LoopIdx, QLIV2Common::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE])
{
    CalcRunInfo(loop, s2LoopIdx, runInfo[loop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE]);
    ProcessPreparedBlock(loop, runInfo);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessPreparedBlock(
    uint32_t loop, QLIV2Common::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE])
{
    int32_t curTaskId = loop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE;
    QLIV2Common::RunInfo &curRunInfo = runInfo[curTaskId];
    QLIV2Common::RunInfo &lastRunInfo = runInfo[1 - curTaskId];

    if (curRunInfo.isValid) {
        if ASCEND_IS_AIC {
            if (curRunInfo.isFirstS2InnerLoop) {
                CrossCoreWaitFlag(constInfo.syncV0C1);
            }
            CrossCoreWaitFlag(constInfo.syncV1C1); // 反向同步 1
            matmulService.ComputeMm1(curRunInfo);
            CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
            if (curRunInfo.isLastS2InnerLoop) {
                // 反向同步 0
                CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
            }
        } else {
            if (curRunInfo.isFirstS2InnerLoop) {
                CrossCoreWaitFlag(constInfo.syncC1V0); // 反向同步 0
                vectorService.ProcessVec0(curRunInfo);
                CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
            }
        }
    }

    if (lastRunInfo.isValid) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(constInfo.syncC1V1);
            vectorService.ProcessVec1(lastRunInfo);
            CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1); // 反向同步 1
        }
        lastRunInfo.isValid = false;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessCandidate()
{
    uint32_t rows = actualSeqLengthsGmQ.GetValue(constInfo.batchSize);
    QLIV2Candidate::Schedule schedule(rows, constInfo.candidateTopkBlocks, GetBlockNum());
    uint32_t group = aiCoreIdx / schedule.parts;
    uint32_t part = aiCoreIdx % schedule.parts;
    if (group < rows && group < schedule.groups) {
        if ASCEND_IS_AIV {
            vectorService.AllocEventID();
            CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
            CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        } else {
            matmulService.AllocEventID();
            CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
            CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        }
        RunInfo runs[LI_QUANT_PRELOAD_TASK_CACHE_SIZE];
        uint32_t loop = 0;
        uint32_t request = 0;
        for (uint32_t row = group; row < rows; row += schedule.groups) {
            while (row >= actualSeqLengthsGmQ.GetValue(request + 1)) {
                ++request;
            }
            uint32_t end = actualSeqLengthsGmQ.GetValue(request + 1);
            uint32_t length = GetActualSeqLen(request, constInfo.actualLenDims, false,
                                              actualSeqLengthsGm, constInfo.kSeqSize);
            // All tokens in a verification request have their own causal end.
            int32_t visible = static_cast<int32_t>(length) - static_cast<int32_t>(end - row) + 1;
            if (!constInfo.attenMaskFlag) {
                visible = length;
            }
            for (uint32_t tile = schedule.Begin(part); tile < schedule.End(part); ++tile) {
                RunInfo &info = runs[loop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE];
                info = RunInfo{};
                info.isValid = true;
                info.loop = loop;
                info.bIdx = info.bN2Idx = request;
                info.s2Idx = tile;
                info.actS1Size = 1;
                info.actMBaseSize = constInfo.gSize;
                info.actS2Size = info.actS2SizeOrig = schedule.slots;
                info.candidateVisible = visible > 0 ? visible : 0;
                info.actualSingleProcessSInnerSize = Min(schedule.slots - tile * S2_BASE_SIZE, S2_BASE_SIZE);
                info.actualSingleProcessSInnerSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);
                info.isFirstS2InnerLoop = tile == schedule.Begin(part);
                info.isLastS2InnerLoop = tile + 1 == schedule.End(part);
                info.isNeedLD = schedule.parts > 1;
                info.saveWorkSpaceIdx = row * schedule.parts + part;
                info.tensorQueryOffset = static_cast<uint64_t>(row) * constInfo.gSize * constInfo.headDim;
                info.tensorWeightsOffset = static_cast<uint64_t>(row) * constInfo.gSize;
                info.indiceOutOffset = static_cast<uint64_t>(row) * constInfo.sparseCount;
                info.candidateOutOffset = static_cast<uint64_t>(row) * constInfo.candidateTopkBlocks;
                info.outputIdxOffsetCoreOffset = row;
                ProcessPreparedBlock(loop++, runs);
            }
        }
        runs[loop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE].isValid = false;
        ProcessPreparedBlock(loop, runs); // Drain the final Vector task.
        if ASCEND_IS_AIV {
            vectorService.FreeEventID();
            CrossCoreWaitFlag(constInfo.syncC1V0);
            CrossCoreWaitFlag(constInfo.syncC1V0);
        } else {
            matmulService.FreeEventID();
            CrossCoreWaitFlag(constInfo.syncV1C1);
            CrossCoreWaitFlag(constInfo.syncV1C1);
        }
    }
    ProcessDecode();
    if ASCEND_IS_AIV {
        // TND graph buckets can have unused trailing query rows.
        for (uint32_t row = rows + tmpBlockIdx; row < constInfo.qSeqSize; row += GetBlockNum() * 2) {
            // InitLDBuffers has reset the normal queues; write padding directly.
            GlobalTensor<int32_t> output = indiceOutGm[static_cast<uint64_t>(row) * constInfo.sparseCount];
            InitGlobalMemory(output, constInfo.sparseCount, -1);
        }
    }
}

// Each core owns complete query groups, retaining each row's Top512 across
// all union tiles. Plan construction is a separate device operation and is
// included in end-to-end microbench timing.
template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessGroupedCandidate()
{
    using namespace QLIV2Candidate;
    uint32_t groups = candidateTopkIndexInGm.GetValue(6);
    uint32_t parts = Max(1U, GetBlockNum() / groups);
    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
    }
    RunInfo runs[LI_QUANT_PRELOAD_TASK_CACHE_SIZE];
    uint32_t loop = 0;
    for (uint32_t task = aiCoreIdx; task < groups * parts; task += GetBlockNum()) {
        uint32_t group = task / parts;
        uint32_t part = task % parts;
        uint64_t base = static_cast<uint64_t>(group) * GROUP_STRIDE;
        uint32_t queryCount = candidateTopkIndexInGm.GetValue(base + 1);
        if (queryCount == 0) {
            continue;
        }
        uint32_t row = candidateTopkIndexInGm.GetValue(base);
        uint32_t request = candidateTopkIndexInGm.GetValue(base + 2);
        uint32_t visible = candidateTopkIndexInGm.GetValue(base + 3);
        uint32_t units = candidateTopkIndexInGm.GetValue(base + 4) / 64;
        uint32_t activeParts = Min(parts, units);
        if (part >= activeParts) {
            continue;
        }
        uint32_t partBegin = units * part / activeParts * 64;
        uint32_t partEnd = units * (part + 1) / activeParts * 64;
        uint32_t lastBucket = candidateTopkIndexInGm.GetValue(base + 5);
        for (uint32_t bucket = 0; bucket <= lastBucket; ++bucket) {
            uint32_t firstBlock = candidateTopkIndexInGm.GetValue(base + 8 + bucket * 2);
            uint32_t bucketEnd = firstBlock + candidateTopkIndexInGm.GetValue(base + 9 + bucket * 2);
            uint32_t validEnd = firstBlock + candidateTopkIndexInGm.GetValue(base + 40 + bucket);
            firstBlock = Max(firstBlock, partBegin);
            bucketEnd = Min(bucketEnd, partEnd);
            if (bucketEnd <= firstBlock) {
                continue;
            }
            uint32_t slots = (bucketEnd - firstBlock) * BLOCK;
            // Bucket zero preserves the bounded shortage-fill prefix. All other
            // buckets compute only the query subset in their membership mask.
            uint32_t mask = bucket == 0 ? (1U << queryCount) - 1 : bucket;
            uint32_t active = 0;
            for (uint32_t query = 0; query < queryCount; ++query) {
                active += (mask >> query) & 1U;
            }
            uint32_t tiles = CeilDiv(slots, S2_BASE_SIZE);
            for (uint32_t tile = 0; tile < tiles; ++tile) {
                RunInfo &info = runs[loop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE];
                info = RunInfo{};
                info.isValid = true;
                info.loop = loop;
                info.bIdx = info.bN2Idx = request;
                info.s2Idx = tile;
                info.groupQueries = queryCount;
                info.queryMask = mask;
                info.prefixBucket = bucket == 0;
                info.groupFirst = firstBlock == partBegin && tile == 0;
                info.groupLast = bucketEnd == partEnd && tile + 1 == tiles;
                info.isNeedLD = parts > 1;
                info.saveWorkSpaceIdx = task;
                info.actS1Size = active;
                info.actMBaseSize = active * constInfo.gSize;
                info.actS2Size = info.actS2SizeOrig = slots;
                info.candidateVisible = visible;
                uint32_t tokenCount = Min((Min(validEnd, bucketEnd) - firstBlock) * BLOCK - tile * S2_BASE_SIZE, S2_BASE_SIZE);
                // SortAll accepts 128/256/384/512 then multiples of 512. Keep
                // the metadata's 512-token bucket offsets, but do not compute
                // all of that padding for rare membership combinations.
                info.actualSingleProcessSInnerSize = tokenCount <= 512 ? Align(tokenCount, 128U) : Align(tokenCount, 512U);
                info.actualSingleProcessSInnerSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);
                info.isFirstS2InnerLoop = tile == 0;
                info.isLastS2InnerLoop = tile + 1 == tiles;
                info.tensorQueryOffset = static_cast<uint64_t>(row) * constInfo.gSize * constInfo.headDim;
                info.tensorWeightsOffset = static_cast<uint64_t>(row) * constInfo.gSize;
                info.indiceOutOffset = static_cast<uint64_t>(row) * constInfo.sparseCount;
                info.candidateOutOffset = base + GROUP_HEADER + firstBlock;
                ProcessPreparedBlock(loop++, runs);
            }
        }
    }
    runs[loop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE].isValid = false;
    ProcessPreparedBlock(loop, runs);
    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncC1V0);
        CrossCoreWaitFlag(constInfo.syncC1V0);
        uint32_t rows = actualSeqLengthsGmQ.GetValue(constInfo.batchSize);
        for (uint32_t row = rows + tmpBlockIdx; row < constInfo.qSeqSize; row += GetBlockNum() * 2) {
            GlobalTensor<int32_t> output = indiceOutGm[static_cast<uint64_t>(row) * constInfo.sparseCount];
            InitGlobalMemory(output, constInfo.sparseCount, -1);
        }
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
    if (parts > 1) {
        ProcessDecode();
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (ldInfo.isLdCoreEnable) {
            vectorService.ProcessLD();
        }
    }
}

} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_KERNEL_H
