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
 * \file quant_lightning_indexer_v2_service_vector_arch22.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
#define QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
#include "kernel_operator.h"
#include "quant_lightning_indexer_v2_candidate_schedule.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common_arch22.h"
#include "quant_lightning_indexer_v2_vector.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
using namespace QLIV2ServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t BASE_TOPK_VALUE_IDX_SIZE = 4096;
constexpr uint32_t ELE_NUM_32 = 32;
constexpr uint32_t ELE_NUM_128 = 128;
constexpr uint32_t ELE_NUM_512 = 512;

template <typename QLIV2T>
class QLIV2Vector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIV2T::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIV2T::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIV2T::pageAttention;
    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline QLIV2Vector(){};
    __aicore__ inline void ProcessVec0(const QLIV2Common::RunInfo &info);
    __aicore__ inline void ProcessVec1(const QLIV2Common::RunInfo &info);
    __aicore__ inline void ProcessCandidateVec1(const QLIV2Common::RunInfo &info);
    __aicore__ inline void ProcessGroupedVec1(const QLIV2Common::RunInfo &info);
    __aicore__ inline void FinishGrouped(const QLIV2Common::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLIV2Common::ConstInfo &constInfo,
                                      const struct QLIV2Common::LdSplitCoreInfo &ldInfo,
                                      const QLIV2TilingData *__restrict tilingData);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm, GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                  GlobalTensor<float> vec1ResGm);
    __aicore__ inline void InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexInGm,
                                                  GlobalTensor<int32_t> candidateTopkIndexOutGm,
                                                  GlobalTensor<int32_t> outputIdxOffsetGm,
                                                  bool outputIdxOffsetValid);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<half> weightsGm, GlobalTensor<half> qScaleGm,
                                              GlobalTensor<half> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline int32_t AlignS2(int32_t cuS2Len);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<half> weightsGm;
    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    GlobalTensor<half> vec0OutGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> candidateTopkIndexInGm;
    GlobalTensor<int32_t> candidateTopkIndexOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================

private:
    __aicore__ inline void GetKeyScale(const QLIV2Common::RunInfo &runInfo, const LocalTensor<half> &resUb,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // candidate (two-level topk)
    __aicore__ inline int32_t CountGE(const LocalTensor<float> &sortedDesc, int32_t n, float x);
    __aicore__ inline void BuildCandidateMask(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                              int32_t cuBaseS2Idx, int32_t innerS1Idx);
    __aicore__ inline void ProcessCandBlockTopk(const QLIV2Common::RunInfo &info, int32_t cuS1Idx, int32_t cuS2Len,
                                                int32_t cuS2LenVecAlign, int32_t cuRealAcSeq, int32_t innerS1Idx);
    __aicore__ inline void CopyOutCandTopkIndex(const QLIV2Common::RunInfo &info, int32_t cuS1Idx, int32_t innerS1Idx);
    // candidate 新增向量 op 的 64 元素分块封装 (本版本 Level-2 count > 64 的 vmax 等指令会触发 aicore 异常)
    __aicore__ inline void VecMaxPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    __aicore__ inline void VecMinPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    __aicore__ inline void VecAbs(const LocalTensor<float> &dst, const LocalTensor<float> &src, int32_t n);
    __aicore__ inline void VecMinsScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src, float v,
                                         int32_t n);
    __aicore__ inline void VecMaxsScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src, float v,
                                         int32_t n);
    __aicore__ inline void VecSubPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    __aicore__ inline void VecAddsScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src, float v,
                                         int32_t n);
    __aicore__ inline void VecMulPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    // tmp buff for candidate (two-level topk)
    TBuf<TPosition::VECCALC> blockSortOutBuf_; // mode=1: 块级 topk 累加器 [CeilDiv(s1BaseSize,2), 2048, 2]
    TBuf<TPosition::VECCALC> directCandidatesBuf_;
    TBuf<TPosition::VECCALC> groupPlanBuf_;
    TBuf<TPosition::VECCALC> groupScaleBuf_;
    TBuf<TPosition::VECCALC> groupLaneBuf_;
    TBuf<TPosition::VECCALC> candBuf_;         // mode=2: 排序后的候选块索引 [2048, 2] (value+idx)
    TBuf<TPosition::VECCALC> candConstBuf_;    // mode=1/2: negHuge 常量 [2048]

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> globalBlockTopkUb_;
    LocalTensor<float> candIsOut_;       // mode=2: position 级 0/1 (1=候选外), 仅用于分数降级 (R11 leak)
    LocalTensor<float> candNegHuge_;     // -1e30 常量
    GlobalTensor<int32_t> outputIdxOffsetGm_;        // A15: 每行输出索引偏移 (仅 sparse_indices)
    bool isOutputIdxOffsetValid_ = false;            // A15: offset 是否传入 (经 InitVecCandidateTensor 传递)

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    uint32_t retainedTopk_ = BASE_TOPK;
    uint32_t retainedPairSize_ = BASE_TOPK_VALUE_IDX_SIZE;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;

    // para for LD
    uint32_t mrgListNum_ = 4;

    struct QLIV2Common::ConstInfo constInfo_;
    struct QLIV2Common::LdSplitCoreInfo ldInfo_;
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::GetKeyScale(const QLIV2Common::RunInfo &runInfo,
                                                        const LocalTensor<half> &resUb, int64_t batchId,
                                                        int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    if constexpr (PAGE_ATTENTION) {
        // A11: k_scale 0 轴非连续 — 块基址 = 表值 x keyDequantScaleStride0 (真实 stride);
        // 0 时兜底原紧凑公式 (kCacheBlockSize_), 现网行为不变 (R8)
        // Paged hybrid caches can span more than 2^31 scale elements. Promote
        // before multiplication so high physical block IDs do not wrap.
        int64_t kScaleBlkStride = constInfo_.keyDequantScaleStride0 != 0 ?
                                      static_cast<int64_t>(constInfo_.keyDequantScaleStride0) : kCacheBlockSize_;
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        int32_t blockTableBatchOffset = batchId * maxBlockNumPerBatch_;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(half);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb,
                                 kScaleGm[blockId * kScaleBlkStride + startBlockTableOffset], copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(half);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(half);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb[resUbBaseOffset + i * kCacheBlockSize_],
                                 kScaleGm[blockId * kScaleBlkStride], copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(half);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(resUb, kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inQueue_, 2, s2BaseSize_ * sizeof(float) * 2);                                    // 32KB
    pipe->InitBuffer(outQueue_, 1, BASE_TOPK * sizeof(float));                                         // 8 KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                        // 8 KB
    pipe->InitBuffer(tmpBuf_, 64 * 1024);                                                              // 64KB
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * retainedPairSize_ * sizeof(float));

    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    globalTopkNum_ = 0;

    // candidate (two-level topk) 按模式分配, mode=3 不占用额外 UB
    // UB 预算 (192KB): 基础 144KB + mode=1 blockSortOutBuf 32KB = 176KB / mode=2 candBuf 16KB + candConstBuf 8KB = 168KB
    if ((constInfo_.candidateMode == CANDIDATE_MODE_SOURCE || constInfo_.candidateMode == CANDIDATE_MODE_SOURCE_TOP512)) {
        pipe->InitBuffer(blockSortOutBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE * sizeof(float)); // 32KB
        globalBlockTopkUb_ = blockSortOutBuf_.Get<float>();
        InitSortOutBuf(globalBlockTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);
    } else if ((constInfo_.candidateMode == CANDIDATE_MODE_CONSUMER || constInfo_.candidateMode == CANDIDATE_MODE_LOCAL_CAP || constInfo_.candidateMode == CANDIDATE_MODE_CACHED_MASK) ||
               false /* retired experimental path */) {
        // R6 修复: candBuf 按行分区 (每 AIV 处理 CeilDiv(s1BaseSize,2)=2 行, 行间 tile0 重排序
        // 会互相覆盖) — 2 行 x 2048 对 x 8B = 32KB; mode=2 UB 预算 144+32+8=184KB <= 192KB
        pipe->InitBuffer(candBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2 * sizeof(float));
        pipe->InitBuffer(candConstBuf_, BASE_TOPK * sizeof(float)); // 8KB: negHuge
        candNegHuge_ = candConstBuf_.Get<float>();
        Duplicate(candNegHuge_.template ReinterpretCast<int32_t>(), QLIV2ServiceVec::NEG_HUGE_F32, BASE_TOPK);
        PipeBarrier<PIPE_V>();
    }

    if (false /* retired experimental path */) {
        pipe->InitBuffer(directCandidatesBuf_, constInfo_.candidateTopkBlocks * sizeof(int32_t));
    }
    if ((false /* retired experimental path */ || constInfo_.candidateMode == CANDIDATE_MODE_PACKED)) {
        pipe->InitBuffer(groupPlanBuf_, 3 * (s2BaseSize_ / QLIV2Candidate::BLOCK) * sizeof(int32_t));
        pipe->InitBuffer(groupScaleBuf_, s2BaseSize_ * sizeof(float));
        pipe->InitBuffer(groupLaneBuf_, s2BaseSize_ * sizeof(int32_t));
    }
    // 基本块执行前初始化UB和GM
    // step1. 初始化一个有序索引 0 - s2BaseSize_
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    if (false /* retired experimental path */ ||
        (false /* retired experimental path */ || constInfo_.candidateMode == CANDIDATE_MODE_PACKED)) {
        // Scale DMA writes eight FP16 values into each 32-byte UB block. Gather
        // offsets discard those eight FP16 padding values after conversion to float.
        // A2 Vector Add supports int32_t, not uint32_t. Keep shifts and their
        // scalar operands in the same type; Gather reinterprets the offsets.
        LocalTensor<int32_t> offsets = globalTopkIndice_;
        LocalTensor<int32_t> groups = tmpBuf_.Get<int32_t>();
        PipeBarrier<PIPE_V>();
        ShiftRight(groups, offsets, 3, s2BaseSize_);
        PipeBarrier<PIPE_V>();
        ShiftLeft(groups, groups, 3, s2BaseSize_);
        PipeBarrier<PIPE_V>();
        if ((false /* retired experimental path */ || constInfo_.candidateMode == CANDIDATE_MODE_PACKED)) {
            Sub(groupLaneBuf_.Get<int32_t>(), offsets, groups, s2BaseSize_);
            PipeBarrier<PIPE_V>();
        }
        Add(offsets, offsets, groups, s2BaseSize_);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, 2, s2BaseSize_);
        PipeBarrier<PIPE_V>();
    }
    // Running token TopK stores only the final requested K for optimized consumers.
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * retainedPairSize_);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, retainedPairSize_ * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldTmpBuf_, retainedPairSize_ * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldOutValueBuf_, retainedTopk_ * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, retainedTopk_ * sizeof(int32_t));
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitParams(const struct QLIV2Common::ConstInfo &constInfo,
                                                       const struct QLIV2Common::LdSplitCoreInfo &ldInfo,
                                                       const QLIV2TilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    this->ldInfo_ = ldInfo;
    // TopK(A union B) = TopK(TopK(A) union TopK(B)). Candidate blocks
    // remain 2048; only the consumer's token-score accumulators shrink.
    if (constInfo.candidateMode == CANDIDATE_MODE_SOURCE_TOP512 ||
        (false /* retired experimental path */ || constInfo.candidateMode == CANDIDATE_MODE_PACKED) ||
        false /* retired experimental path */ ||
        false /* retired experimental path */ ||
        ((constInfo.candidateMode == CANDIDATE_MODE_CONSUMER || constInfo.candidateMode == CANDIDATE_MODE_LOCAL_CAP || constInfo.candidateMode == CANDIDATE_MODE_CACHED_MASK) && constInfo.sparseCount == ELE_NUM_512)) {
        retainedTopk_ = constInfo.sparseCount;
        retainedPairSize_ = retainedTopk_ * VALUE_AND_INDEX_NUM;
    }
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize; // 4
    s2BaseSize_ = constInfo.s2BaseSize; // 2048
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecInputTensor(GlobalTensor<half> weightsGm,
                                                               GlobalTensor<half> qScaleGm, GlobalTensor<half> kScaleGm,
                                                               GlobalTensor<int32_t> indiceOutGm,
                                                               GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm,
                                                                   GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                                   GlobalTensor<float> vec1ResGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec0OutGm = vec0OutGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexInGm,
                                                                   GlobalTensor<int32_t> candidateTopkIndexOutGm,
                                                                   GlobalTensor<int32_t> outputIdxOffsetGm,
                                                                   bool outputIdxOffsetValid)
{
    this->candidateTopkIndexInGm = candidateTopkIndexInGm;
    this->candidateTopkIndexOutGm = candidateTopkIndexOutGm;
    this->outputIdxOffsetGm_ = outputIdxOffsetGm;  // A15
    this->isOutputIdxOffsetValid_ = outputIdxOffsetValid;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::AllocEventID()
{}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::FreeEventID()
{}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    QLIV2ServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);
    // mode=1: candidate_topk_index 同步填 -1 (行偏移 = sparse 偏移 / sparseCount * candidateTopkBlocks)
    if ((constInfo_.candidateMode == CANDIDATE_MODE_SOURCE || constInfo_.candidateMode == CANDIDATE_MODE_SOURCE_TOP512)) {
        uint64_t candOffset = static_cast<uint64_t>(invalidS1offset) / constInfo_.sparseCount *
                              constInfo_.candidateTopkBlocks;
        LocalTensor<float> candULocal = outQueue_.AllocTensor<float>();
        LocalTensor<int32_t> candIdxLocal = candULocal.template ReinterpretCast<int32_t>();
        Duplicate(candIdxLocal, constInfo_.INVALID_IDX, constInfo_.candidateTopkBlocks);
        outQueue_.EnQue<float>(candULocal);
        candULocal = outQueue_.DeQue<float>();
        QLIV2ServiceVec::CopyOut(candidateTopkIndexOutGm[candOffset], candIdxLocal, constInfo_.candidateTopkBlocks);
        outQueue_.FreeTensor(candULocal);
    }
}


// 64 元素分块封装: 规避 Level-2 大 count 的 vmax/vmin/vabs 等指令异常
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMaxPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Max(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMinPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Min(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecAbs(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                   int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Abs(dst[off], src[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMinsScalar(const LocalTensor<float> &dst,
                                                          const LocalTensor<float> &src, float v, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Mins(dst[off], src[off], v, 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMaxsScalar(const LocalTensor<float> &dst,
                                                          const LocalTensor<float> &src, float v, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Maxs(dst[off], src[off], v, 64);
    }
    PipeBarrier<PIPE_V>();
}


template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecSubPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Sub(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecAddsScalar(const LocalTensor<float> &dst,
                                                          const LocalTensor<float> &src, float v, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Adds(dst[off], src[off], v, 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMulPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Mul(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

// 统计降序序列 (偶数 lane 为 value) 中 >= x 的元素个数, 二分实现
template <typename QLIV2T>
__aicore__ inline int32_t QLIV2Vector<QLIV2T>::CountGE(const LocalTensor<float> &sortedDesc, int32_t n, float x)
{
    int32_t lo = 0;
    int32_t hi = n;
    while (lo < hi) {
        int32_t mid = (lo + hi) / 2;
        if (sortedDesc.GetValue(2 * mid) >= x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// mode=2 (use_candidate): 加载/排序候选行 (每行首个S2分片), 判定 tile 内块归属,
// 产出 position 级 isOut (fp32 0/1, 1=候选外) 与其 int32 形式
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::BuildCandidateMask(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                                               int32_t cuBaseS2Idx, int32_t innerS1Idx)
{
    int32_t candBlocks = static_cast<int32_t>(constInfo_.candidateTopkBlocks);
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t tileBlkNum = s2BaseSize_ / blockSize;
    int32_t tileBlockBase = cuBaseS2Idx / blockSize;
    if (constInfo_.candidateMode == CANDIDATE_MODE_CACHED_MASK) {
        uint32_t stride = CeilDiv(maxBlockNumPerBatch_ * kCacheBlockSize_, 2048) * 256 + 256;
        uint64_t row = info.candidateOutOffset / candBlocks + cuS1Idx;
        LocalTensor<float> blockMask = tmpBuf_.Get<float>()[4096];
        LocalTensor<float> posMask = tmpBuf_.Get<float>()[12288];
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        DataCopy(blockMask.template ReinterpretCast<int32_t>(), candidateTopkIndexInGm[row * stride + tileBlockBase], tileBlkNum);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Brcb(posMask, blockMask, tileBlkNum / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        candIsOut_ = posMask;
        return;
    }
    // R6 修复: 同核多行 (每 AIV 处理 CeilDiv(s1BaseSize,2)=2 行) 共享 candBuf 时,
    // 后一行的 tile0 重排序会覆盖前行候选 (s2 内层循环按 gS1 块整体推进) — candBuf 按行分区
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    LocalTensor<float> candPairs = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    if (info.isFirstS2InnerLoop) {
        // 加载候选行到 tmp 尾部, 转 fp32 后与有序索引组成 [values | idx] 对并降序排序
        LocalTensor<int32_t> candInt = tmp[12288].template ReinterpretCast<int32_t>();
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = candBlocks * sizeof(int32_t);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        AscendC::DataCopyPad(candInt,
                             candidateTopkIndexInGm[info.candidateOutOffset +
                                                    cuS1Idx * constInfo_.candidateTopkBlocks],
                             copyInParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        PipeBarrier<PIPE_V>();
        Cast(candPairs, candInt, RoundMode::CAST_NONE, candBlocks);
        PipeBarrier<PIPE_V>();
        DataCopy(candPairs[candBlocks].template ReinterpretCast<int32_t>(), globalTopkIndice_, candBlocks);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> candSortTmp = tmp[4096];
        QLIV2ServiceVec::SortAll(candPairs, candSortTmp, candBlocks);
        PipeBarrier<PIPE_V>();
        // V->S 围栏: 排序结果对后续标量 GetValue (CountGE 二分) 可见
        // (PipeBarrier<PIPE_V> 不足以保证标量读到 V 写数据; 参照 CANN topk_v200 的 SetFlag/WaitFlag<V_S> 模式)
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    }
    // Sorted candidate IDs restrict the scatter to blocks inside this tile.
    // (candBuf 按行分区, R6: 同核行间覆盖已修复)
    LocalTensor<float> candPairs2 = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    int32_t lo = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase));
    int32_t hi = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase + tileBlkNum));
    // Membership is a scatter, not a minimum-distance reduction over
    // every candidate x every tile block. Fence UB stores explicitly.
    LocalTensor<float> blockMask = tmp[4096];
    bool localCap = constInfo_.candidateMode == CANDIDATE_MODE_LOCAL_CAP;
    Duplicate(blockMask.template ReinterpretCast<int32_t>(),
              localCap ? QLIV2ServiceVec::NEG_HUGE_F32 : 0x3F800000, tileBlkNum);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    for (int32_t j = hi; j < lo; ++j) {
        int32_t block = static_cast<int32_t>(candPairs2.GetValue(2 * j)) - tileBlockBase;
        blockMask.template ReinterpretCast<int32_t>().SetValue(
            block, localCap ? QLIV2ServiceVec::POS_INF_F32 : 0);
    }
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    LocalTensor<float> posMask = tmp[12288];
    Brcb(posMask, blockMask, tileBlkNum / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    candIsOut_ = posMask;

    // R11 (leak 语义, §1.6): isOut 仅用于分数降级 (pen 链), 索引保留真实位置号 —
    // 候选外可达位置作为 topk 填充泄漏成有效索引 (对齐模型 where(idxs < compress_lens) 语义),
    // 不再把候选外索引改 -1 (原 isOutI32/CAST_RINT 链已删)。
}

// mode=1 (is_candidate_source): 块内 amax (log2(blockSize) 轮 Max 树) + pin 尾块 + 块级排序/归并
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessCandBlockTopk(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                                                 int32_t cuS2Len, int32_t cuS2LenVecAlign,
                                                                 int32_t cuRealAcSeq, int32_t innerS1Idx)
{
            int32_t blkLen = cuS2LenVecAlign;
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t blockNum = blkLen / blockSize;                         // 含尾块 (pad -inf 后按对齐长度计)
    int32_t realBlockNum = (cuS2Len + blockSize - 1) / blockSize; // 含有效位置的块数
    // 所有向量 count 按 64 对齐 (非对齐 count 的向量指令在本架构触发 aicore 异常);
    // 且块数必须过 AlignS2 (32*(4^n)*m, m<=3): SortAll 的 MrgSort 循环按 mrgGroups/4 缩减,
    // 非 4 幂组数 (如 192 块=6 组) 会整组丢失 (实测 s2=1536 丢块 128..191 段)
    int32_t blockNumPad = (blockNum < 64) ? 64 : AlignS2(blockNum);
    int32_t tileBlockBase = info.s2Idx * s2BaseSize_ / blockSize;
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    // 块内归约: vcgmax 每 32B 块 (8 fp32) 出 1 个 max, 紧凑输出 [blockNum]
    LocalTensor<float> blkScore = tmp[6144];
    // R10: blkLen=96 是 AlignS2 输出中唯一非 64 倍数的值 (≤128 段对齐到 32 的倍数),
    // 96/64 整除截断为 1 只归约 [0,64) — 块 8..11 残留上一 tile/行的 stale 分数
    // (实测 big128k_b2_varlen b1 行 73 tile 56: 块 14344 拿到 tile 55 块 14088 的
    // 3.5568, 虚高挤掉 2048 名边界块 2760; 小 shape 总块数≤2048 集合不变故未暴露)。
    // 改 CeilDiv 覆盖全部块; 多归约的 [96,128) stale 只落在 pad 块槽位, 被 -inf
    // 位型链位精确覆盖, 无害。
    int32_t brmRepeat = CeilDiv(blkLen, 64);
    BlockReduceMax(blkScore, tmp[0], brmRepeat, 64, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    int32_t lastBlk = (cuRealAcSeq - 1) / blockSize;
    int32_t pinLocal = lastBlk - tileBlockBase;
    // 块级 [scores | idx] 对: idx = tileBlockBase + j; j >= realBlockNum 的 pad 块置 -1
    // (禁止标量 SetValue: 标量写不受 PipeBarrier<PIPE_V> 围栏, 与后续 V 管道读存在确定性竞态
    //  (实测 innerS1Idx=1 行的填充被 SortAll 抢跑覆盖); 禁止 s32->f32 Cast (v220 无此组合)。
    //  改为纯 int32 向量算术: idx' = idx - (idx+1)*isPad, isPad = clamp(idx - thr, 0, 1),
    //  thr = base + realBlockNum - 1; 所有指令 count=blockNumPad (64 对齐), 按 64 分块避免 mask 寄存器限制。
    //  scratch 用 mode=1 独占区 [14336,15360) (mode=2 的 isOutI32 已随 R11 leak 改造删除),
    //  避开主路径 tmpSortBuf [4096,12288), 消除跨迭代残留读的隐患)
    LocalTensor<int32_t> idxScr = tmp[14336].template ReinterpretCast<int32_t>();      // 等差源
    LocalTensor<int32_t> thrI = tmp[14592].template ReinterpretCast<int32_t>();        // 阈值
    LocalTensor<int32_t> offI = tmp[14848].template ReinterpretCast<int32_t>();        // idx-thr -> isPad
    LocalTensor<int32_t> tI = tmp[15104].template ReinterpretCast<int32_t>();          // (idx+1)*isPad
    LocalTensor<int32_t> blkIdx = tmp[6144 + blockNumPad].template ReinterpretCast<int32_t>(); // 最终 idx
    ArithProgression<int32_t>(idxScr, static_cast<int32_t>(tileBlockBase), 1, blockNumPad);
    Duplicate(thrI, tileBlockBase + realBlockNum - 1, blockNumPad);
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Sub(offI[off], idxScr[off], thrI[off], 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Maxs(offI[off], offI[off], 0, 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Mins(offI[off], offI[off], 1, 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Adds(tI[off], idxScr[off], 1, 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Mul(tI[off], tI[off], offI[off], 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Sub(blkIdx[off], idxScr[off], tI[off], 64);
    }
    PipeBarrier<PIPE_V>();
    // pad 块 score 归一为 -inf 位型: 覆盖 [realBlockNum, blockNumPad), 含 stale 区 [blockNum, blockNumPad)
    // (尾 tile cuS2Len 非整块时 BlockReduceMax 只写 [0, blockNum), 其后是脏数据 —
    //  脏 score 的 (-inf,-1) 不同构对会挤进 topk 挤掉真实块, 实测 128K 场景 -1 槽超标)
    // score' = score + (score - NINF_bits)*(-isPad), isPad 于 pad 位为 1; 全 int32 向量
    if (blockNumPad > realBlockNum) {
        LocalTensor<int32_t> pRaw = tmp[15360].template ReinterpretCast<int32_t>(); // -isPad
        LocalTensor<int32_t> pNeg = tmp[15616].template ReinterpretCast<int32_t>(); // (score-NINF)*(-isPad)
        LocalTensor<int32_t> scoreI = blkScore.template ReinterpretCast<int32_t>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Duplicate(pRaw[off], -1, 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mul(pRaw[off], pRaw[off], offI[off], 64); // -isPad
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pNeg[off], scoreI[off], 8388608, 64); // score - NINF_bits (NINF=0xFF800000=-8388608)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mul(pNeg[off], pNeg[off], pRaw[off], 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Add(scoreI[off], scoreI[off], pNeg[off], 64); // pad 块 -> -inf 位型
        }
        PipeBarrier<PIPE_V>();
    }
    // pin 最新 token 所在块 (+inf 位型注入该块 score): 纯 int32 向量实现, 在 isPad 链之后
    // (复用其 scratch; 标量 SetValue 写不受 PipeBarrier<PIPE_V> fence, 与 SortAll 多轮读存在竞态, 已弃用)。
    // flag = -1 于 pin 位, 0 于其余: score' = score + (score - PINF_bits) * flag,
    // pin 位得 PINF_bits (0x7F800000 = +inf), 其余不变 (flag=0 使 pad 位 -inf 的中间溢出无害)。
    if (pinLocal >= 0 && pinLocal < realBlockNum) {
        LocalTensor<int32_t> pRaw = tmp[15360].template ReinterpretCast<int32_t>(); // [15360,15616)
        LocalTensor<int32_t> pNeg = tmp[15616].template ReinterpretCast<int32_t>(); // [15616,15872)
        LocalTensor<int32_t> pA = thrI;   // 复用 isPad 链已释放 scratch
        LocalTensor<int32_t> pB = offI;
        LocalTensor<int32_t> pFlag = tI;
        LocalTensor<int32_t> scoreI = blkScore.template ReinterpretCast<int32_t>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            // idxScr 为全局块号, pin 也必须用全局 id (tileBlockBase + pinLocal) 比较
            Adds(pRaw[off], idxScr[off], -(tileBlockBase + pinLocal), 64); // 0 于 pin 块
        }
        PipeBarrier<PIPE_V>();
        Duplicate(pNeg, 0, blockNumPad);
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Sub(pNeg[off], pNeg[off], pRaw[off], 64); // -(idx-pinLocal)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Maxs(pA[off], pRaw[off], 0, 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mins(pA[off], pA[off], 1, 64); // 1 若 idx > pin
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Maxs(pB[off], pNeg[off], 0, 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mins(pB[off], pB[off], 1, 64); // 1 若 idx < pin
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Add(pFlag[off], pA[off], pB[off], 64); // isNotPin (0/1, 互斥)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pFlag[off], pFlag[off], -1, 64); // -1 于 pin, 0 其余
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pRaw[off], scoreI[off], -2139095040, 64); // score - 0x7F800000 (复用 pRaw)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mul(pRaw[off], pRaw[off], pFlag[off], 64); // (score-PINF)*flag
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Add(scoreI[off], scoreI[off], pRaw[off], 64); // pin -> +inf 位型
        }
        PipeBarrier<PIPE_V>();
    }
    // 块级排序 + 归并到块级累加器
    LocalTensor<float> blkPairs = tmp[6144];   // [scores blockNumPad | idx blockNumPad]
    // MrgSort tmp 需容纳 mrgDst+mrgSrc = 2048*2 + blockNumPad*2 <= 4608 floats;
    // 11776 + 4608 = 16384 恰为 tmpBuf_(64KB) 末尾, 12288 起会越界 2KB (aicore 异常)
    LocalTensor<float> blkSortTmp = tmp[11776]; // 与 blkPairs 不重叠 (MrgSort src/tmp 禁止重叠)
    QLIV2ServiceVec::SortAll(blkPairs, blkSortTmp, blockNumPad);
    PipeBarrier<PIPE_V>();
    // 候选专用合并: 纯 V 回拷 (MergeSort 尾部 UB->UB DataCopy 不受 PipeBarrier<PIPE_V> fence,
    // 多 tile 下一次 MrgSort 读 acc 存在调度敏感竞态, 实测 tile1 读到 stale 数据)
    QLIV2ServiceVec::MergeSortVecCopy(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE],
                                      constInfo_.candidateTopkBlocks, blkPairs, blockNumPad, blkSortTmp);
    PipeBarrier<PIPE_V>();
}

// mode=1: 行末直出 candidate_topk_index 并复位块级累加器
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::CopyOutCandTopkIndex(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                                                 int32_t innerS1Idx)
{
    LocalTensor<uint32_t> candIdxULocal = outQueue_.AllocTensor<uint32_t>();
    ExtractIndex(candIdxULocal,
                 globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE].template ReinterpretCast<uint32_t>(),
                 constInfo_.candidateTopkBlocks);
    PipeBarrier<PIPE_V>();
    InitSortOutBuf(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
    outQueue_.EnQue<uint32_t>(candIdxULocal);
    candIdxULocal = outQueue_.DeQue<uint32_t>();
    QLIV2ServiceVec::CopyOut(candidateTopkIndexOutGm[info.candidateOutOffset +
                                                     cuS1Idx * constInfo_.candidateTopkBlocks],
                             candIdxULocal.template ReinterpretCast<int32_t>(), constInfo_.candidateTopkBlocks);
    outQueue_.FreeTensor(candIdxULocal);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessVec0(const QLIV2Common::RunInfo &info)
{
    // 只需要一个v核做
    if (blockId_ % 2 != 0) {
        return;
    }
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    // 计算输出w基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 4*64 + aic_offset
    int64_t vec0OutGmOffset = (info.loop % 2) * ((s1BaseSize_ * gSize_ * BLOCK_CUBE));
    // 计算输入weight的地址偏移，qScale的地址偏移与weight相同
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * qHeadNum_;
    // 当前需要计算的S1行数，处理尾块场景
    int32_t cuS1ProcNum = cuBaseS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuProcRealNum = cuS1ProcNum * gSize_;
    int32_t cuProcEleNum = QLIV2Common::Align(cuProcRealNum, 32); // 32: UB对齐, 参照v1

    LocalTensor<half> inWeightsUb = inQueue_.AllocTensor<half>();
    LocalTensor<half> inQScaleUb = inWeightsUb[cuProcEleNum];
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = cuProcRealNum * sizeof(half);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    if ((false /* retired experimental path */ || constInfo_.candidateMode == CANDIDATE_MODE_PACKED)) {
        copyInParams.blockLen = gSize_ * sizeof(half);
        uint32_t packed = 0;
        for (uint32_t query = 0; query < info.groupQueries; ++query) {
            if ((info.queryMask & (1U << query)) == 0) {
                continue;
            }
            AscendC::DataCopyPad(inWeightsUb[packed * gSize_],
                                weightsGm[info.tensorWeightsOffset + query * gSize_], copyInParams, padParams);
            AscendC::DataCopyPad(inQScaleUb[packed * gSize_],
                                qScaleGm[info.tensorWeightsOffset + query * gSize_], copyInParams, padParams);
            ++packed;
        }
    } else {
        AscendC::DataCopyPad(inWeightsUb, weightsGm[weightGmOffset], copyInParams, padParams);
        AscendC::DataCopyPad(inQScaleUb, qScaleGm[weightGmOffset], copyInParams, padParams);
    }

    inQueue_.EnQue<half>(inWeightsUb);
    inWeightsUb = inQueue_.DeQue<half>();
    AscendC::Mul(inWeightsUb, inWeightsUb, inQScaleUb, cuProcEleNum);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> resUb = outQueue_.AllocTensor<half>();
    AscendC::Brcb(resUb, inWeightsUb, static_cast<uint8_t>(cuProcEleNum / 8), {1, 8});
    inQueue_.FreeTensor(inWeightsUb);

    outQueue_.EnQue<half>(resUb);
    resUb = outQueue_.DeQue<half>();
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = cuProcRealNum * BLOCK_CUBE * sizeof(half);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    AscendC::DataCopyPad(vec0OutGm[vec0OutGmOffset], resUb, copyOutParams);
    outQueue_.FreeTensor(resUb);
}

template <typename QLIV2T>
__aicore__ inline int32_t QLIV2Vector<QLIV2T>::AlignS2(int32_t cuS2Len)
{
    // 限制：当前cuS2Len最大为2048，暂不考虑更长
    // 该函数目的是将cuS2Len对齐到形如 32*(4^n)*m 的形式 (m ∈ [1, 3])，方便后续sort/merge
    if (cuS2Len <= ELE_NUM_128) {
        return Align(cuS2Len, ELE_NUM_32);
    } else if (cuS2Len <= ELE_NUM_512) {
        return Align(cuS2Len, ELE_NUM_128);
    } else {
        return Align(cuS2Len, ELE_NUM_512);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessVec1(const QLIV2Common::RunInfo &info)
{
    if ((false /* retired experimental path */ || constInfo_.candidateMode == CANDIDATE_MODE_PACKED)) {
        ProcessGroupedVec1(info);
        return;
    }
    if (false /* retired experimental path */) {
        ProcessCandidateVec1(info);
        return;
    }
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 4*2048 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * (s1BaseSize_ * s2BaseSize_);

    // cuS1BeginIdxPerAiv: 每个AIV的S1起始偏移
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // cuS1ProcNumPerAiv: 每个AIv的S1计算量
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);
    // 基本块基地址偏移奇数核加一个S1地址偏移
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * s2BaseSize_;
    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * retainedPairSize_);
        blockS2StartIdx_ = 0;
        if ((constInfo_.candidateMode == CANDIDATE_MODE_SOURCE || constInfo_.candidateMode == CANDIDATE_MODE_SOURCE_TOP512)) {
            InitSortOutBuf(globalBlockTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);
        }
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    int32_t cuRealAcSeqCount = 0;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + cuS1BeginIdxPerAiv;
    }
    int32_t cuRealAcSeqIni = cuRealAcSeq;

    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeqCount += 1;
            cuRealAcSeq = (cuRealAcSeqCount + cuRealAcSeqIni) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        // 当前vec1ResGm对应S1的位置
        uint64_t wsOffset = static_cast<uint64_t>(info.saveWorkSpaceIdx) * s1BaseSize_ * retainedPairSize_ +
                            static_cast<uint64_t>(cuS1Idx - cuBaseS1Idx) * retainedPairSize_;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = AlignS2(cuS2Len);
            LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
            LocalTensor<float> kScaleUb = mmInUb[cuS2LenVecAlign];
            LocalTensor<half> kScaleTUb = kScaleUb.template ReinterpretCast<half>()[cuS2LenVecAlign];
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> padTParams{false, 0, 0, 0};
            AscendC::DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = cuS2Len * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            copyInParams.rsv = 0;
            AscendC::DataCopyPad(mmInUb, mm1ResGm[mmGmOffset + innerS1Idx * s2BaseSize_], copyInParams, padParams);
            GetKeyScale(info, kScaleTUb, info.bIdx, cuBaseS2Idx, cuS2Len);
            inQueue_.EnQue<float>(mmInUb);
            mmInUb = inQueue_.DeQue<float>();
            AscendC::Cast(kScaleUb, kScaleTUb, RoundMode::CAST_NONE, cuS2Len);
            PipeBarrier<PIPE_V>();
            AscendC::Mul(mmInUb, mmInUb, kScaleUb, cuS2Len);
            PipeBarrier<PIPE_V>();
            // mode=2 (use_candidate, R11 leak): 候选块外 score 降级 NEG_HUGE (S4a, 纯算术无 NaN);
            // 仅降级排序资格、不取消入选资格 — 不足 topk 时候选外可达位置作为填充泄漏 (§1.6)
            if (constInfo_.candidateMode == CANDIDATE_MODE_CACHED_MASK ||
                constInfo_.candidateMode == CANDIDATE_MODE_LOCAL_CAP) {
                BuildCandidateMask(info, cuS1Idx, cuBaseS2Idx, innerS1Idx);
                Min(mmInUb, mmInUb, candIsOut_, uint64_t(64), static_cast<uint8_t>(CeilDiv(cuS2Len, 64)),
                    BinaryRepeatParams{1, 1, 1, 8, 8, 8});
                PipeBarrier<PIPE_V>();
            } else if (constInfo_.candidateMode == CANDIDATE_MODE_CONSUMER ||
                       false /* retired experimental path */) {
                BuildCandidateMask(info, cuS1Idx, cuBaseS2Idx, innerS1Idx);
                PipeBarrier<PIPE_V>();
                LocalTensor<float> pen = tmpBuf_.Get<float>()[4096]; // blkIdxF/acc/d 已释放, 复用
                Sub(pen, candNegHuge_, mmInUb, cuS2Len);
                PipeBarrier<PIPE_V>();
                Mul(pen, pen, candIsOut_, cuS2Len);
                PipeBarrier<PIPE_V>();
                Add(mmInUb, mmInUb, pen, cuS2Len);
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<float> sortBuff = tmpBuf_.Get<float>();
            LocalTensor<float> sortScoreUb = sortBuff;
            LocalTensor<float> sortIndiceUb = sortBuff[cuS2LenVecAlign];
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), QLIV2ServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, mmInUb, 0.0f, cuS2Len);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(mmInUb);
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // 无效数据索引填充为-1
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
                PipeBarrier<PIPE_V>();
            }
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            PipeBarrier<PIPE_V>();
            // R11 (leak 语义, §1.6): 候选块外索引不再置 -1 —
            // 分数已在 S4a 降级 NEG_HUGE (仍高于 -inf: 不足 topk 时作为填充入选, 排候选内之后),
            // 索引保留真实位置号, 与模型 where(idxs < compress_lens, idxs + offset, -1) 等价:
            // 仅不可达位置 (score -inf 沉底 + 尾部 -1 填充) 输出 -1。
            // mode=1 (is_candidate_source): 块化 amax + pin + 块级排序归并 (S5a/S6a)
            if ((constInfo_.candidateMode == CANDIDATE_MODE_SOURCE || constInfo_.candidateMode == CANDIDATE_MODE_SOURCE_TOP512)) {
                ProcessCandBlockTopk(info, cuS1Idx, cuS2Len, cuS2LenVecAlign, cuRealAcSeq, innerS1Idx);
            }
            LocalTensor<float> tmpSortBuf = sortBuff[2 * cuS2LenVecAlign];
            QLIV2ServiceVec::SortAll(sortBuff, tmpSortBuf, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            QLIV2ServiceVec::MergeSort(globalTopkUb_[innerS1Idx * retainedPairSize_], retainedTopk_, sortBuff,
                                       Min(static_cast<uint32_t>(cuS2LenVecAlign), retainedTopk_), tmpSortBuf);
            PipeBarrier<PIPE_V>();
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            // 中间结果保存
            if (needCopyOutGm && !info.isNeedLD) {
                LocalTensor<uint32_t> idxULocal = outQueue_.AllocTensor<uint32_t>();
                ExtractIndex(idxULocal,
                             globalTopkUb_[innerS1Idx * retainedPairSize_].template ReinterpretCast<uint32_t>(),
                             retainedTopk_);
                PipeBarrier<PIPE_V>();
                // A15: output_idx_offset — 输出拷 GM 前逐元素加行偏移 (对齐 arch35 IndicesAddOffset;
                // 零偏移零开销; GM 标量读无 V_S 竞态; int32 整数域 Adds 位精确, -1 槽 +0 不变。
                // candidate_topk_index 不加 (相对块号契约, §11.6))
                if (isOutputIdxOffsetValid_) {
                    int32_t rowOff = outputIdxOffsetGm_.GetValue(info.outputIdxOffsetCoreOffset +
                                                                 cuS1Idx * kHeadNum_);
                    if (rowOff != 0) {
                        LocalTensor<int32_t> idxI32 = idxULocal.template ReinterpretCast<int32_t>();
                        for (int32_t off = 0; off < constInfo_.sparseCount; off += 64) {
                            Adds(idxI32[off], idxI32[off], rowOff, 64);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
                InitSortOutBuf(globalTopkUb_[innerS1Idx * retainedPairSize_], retainedPairSize_);
                outQueue_.EnQue<uint32_t>(idxULocal);
                idxULocal = outQueue_.DeQue<uint32_t>();
                QLIV2ServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                                         idxULocal.template ReinterpretCast<int32_t>(), constInfo_.sparseCount);
                outQueue_.FreeTensor(idxULocal);
                // mode=1: 块级 topk 直出 candidate_topk_index
                if ((constInfo_.candidateMode == CANDIDATE_MODE_SOURCE || constInfo_.candidateMode == CANDIDATE_MODE_SOURCE_TOP512)) {
                    CopyOutCandTopkIndex(info, cuS1Idx, innerS1Idx);
                }
            }
            // LD拷贝到当前vector对应S1的位置
            if (info.isNeedLD && info.isLastS2InnerLoop) { // 当前核存在归约任务 且是最后处理的一段
                AscendC::DataCopyExtParams copyWsParams;
                copyWsParams.blockLen = retainedPairSize_ * sizeof(float);
                copyWsParams.srcStride = 0;
                copyWsParams.dstStride = 0;
                copyWsParams.blockCount = 1;
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                AscendC::DataCopyPad(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * retainedPairSize_],
                                     copyWsParams);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                InitSortOutBuf(globalTopkUb_[innerS1Idx * retainedPairSize_], retainedPairSize_);
                PipeBarrier<PIPE_V>();
            }
        } else if (cuRealAcSeq <= 0) {
            // 无效长度处理
            if (info.isNeedLD && info.isLastS2InnerLoop) {
                PipeBarrier<PIPE_V>();
                InitSortOutBuf(globalTopkUb_[innerS1Idx * retainedPairSize_], retainedPairSize_);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                QLIV2ServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * retainedPairSize_],
                                         retainedPairSize_);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            } else {
                CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
            }
        } else if (cuS2Len <= 0) {
            // LD拷贝到当前vector对应S1的位置
            if (info.isNeedLD && info.isLastS2InnerLoop) { // 当前核存在归约任务 且是最后处理的一段
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                QLIV2ServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * retainedPairSize_],
                                         retainedPairSize_);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                InitSortOutBuf(globalTopkUb_[innerS1Idx * retainedPairSize_], retainedPairSize_);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    // BNSD场景无效S1 输出-1
    if (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2SizeOrig;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

// Direct consumer: reuse the Cube score and the native Sort/Merge pipeline.
// Only the even AIV owns the single query row; the odd AIV still participates
// in the existing cross-core flags and final SyncAll.
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessCandidateVec1(const RunInfo &info)
{
    if (blockId_ % 2 != 0) {
        return;
    }
    LocalTensor<int32_t> candidates = directCandidatesBuf_.Get<int32_t>();
    // tmpBuf_ held the previous tile's Sort/Merge scratch. MTE2 must not
    // overwrite it with scales until those Vector reads have completed.
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    if (info.isFirstS2InnerLoop) {
        InitSortOutBuf(globalTopkUb_, retainedPairSize_);
        DataCopy(candidates, candidateTopkIndexInGm[info.candidateOutOffset], constInfo_.candidateTopkBlocks);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    }
    uint32_t count = info.actualSingleProcessSInnerSize;
    uint32_t first = info.s2Idx * s2BaseSize_;
    LocalTensor<float> scores = inQueue_.AllocTensor<float>();
    LocalTensor<float> scales = scores[s2BaseSize_];
    LocalTensor<half> paddedScales = tmpBuf_.Get<half>();
    LocalTensor<float> floatScales = tmpBuf_.Get<float>()[s2BaseSize_];
    // Invalid candidate slots need no scale DMA. Zero them once per tile so
    // unused half values cannot introduce NaNs before score masking.
    Duplicate(paddedScales, half(0), 2 * count);
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    DataCopy(scores, mm1ResGm[(info.loop % 2) * s2BaseSize_], count);
    DataCopyExtParams scaleCopy{1, QLIV2Candidate::BLOCK * sizeof(half), 0, 0, 0};
    DataCopyPadExtParams<half> scalePad{true, 0, QLIV2Candidate::BLOCK, 0};
    uint64_t scaleStride = constInfo_.keyDequantScaleStride0 != 0 ?
        constInfo_.keyDequantScaleStride0 : kCacheBlockSize_;
    for (uint32_t local = 0; local < count; local += QLIV2Candidate::BLOCK) {
        uint32_t slot = (first + local) / QLIV2Candidate::BLOCK;
        int32_t candidate = slot < QLIV2Candidate::PREFIX / QLIV2Candidate::BLOCK ? 0 :
            candidates.GetValue(slot - QLIV2Candidate::PREFIX / QLIV2Candidate::BLOCK);
        int32_t block = QLIV2Candidate::OriginalBlock(slot, candidate, info.candidateVisible);
        if (block < 0) {
            continue;
        }
        uint32_t position = block * QLIV2Candidate::BLOCK;
        uint64_t page = blockTableGm.GetValue(info.bIdx * maxBlockNumPerBatch_ + position / kCacheBlockSize_);
        uint64_t offset = page * scaleStride + position % kCacheBlockSize_;
        // The scalar result is a DMA address, not a UB store consumed by MTE2.
        // Candidate loads were fenced above; no per-block S->MTE2 fence is needed.
        DataCopyPad(paddedScales[2 * local], kScaleGm[offset], scaleCopy, scalePad);
    }
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Cast(floatScales, paddedScales, RoundMode::CAST_NONE, 2 * count);
    PipeBarrier<PIPE_V>();
    Gather(scales, floatScales, globalTopkIndice_.template ReinterpretCast<uint32_t>(), 0, count);
    PipeBarrier<PIPE_V>();
    Mul(scores, scores, scales, count);
    PipeBarrier<PIPE_V>();

    // Prefix candidates keep their true scores. Non-candidate prefix positions
    // provide the same -1e30 shortage-fill eligibility as mode 2.
    uint64_t prefixMask = 0;
    if (first == 0) {
        for (uint32_t i = 0; i < constInfo_.candidateTopkBlocks; ++i) {
            int32_t block = candidates.GetValue(i);
            if (block >= 0 && block < static_cast<int32_t>(QLIV2Candidate::PREFIX / QLIV2Candidate::BLOCK)) {
                prefixMask |= uint64_t{1} << block;
            }
        }
    }
    LocalTensor<float> sortBuff = tmpBuf_.Get<float>();
    LocalTensor<int32_t> sortBits = sortBuff.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> indices = sortBits[count];
    Adds(sortBuff, scores, 0.0f, count);
    PipeBarrier<PIPE_V>();
    // Build one origin and score cap per block, then expand all blocks together.
    // Issuing ArithProgression/Duplicate for each eight-token block is costly,
    // especially when invalid candidate slots interleave with valid ones.
    LocalTensor<int32_t> blockOrigins = sortBits[2 * count];
    LocalTensor<int32_t> blockCaps = blockOrigins[count / QLIV2Candidate::BLOCK];
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    for (uint32_t local = 0; local < count; local += QLIV2Candidate::BLOCK) {
        uint32_t slot = (first + local) / QLIV2Candidate::BLOCK;
        int32_t candidate = slot < QLIV2Candidate::PREFIX / QLIV2Candidate::BLOCK ? 0 :
            candidates.GetValue(slot - QLIV2Candidate::PREFIX / QLIV2Candidate::BLOCK);
        int32_t block = QLIV2Candidate::OriginalBlock(slot, candidate, info.candidateVisible);
        int32_t cap = QLIV2ServiceVec::POS_INF_F32;
        if (slot < QLIV2Candidate::PREFIX / QLIV2Candidate::BLOCK && ((prefixMask >> slot) & 1) == 0) {
            cap = QLIV2ServiceVec::NEG_HUGE_F32;
        }
        if (block < 0) {
            cap = QLIV2ServiceVec::NEG_INF;
        }
        blockOrigins.SetValue(local / QLIV2Candidate::BLOCK, block * static_cast<int32_t>(QLIV2Candidate::BLOCK));
        blockCaps.SetValue(local / QLIV2Candidate::BLOCK, cap);
    }
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    Brcb(indices.template ReinterpretCast<float>(), blockOrigins.template ReinterpretCast<float>(),
         count / (QLIV2Candidate::BLOCK * 8), {1, 8});
    Brcb(scales, blockCaps.template ReinterpretCast<float>(), count / (QLIV2Candidate::BLOCK * 8), {1, 8});
    LocalTensor<int32_t> lanes = scores.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> groups = sortBits[3 * count];
    ArithProgression<int32_t>(lanes, 0, 1, count);
    PipeBarrier<PIPE_V>();
    ShiftRight(groups, lanes, 3, count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(groups, groups, 3, count);
    PipeBarrier<PIPE_V>();
    Sub(lanes, lanes, groups, count);
    VecMinPair(sortBuff, sortBuff, scales, count);
    PipeBarrier<PIPE_V>();
    Add(indices, indices, lanes, count);
    PipeBarrier<PIPE_V>();
    Cast(scales, indices, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    // A partial last block is masked per token. Invalid blocks already have
    // -inf scores; clamp their negative origins to the output sentinel -1.
    LocalTensor<uint8_t> visibleMask = blockOrigins.template ReinterpretCast<uint8_t>();
    CompareScalar(visibleMask, scales, static_cast<float>(static_cast<int32_t>(info.candidateVisible)), CMPMODE::LT, count);
    VecMaxsScalar(scales, scales, -1.0f, count);
    LocalTensor<float> replacement = groups.template ReinterpretCast<float>();
    Duplicate(replacement, -1.0f, count);
    PipeBarrier<PIPE_V>();
    Select(scales, visibleMask, scales, replacement, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    PipeBarrier<PIPE_V>();
    Cast(indices, scales, RoundMode::CAST_RINT, count);
    Duplicate(groups, QLIV2ServiceVec::NEG_INF, count);
    PipeBarrier<PIPE_V>();
    Select(sortBuff, visibleMask, sortBuff, replacement, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    PipeBarrier<PIPE_V>();
    inQueue_.FreeTensor(scores);
    LocalTensor<float> scratch = sortBuff[2 * count];
    QLIV2ServiceVec::SortAll(sortBuff, scratch, count);
    PipeBarrier<PIPE_V>();
    QLIV2ServiceVec::MergeSort(globalTopkUb_, retainedTopk_, sortBuff, Min(count, retainedTopk_), scratch);
    PipeBarrier<PIPE_V>();
    if (!info.isLastS2InnerLoop) {
        return;
    }
    if (info.isNeedLD) {
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopy(vec1ResGm[static_cast<uint64_t>(info.saveWorkSpaceIdx) * retainedPairSize_],
                 globalTopkUb_, retainedPairSize_);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    } else {
        LocalTensor<uint32_t> result = outQueue_.AllocTensor<uint32_t>();
        ExtractIndex(result, globalTopkUb_.template ReinterpretCast<uint32_t>(), retainedTopk_);
        outQueue_.EnQue<uint32_t>(result);
        result = outQueue_.DeQue<uint32_t>();
        QLIV2ServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset], result.template ReinterpretCast<int32_t>(),
                                constInfo_.sparseCount);
        outQueue_.FreeTensor(result);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessGroupedVec1(const RunInfo &info)
{
    using namespace QLIV2Candidate;
    uint32_t halfQueries = 2;
    uint32_t begin = (blockId_ % 2) * halfQueries;
    uint32_t end = Min(begin + halfQueries, info.groupQueries);
    if (begin >= end) {
        return;
    }
    if (info.groupFirst) {
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * retainedPairSize_);
    }
    if ((info.queryMask & (3U << begin)) == 0) {
        FinishGrouped(info);
        return;
    }
    uint32_t count = info.actualSingleProcessSInnerSize;
    uint32_t blocks = count / BLOCK;
    uint32_t first = info.s2Idx * s2BaseSize_ / BLOCK;
    uint32_t tileBlocks = s2BaseSize_ / BLOCK;
    LocalTensor<int32_t> origins = groupPlanBuf_.Get<int32_t>();
    LocalTensor<int32_t> membership = origins[tileBlocks];
    LocalTensor<int32_t> physical = origins[2 * tileBlocks];
    LocalTensor<float> sharedScales = groupScaleBuf_.Get<float>();
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    DataCopy(origins, candidateTopkIndexInGm[info.candidateOutOffset + first], blocks);
    DataCopy(membership, candidateTopkIndexInGm[info.candidateOutOffset + GROUP_CAPACITY + first], blocks);
    if (constInfo_.candidateMode != CANDIDATE_MODE_PACKED) {
        DataCopy(physical, candidateTopkIndexInGm[info.candidateOutOffset + 2 * GROUP_CAPACITY + first], blocks);
    }
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    LocalTensor<half> paddedScales = tmpBuf_.Get<half>();
    if (constInfo_.candidateMode == CANDIDATE_MODE_PACKED) {
        uint64_t group = info.candidateOutOffset / GROUP_STRIDE;
        uint64_t base = info.candidateOutOffset % GROUP_STRIDE - GROUP_HEADER;
        uint64_t token = (group * GROUP_CAPACITY + base + first) * BLOCK;
        DataCopy(paddedScales, kScaleGm[token], count);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Cast(sharedScales, paddedScales, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<float> floatScales = tmpBuf_.Get<float>()[s2BaseSize_];
        Duplicate(paddedScales, half(0), 2 * count);
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        uint64_t scaleStride = constInfo_.keyDequantScaleStride0 != 0 ?
            constInfo_.keyDequantScaleStride0 : kCacheBlockSize_;
        for (uint32_t block = 0; block < blocks;) {
            int32_t token = physical.GetValue(block);
            if (token < 0) {
                ++block;
                continue;
            }
            uint64_t offset = static_cast<uint64_t>(token / kCacheBlockSize_) * scaleStride + token % kCacheBlockSize_;
            DataCopyExtParams copy{1, BLOCK * sizeof(half), 0, 0, 0};
            DataCopyPadExtParams<half> pad{true, 0, BLOCK, 0};
            if (block + 1 < blocks) {
                int32_t next = physical.GetValue(block + 1);
                if (next >= 0) {
                    uint64_t nextOffset = static_cast<uint64_t>(next / kCacheBlockSize_) * scaleStride + next % kCacheBlockSize_;
                    if (nextOffset >= offset + BLOCK && (nextOffset - offset - BLOCK) * sizeof(half) <= 0xffffffffULL) {
                        copy.blockCount = 2;
                        copy.srcStride = (nextOffset - offset - BLOCK) * sizeof(half);
                    }
                }
            }
            DataCopyPad(paddedScales[2 * block * BLOCK], kScaleGm[offset], copy, pad);
            block += copy.blockCount;
        }
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Cast(floatScales, paddedScales, RoundMode::CAST_NONE, 2 * count);
        PipeBarrier<PIPE_V>();
        Gather(sharedScales, floatScales, globalTopkIndice_.template ReinterpretCast<uint32_t>(), 0, count);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t query = begin; query < end; ++query) {
        if ((info.queryMask & (1U << query)) == 0) {
            continue;
        }
        uint32_t packed = 0;
        for (uint32_t q = 0; q < query; ++q) {
            packed += (info.queryMask >> q) & 1U;
        }
        uint32_t visible = info.candidateVisible + (constInfo_.attenMaskFlag ? query : 0);
        LocalTensor<float> topk = globalTopkUb_[(query - begin) * retainedPairSize_];
        LocalTensor<float> scores = inQueue_.AllocTensor<float>();
        LocalTensor<float> scales = scores[s2BaseSize_];
        uint64_t offset = (info.loop % 2) * s1BaseSize_ * s2BaseSize_ + packed * s2BaseSize_;
        DataCopy(scores, mm1ResGm[offset], count);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Mul(scores, scores, sharedScales, count);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> sortBuff = tmpBuf_.Get<float>();
        LocalTensor<int32_t> sortBits = sortBuff.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> indices = sortBits[count];
        Adds(sortBuff, scores, 0.0f, count);
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> blockOrigins = sortBits[2 * count];
        LocalTensor<int32_t> blockCaps = blockOrigins[count / BLOCK];
        if (info.prefixBucket) {
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            for (uint32_t block = 0; block < blocks; ++block) {
                int32_t origin = origins.GetValue(block);
                int32_t cap = (!info.prefixBucket || (membership.GetValue(block) & (1 << query))) ?
                    QLIV2ServiceVec::POS_INF_F32 : QLIV2ServiceVec::NEG_HUGE_F32;
                if (origin < 0) {
                    cap = QLIV2ServiceVec::NEG_INF;
                }
                blockOrigins.SetValue(block, origin);
                blockCaps.SetValue(block, cap);
            }
            SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
            Brcb(indices.template ReinterpretCast<float>(), blockOrigins.template ReinterpretCast<float>(),
                 count / (QLIV2Candidate::BLOCK * 8), {1, 8});
            Brcb(scales, blockCaps.template ReinterpretCast<float>(), count / (QLIV2Candidate::BLOCK * 8), {1, 8});
        } else {
            Brcb(indices.template ReinterpretCast<float>(), origins.template ReinterpretCast<float>(),
                 count / (BLOCK * 8), {1, 8});
            Brcb(scales, membership.template ReinterpretCast<float>(), count / (BLOCK * 8), {1, 8});
        }
        LocalTensor<int32_t> groups = sortBits[3 * count];
        PipeBarrier<PIPE_V>();
        VecMinPair(sortBuff, sortBuff, scales, count);
        PipeBarrier<PIPE_V>();
        Add(indices, indices, groupLaneBuf_.Get<int32_t>(), count);
        PipeBarrier<PIPE_V>();
        Cast(scales, indices, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        // A partial last block is masked per token. Invalid blocks already have
        // -inf scores; clamp their negative origins to the output sentinel -1.
        LocalTensor<uint8_t> visibleMask = blockOrigins.template ReinterpretCast<uint8_t>();
        CompareScalar(visibleMask, scales, static_cast<float>(static_cast<int32_t>(visible)), CMPMODE::LT, count);
        VecMaxsScalar(scales, scales, -1.0f, count);
        LocalTensor<float> replacement = groups.template ReinterpretCast<float>();
        Duplicate(replacement, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Select(scales, visibleMask, scales, replacement, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
        Cast(indices, scales, RoundMode::CAST_RINT, count);
        Duplicate(groups, QLIV2ServiceVec::NEG_INF, count);
        PipeBarrier<PIPE_V>();
        Select(sortBuff, visibleMask, sortBuff, replacement, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(scores);
        LocalTensor<float> scratch = sortBuff[2 * count];
        QLIV2ServiceVec::SortAll(sortBuff, scratch, count);
        PipeBarrier<PIPE_V>();
        QLIV2ServiceVec::MergeSort(topk, retainedTopk_, sortBuff, Min(count, retainedTopk_), scratch);
        PipeBarrier<PIPE_V>();
    }
    FinishGrouped(info);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::FinishGrouped(const RunInfo &info)
{
    uint32_t begin = (blockId_ % 2) * 2;
    uint32_t end = Min(begin + 2, info.groupQueries);
    if (info.groupLast) {
        // Finish every original query, even if it is absent from the last bucket.
        for (uint32_t query = begin; query < end; ++query) {
            LocalTensor<float> topk = globalTopkUb_[(query - begin) * retainedPairSize_];
            if (info.isNeedLD) {
                uint64_t offset = static_cast<uint64_t>(info.saveWorkSpaceIdx) * s1BaseSize_ * retainedPairSize_ +
                                  query * retainedPairSize_;
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopy(vec1ResGm[offset], topk, retainedPairSize_);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                continue;
            }
            LocalTensor<uint32_t> result = outQueue_.AllocTensor<uint32_t>();
            ExtractIndex(result, topk.template ReinterpretCast<uint32_t>(), retainedTopk_);
            outQueue_.EnQue<uint32_t>(result);
            result = outQueue_.DeQue<uint32_t>();
            QLIV2ServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + query * constInfo_.sparseCount],
                                    result.template ReinterpretCast<int32_t>(), constInfo_.sparseCount);
            outQueue_.FreeTensor(result);
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessLD()
{
    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
    LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<float> indexValuePadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams indexValueParams;
    indexValueParams.blockLen = retainedPairSize_ * sizeof(int32_t); // bytes
    indexValueParams.srcStride = (s1BaseSize_ - 1) * retainedPairSize_ * sizeof(int32_t);
    indexValueParams.dstStride = 0;

    uint32_t ldProWorkspaceNum = ldInfo_.workspaceNum;
    uint32_t ldProcessLen = 4; // 4: 4块归约任务做一次Merge
    uint32_t ldProcessNum = (ldProWorkspaceNum - 1) / (ldProcessLen - 1);
    uint32_t ldTailLen = ldProWorkspaceNum - (ldProcessNum * (ldProcessLen - 1) + 1); // 尾块长度

    for (uint32_t j = 0; j < ldInfo_.mNum; j++) {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        // 拷贝第一块归约任务
        indexValueParams.blockCount = 1;
        uint64_t wsOffsetIni = static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * retainedPairSize_ +
                               static_cast<uint64_t>(ldInfo_.mStart + j) * retainedPairSize_;
        AscendC::DataCopyPad(curValueIdxUb, vec1ResGm[wsOffsetIni], indexValueParams, indexValuePadParams);
        uint64_t valueOffset = retainedPairSize_;
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        // 处理等于4的部分
        for (uint32_t i = 0; i < ldProcessNum; i++) {
            // LD处理偏移
            uint64_t wsOffset =
                static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * retainedPairSize_ +
                static_cast<uint64_t>(ldInfo_.mStart + j) * retainedPairSize_ +
                static_cast<uint64_t>(i * (ldProcessLen - 1) + 1) * s1BaseSize_ * retainedPairSize_;
            indexValueParams.blockCount = ldProcessLen - 1; // 拷贝4块进行merge

            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            AscendC::DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset], indexValueParams,
                                 indexValuePadParams);
            // merge参数
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = retainedTopk_;
            params.elementLengths[1] = retainedTopk_;
            params.elementLengths[2] = retainedTopk_;
            params.elementLengths[3] = retainedTopk_;
            params.ifExhaustedSuspension = true;
            params.validBit = 0b1111;
            params.repeatTimes = 1;

            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[retainedPairSize_];
            srcList.src3 = curValueIdxUb[2 * retainedPairSize_];
            srcList.src4 = curValueIdxUb[3 * retainedPairSize_];
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, retainedPairSize_);
            PipeBarrier<PIPE_V>();
        }

        // 处理不等于4的部分
        if (ldTailLen != 0) {
            // 搬运尾块
            uint64_t wsOffsetTail =
                static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * retainedPairSize_ +
                static_cast<uint64_t>(ldInfo_.mStart + j) * retainedPairSize_ +
                static_cast<uint64_t>(ldProcessNum * (ldProcessLen - 1) + 1) * s1BaseSize_ * retainedPairSize_;
            indexValueParams.blockCount = ldTailLen;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            AscendC::DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffsetTail], indexValueParams,
                                 indexValuePadParams);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = retainedTopk_;
            params.elementLengths[1] = retainedTopk_;
            params.elementLengths[2] = retainedTopk_;
            params.elementLengths[3] = retainedTopk_;
            params.ifExhaustedSuspension = true;
            if (ldTailLen == 1) {
                params.validBit = 0b0011;
            } else if (ldTailLen == 2) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[retainedPairSize_];
            srcList.src3 = curValueIdxUb[2 * retainedPairSize_];
            srcList.src4 = curValueIdxUb[3 * retainedPairSize_];
            PipeBarrier<PIPE_V>();
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, retainedPairSize_);
            PipeBarrier<PIPE_V>();
        }

        // 搬出
        Extract(outValueUb, outIdxUb, curValueIdxUb, (retainedTopk_ / 32));
        PipeBarrier<PIPE_V>();
        InitSortOutBuf(curValueIdxUb, retainedPairSize_);
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        // A15: LD(decode) 路径同样加 output_idx_offset (行前缀 = indiceOutCoreOffset/sparseCount, 即
        // batch 前缀 x kHeadNum + n2; 与 ProcessVec1 消费点同语义)
        if (isOutputIdxOffsetValid_) {
            uint32_t rowGlobal = ldInfo_.mStart + j;
            int32_t rowOff = outputIdxOffsetGm_.GetValue(
                ldInfo_.indiceOutCoreOffset / constInfo_.sparseCount + rowGlobal * constInfo_.kHeadNum);
            if (rowOff != 0) {
                for (int32_t off = 0; off < constInfo_.sparseCount; off += 64) {
                    Adds(idxULocal1[off], idxULocal1[off], rowOff, 64);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        uint64_t outOffset =
            ldInfo_.indiceOutCoreOffset + (ldInfo_.mStart + j) * constInfo_.kHeadNum * constInfo_.sparseCount;
        AscendC::DataCopyPad(indiceOutGm[outOffset], idxULocal1, copyOutParams);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
