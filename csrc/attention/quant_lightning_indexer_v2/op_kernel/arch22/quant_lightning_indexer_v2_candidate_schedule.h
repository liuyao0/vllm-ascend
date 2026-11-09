// SPDX-License-Identifier: Apache-2.0
#ifndef QLI_V2_CANDIDATE_SCHEDULE_H
#define QLI_V2_CANDIDATE_SCHEDULE_H

// Scalar scheduling shared by AIC and AIV. No host reads of device lengths.
namespace QLIV2Candidate {
constexpr unsigned BLOCK = 8;
constexpr unsigned PREFIX = 512;
constexpr unsigned TILE = 2048;
constexpr unsigned GROUP_Q = 4;
constexpr unsigned GROUP_CAPACITY = GROUP_Q * 2048 + 16 * 64;
constexpr unsigned GROUP_HEADER = 64;
constexpr unsigned GROUP_STRIDE = GROUP_HEADER + 3 * GROUP_CAPACITY;


struct Schedule {
    unsigned rows;
    unsigned slots;
    unsigned tiles;
    unsigned parts;
    unsigned groups;
    __aicore__ inline Schedule(unsigned activeRows, unsigned candidates, unsigned cores)
        : rows(activeRows), slots(PREFIX + candidates * BLOCK), tiles((slots + TILE - 1) / TILE)
    {
        unsigned rowsForSplit = rows == 0 ? 1 : rows;
        parts = cores / rowsForSplit;
        parts = parts == 0 ? 1 : parts;
        parts = parts < tiles ? parts : tiles;
        groups = cores / parts;
    }
    __aicore__ inline unsigned Begin(unsigned part) const { return tiles * part / parts; }
    __aicore__ inline unsigned End(unsigned part) const { return tiles * (part + 1) / parts; }
};

// The first 512 positions provide native mode-2 shortage fillers. Candidate
// blocks inside that prefix are represented there exactly once.
__aicore__ inline int OriginalBlock(unsigned slotBlock, int candidate, unsigned visible)
{
    int block = slotBlock < PREFIX / BLOCK ? static_cast<int>(slotBlock) : candidate;
    if (block < 0 || static_cast<unsigned>(block) >= (visible + BLOCK - 1) / BLOCK ||
        (slotBlock >= PREFIX / BLOCK && block < static_cast<int>(PREFIX / BLOCK))) {
        return -1;
    }
    return block;
}
} // namespace QLIV2Candidate
#endif
