#ifndef __COMPUTE_UTILS_HEADER__
#define __COMPUTE_UTILS_HEADER__

#ifndef GROUP_SIZE
    #error "ComputeUtils.hlsl requires definition of GROUP_SIZE"
#endif
#ifndef GROUP_SIZE_LOG_2
    #if GROUP_SIZE == 32
        #define GROUP_SIZE_LOG_2 5
    #elif GROUP_SIZE == 64
        #define GROUP_SIZE_LOG_2 6
    #elif GROUP_SIZE == 128
        #define GROUP_SIZE_LOG_2 7
    #elif GROUP_SIZE == 256
        #define GROUP_SIZE_LOG_2 8
    #else
        #error "ComputeUtils.hlsl requires definition of GROUP_SIZE_LOG_2, which must be the log2 of GROUP_SIZE"
    #endif
#endif

#define DWORD_BIT_SIZE_LOG2 5
#define DWORD_BIT_SIZE (1 << 5)
#define BIT_MASK_SIZE ((GROUP_SIZE + DWORD_BIT_SIZE - 1)/ DWORD_BIT_SIZE)

namespace ComputeUtils
{

groupshared uint gs_BitMask[BIT_MASK_SIZE];

void PrefixBitSum(
    uint groupThreadIndex,
    bool bitValue,
    out uint outOffset,
    out uint outCount)
{
    if (groupThreadIndex < BIT_MASK_SIZE)
        gs_BitMask[groupThreadIndex] = 0;

    GroupMemoryBarrierWithGroupSync();

    uint maskOffset = groupThreadIndex >> DWORD_BIT_SIZE_LOG2;
    uint maskBit = (groupThreadIndex & (DWORD_BIT_SIZE - 1));
    uint mask = 1u << maskBit;

    [branch]
    if (bitValue)
    {
        uint unused;
        InterlockedOr(gs_BitMask[maskOffset], mask, unused);
    }

    GroupMemoryBarrierWithGroupSync();

    outOffset = 0;
    if (bitValue)
    {
        for (uint i = 0; i < maskOffset; ++i)
            outOffset += countbits(gs_BitMask[i]);
        uint v = gs_BitMask[maskOffset];
        outOffset += countbits((mask - 1u) & v);
    }

    outCount = 0;
    {
        [unroll]
        for (uint i = 0; i < BIT_MASK_SIZE; ++i)
            outCount += countbits(gs_BitMask[i]);
    }
}

groupshared uint gs_PrefixCache[GROUP_SIZE];

void PrefixExclusive(
    uint groupThreadIndex,
    uint value,
    out uint outOffset,
    out uint outCount)
{
    gs_PrefixCache[groupThreadIndex] = value;

    GroupMemoryBarrierWithGroupSync();

    for (uint i = 1; i < GROUP_SIZE; i <<= 1)
    {
        uint sampleVal = groupThreadIndex >= i ? gs_PrefixCache[groupThreadIndex - i] : 0u;

        GroupMemoryBarrierWithGroupSync();

        gs_PrefixCache[groupThreadIndex] += sampleVal;

        GroupMemoryBarrierWithGroupSync();
    }

    outOffset = gs_PrefixCache[groupThreadIndex] - value;
    outCount = gs_PrefixCache[GROUP_SIZE - 1];
}

void PrefixInclusive(
    uint groupThreadIndex,
    uint value,
    out uint outOffset,
    out uint outCount)
{
    PrefixExclusive(groupThreadIndex, value, outOffset, outCount);
    outOffset += value;
}

uint CalculateGlobalStorageOffset(
    RWByteAddressBuffer counterBuffer,
    uint groupThreadIndex,
    bool bitValue,
    out uint totalCount)
{
    uint localOffset;
    PrefixBitSum(groupThreadIndex, bitValue, localOffset, totalCount);

    if (groupThreadIndex == 0 && totalCount > 0)
    {
        uint globalOffset = 0;
        counterBuffer.InterlockedAdd(0, totalCount, globalOffset);
        gs_BitMask[0] = globalOffset;
    }

    GroupMemoryBarrierWithGroupSync();

    return gs_BitMask[0] + localOffset;
}

uint CalculateGlobalStorageOffset(
    RWByteAddressBuffer counterBuffer,
    uint groupThreadIndex,
    bool bitValue)
{
    uint unused0;
    return CalculateGlobalStorageOffset(counterBuffer, groupThreadIndex, bitValue, unused0); 
}

uint CalculateGlobalValueStorageOffset(
    RWByteAddressBuffer counterBuffer,
    uint groupThreadIndex,
    uint valueCount,
    out uint totalCount)
{
    uint localOffset;
    PrefixExclusive(groupThreadIndex, valueCount, localOffset, totalCount);

    if (groupThreadIndex == 0 && totalCount > 0)
    {
        uint globalOffset = 0;
        counterBuffer.InterlockedAdd(0, totalCount, globalOffset);
        gs_PrefixCache[0] = globalOffset;
    }

    GroupMemoryBarrierWithGroupSync();

    return gs_PrefixCache[0] + localOffset;
}

uint CalculateGlobalValueStorageOffset(
    RWByteAddressBuffer counterBuffer,
    uint groupThreadIndex,
    uint valueCount)
{
    uint unused0;
    return CalculateGlobalValueStorageOffset(counterBuffer, groupThreadIndex, valueCount, unused0);
}

#define BROADCAST_COMPACT_IDX_PER_DWORD (32/GROUP_SIZE_LOG_2)
#define BROADCAST_COMPACT_DWORDS ((GROUP_SIZE + BROADCAST_COMPACT_IDX_PER_DWORD - 1)/BROADCAST_COMPACT_IDX_PER_DWORD)
groupshared uint gs_BroadcastLocalCount[GROUP_SIZE];
groupshared uint gs_BroadcastPackedIdxMap[BROADCAST_COMPACT_DWORDS];
 
bool BroadcastWork(
    uint groupThreadIndex,
    uint workCount,
    out uint outIndex,
    out uint outParentCount,
    out uint outParentGroupID)
{
    uint unused0;
    if (groupThreadIndex < BROADCAST_COMPACT_DWORDS)
        gs_BroadcastPackedIdxMap[groupThreadIndex] = 0;

    bool validWorkCount = workCount != 0;
    uint compactIdx;
    PrefixBitSum(groupThreadIndex, validWorkCount, compactIdx, unused0);
    if (validWorkCount)
        InterlockedOr(gs_BroadcastPackedIdxMap[compactIdx/BROADCAST_COMPACT_IDX_PER_DWORD], groupThreadIndex << ((compactIdx%BROADCAST_COMPACT_IDX_PER_DWORD)*GROUP_SIZE_LOG_2), unused0);

    uint groupOffset, groupCount; 
    PrefixInclusive(groupThreadIndex, workCount, groupOffset, groupCount);

    if (groupThreadIndex < BIT_MASK_SIZE)
        gs_BitMask[groupThreadIndex] = 0;

    int leftInLDS = max((int)GROUP_SIZE - (int)(groupOffset - workCount), 0);
    int actualWorkCount = min(leftInLDS, workCount);
    gs_BroadcastLocalCount[groupThreadIndex] = actualWorkCount;

    
    GroupMemoryBarrierWithGroupSync();

    [branch]
    if (actualWorkCount > 0 && groupOffset < GROUP_SIZE)
        InterlockedOr(gs_BitMask[groupOffset >> DWORD_BIT_SIZE_LOG2], 1u << (groupOffset & (DWORD_BIT_SIZE - 1u)), unused0);

    GroupMemoryBarrierWithGroupSync();

    uint compactSampleIdx = 0; 
    {
        uint groupDWOffset = groupThreadIndex >> DWORD_BIT_SIZE_LOG2;
        uint groupBitMask = 1u << (groupThreadIndex & (DWORD_BIT_SIZE - 1u));
        for (uint i = 0; i < groupDWOffset; ++i)
            compactSampleIdx += countbits(gs_BitMask[i]);
        uint v = gs_BitMask[groupDWOffset];
        compactSampleIdx += countbits(((groupBitMask - 1u) | groupBitMask) & v);
    }

    outParentGroupID = (gs_BroadcastPackedIdxMap[compactSampleIdx/BROADCAST_COMPACT_IDX_PER_DWORD] >> ((compactSampleIdx%BROADCAST_COMPACT_IDX_PER_DWORD)*GROUP_SIZE_LOG_2)) & ((1u << GROUP_SIZE_LOG_2) - 1u);

    uint parentPrefixExclusive = (outParentGroupID == 0 ? 0 : gs_PrefixCache[outParentGroupID - 1u]);
    outParentCount = gs_BroadcastLocalCount[outParentGroupID];
    outIndex = groupThreadIndex - parentPrefixExclusive;
    return outIndex < outParentCount;
}

}

#endif
