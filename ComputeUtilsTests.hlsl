#define GROUP_SIZE 64
#include "ComputeUtils.hlsl"

ByteAddressBuffer g_workLayout : register(t0);
RWByteAddressBuffer g_outputWorkBuffer : register(u0);

cbuffer Constants : register(b0)
{
    uint g_InstanceCount;
    uint g_LayoutWorkloadsCount;
    uint2 g_padding;
}

[numthreads(GROUP_SIZE,1,1)]
void InitInputBufferMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= g_InstanceCount)
        return;

    uint workLayoutCount = g_workLayout.Load(0);
    uint workID = dispatchThreadID.x;
    uint workCount = g_workLayout.Load((workID % g_LayoutWorkloadsCount) << 2);
    g_outputWorkBuffer.Store2(dispatchThreadID.x << 3, uint2(workID, workCount));
}

ByteAddressBuffer g_InputBuffer : register(t0);
RWByteAddressBuffer g_CounterBuffer : register(u0);
RWByteAddressBuffer g_OutputBuffer : register(u1);
RWByteAddressBuffer g_OutputCounter : register(u2);
RWByteAddressBuffer g_DebugBuffer : register(u3);
RWByteAddressBuffer g_DebugBufferCounter : register(u4);

groupshared bool gs_GroupActive;
groupshared bool gs_LoadInstanceBatch;
groupshared uint gs_GroupInputOffset;
groupshared uint gs_InstanceIDCache[GROUP_SIZE];
groupshared uint gs_InstanceIDWorkLeft[GROUP_SIZE];
groupshared uint gs_InstanceIDWorkOffset[GROUP_SIZE];

[numthreads(GROUP_SIZE, 1, 1)]
void DistributeMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint groupThreadIndex : SV_GroupIndex)
{
    if (groupThreadIndex == 0)
    {
        gs_GroupActive = true;
        gs_LoadInstanceBatch = true;
    }

    GroupMemoryBarrierWithGroupSync();

    while (gs_GroupActive)
    {
        if (gs_LoadInstanceBatch)
        {
            if (groupThreadIndex == 0)
            {
                uint inputOffset;
                g_CounterBuffer.InterlockedAdd(0, GROUP_SIZE, inputOffset);
                gs_GroupInputOffset = inputOffset;
                gs_GroupActive = inputOffset < g_InstanceCount;
            }

            GroupMemoryBarrierWithGroupSync();

            uint sampleIndex = gs_GroupInputOffset + groupThreadIndex;
            uint2 data = sampleIndex < g_InstanceCount ? g_InputBuffer.Load2(sampleIndex << 3) : uint2(0,0);
            gs_InstanceIDCache[groupThreadIndex] = data.x;
            gs_InstanceIDWorkLeft[groupThreadIndex] = data.y;
            gs_InstanceIDWorkOffset[groupThreadIndex] = 0;
            gs_LoadInstanceBatch = false;

            GroupMemoryBarrierWithGroupSync();
        }

        if (!gs_GroupActive)
            return;

        uint workIndex;
        uint workParentCount;
        uint workParentID;
        bool validWork = ComputeUtils::BroadcastWork(
            groupThreadIndex,
            gs_InstanceIDWorkLeft[groupThreadIndex], 
            workIndex, workParentCount, workParentID);

        uint workOffset = workIndex + gs_InstanceIDWorkOffset[workParentID];

        uint outputCount;
        uint outputIndex = ComputeUtils::CalculateGlobalStorageOffset(g_OutputCounter, groupThreadIndex, validWork, outputCount);
        if (validWork)
            g_OutputBuffer.Store2(outputIndex << 3, uint2(gs_InstanceIDCache[workParentID], workOffset));

        if (workIndex == 0)
        {
            gs_InstanceIDWorkLeft[workParentID] -= workParentCount;
            gs_InstanceIDWorkOffset[workParentID] += workParentCount;
        }

        if (outputCount == 0)
            gs_LoadInstanceBatch = true;

        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(GROUP_SIZE, 1, 1)]
void DistributeNaiveMain(
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= g_InstanceCount)
        return;
    
    uint2 instanceJob = g_InputBuffer.Load2(dispatchThreadID.x << 3);
    uint outputOffset;
    g_OutputCounter.InterlockedAdd(0, instanceJob.y, outputOffset);
    for (uint i = 0; i < instanceJob.y; ++i)
        g_OutputBuffer.Store((outputOffset + i) << 3, uint2(instanceJob.x, i));
}

