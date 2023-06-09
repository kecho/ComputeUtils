import coalpy.gpu as g
import numpy as nm

## Compute Utils Test Script ##

#variables
g_layout_workloads = [540, 340, 299, 689, 229, 770, 8]
g_round_count = 8450
#g_round_count = 500
g_instance_count = len(g_layout_workloads) * g_round_count 
g_total_output_count = sum(g_layout_workloads) * g_round_count;
print ("instances: %d output: %d" % (g_instance_count, g_total_output_count))
#######

def load_gpu_buffer(gpu_buffer, numpy_type='int'):
    request = g.ResourceDownloadRequest(resource = gpu_buffer)
    request.resolve()
    return nm.frombuffer(request.data_as_bytearray(), dtype=numpy_type)

def test_broadcast_work(workloads, rounds, output_list, output_list_count):
    expected_mask_workloads = [((1 << v) - 1) for v in workloads]
    instance_counts = len(expected_mask_workloads) * rounds 
    expected_masks = [expected_mask_workloads[i%len(expected_mask_workloads)] for i in range(0, instance_counts)]
    output_masks = [0 for i in range(0, instance_counts)]

    for i in range(0, output_list_count):
        output_masks[output_list[(i*2)]] |= 1 << int(output_list[(i*2) + 1])

    for i in range(0, len(output_masks)):
        if (expected_masks[i] != output_masks[i]):
            return False

    return True

g_init_shader = g.Shader(file = "ComputeUtilsTests.hlsl", main_function = "InitInputBufferMain")
g_init_shader.resolve()

g_distribute_shader = g.Shader(file = "ComputeUtilsTests.hlsl", main_function = "DistributeMain")
g_distribute_shader.resolve()

g_distribute_naive_shader = g.Shader(file = "ComputeUtilsTests.hlsl", main_function = "DistributeNaiveMain")
g_distribute_naive_shader.resolve()

work_layout_buffer = g.Buffer(type=g.BufferType.Raw, element_count=len(g_layout_workloads))
work_buffer = g.Buffer(type=g.BufferType.Raw, element_count = g_instance_count * 2)

counter_buffer = g.Buffer(type=g.BufferType.Raw, element_count = 1)
output_counter_buffer = g.Buffer(type=g.BufferType.Raw, element_count = 1)
output_buffer = g.Buffer(type=g.BufferType.Raw, element_count = g_total_output_count * 2)

output_naive_counter_buffer = g.Buffer(type=g.BufferType.Raw, element_count = 1)
output_naive_buffer = g.Buffer(type=g.BufferType.Raw, element_count = g_total_output_count * 2)

debug_counter_buffer = g.Buffer(type=g.BufferType.Raw, element_count = 1)
debug_buffer = g.Buffer(type=g.BufferType.Raw, element_count = 400 * 4)

cmd = g.CommandList()

g.begin_collect_markers()

cmd.upload_resource(
    source = g_layout_workloads, 
    destination = work_layout_buffer)

cmd.upload_resource(
    source = [0], 
    destination = counter_buffer)

cmd.upload_resource(
    source = [0], 
    destination = output_counter_buffer)

cmd.upload_resource(
    source = [0], 
    destination = output_naive_counter_buffer)

cmd.dispatch(
    constants = [ int(g_instance_count), int(len(g_layout_workloads)), int(0), int(0)],
    inputs = work_layout_buffer,
    outputs = work_buffer,
    shader = g_init_shader,
    x = int((g_instance_count +63)/64),
    y = 1,
    z = 1)

cmd.begin_marker("GroupDistribute")
cmd.dispatch(
    constants = [ int(g_instance_count), int(0), int(0), int(0)],
    inputs = work_buffer,
    outputs = [counter_buffer, output_buffer, output_counter_buffer, debug_buffer, debug_counter_buffer],
    shader = g_distribute_shader,
    x = 3500,
    y = 1,
    z = 1)
cmd.end_marker()

cmd.begin_marker("GroupDistributeNaive")
cmd.dispatch(
    constants = [ int(g_instance_count), int(0), int(0), int(0)],
    inputs = work_buffer,
    outputs = [counter_buffer, output_naive_buffer, output_naive_counter_buffer, debug_buffer, debug_counter_buffer],
    shader = g_distribute_naive_shader,
    x = int((g_instance_count + 63)/64),
    y = 1,
    z = 1)
cmd.end_marker()

g.schedule(cmd)

marker_results = g.end_collect_markers()
gpu_timestamps = load_gpu_buffer(marker_results.timestamp_buffer, nm.uint64)
perf_data = [(name, (gpu_timestamps[ets] - gpu_timestamps[bts])/marker_results.timestamp_frequency)  for (name, p, bts, ets) in marker_results.markers]
print (perf_data)

# Output
"""
output_counter_buffer_readback = load_gpu_buffer(output_counter_buffer)[0]
output_buffer_readback = load_gpu_buffer(output_buffer)
print("Distributed Received: %d" % output_counter_buffer_readback)
print(test_broadcast_work(g_layout_workloads, g_round_count, output_buffer_readback, output_counter_buffer_readback))

output_counter_buffer_readback = load_gpu_buffer(output_naive_counter_buffer)[0]
output_buffer_readback = load_gpu_buffer(output_naive_buffer)
print("Distributed Naive Received: %d" % output_counter_buffer_readback)
print(test_broadcast_work(g_layout_workloads, g_round_count, output_buffer_readback, output_counter_buffer_readback))
"""
