# Copyright (c) 2023, Alibaba Inc.
# All right reserved.
#
# Author: Junqi Hu <hujunqi.hjq@alibaba-inc.com>
# Created: 2023/09/20
# Description:
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import adagrad
from tensorflow.core.framework.embedding import config_pb2
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training import training_util
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import kv_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.grappler import tf_optimizer
from collections import defaultdict
from tensorflow.python.ops import gen_kv_variable_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.kernel_tests.signal import test_util as test_main

class EmbeddingVariableTest(test_util.TensorFlowTestCase):
#   def test(checkpoint_directory):
#     print("testEmbeddingVariableForSaveAndRestoreForSingleTier")
#     with ops.Graph().as_default() as g, ops.device('/cpu:0'):
#       var = variable_scope.get_embedding_variable("var_1",
#               embedding_dim = 3,
#               initializer=init_ops.ones_initializer(dtypes.float32),
#               partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2),
#               ev_option = variables.EmbeddingVariableOption(
#                   storage_option=variables.StorageOption(
#                       storage_type=config_pb2.StorageType.DRAM)))

#       emb = embedding_ops.embedding_lookup(var,
#                                            math_ops.cast([0,1,2,5,6,7],
#                                            dtypes.int64))
#       fun = math_ops.multiply(emb, 0.0, name='multiply')
#       loss = math_ops.reduce_sum(fun, name='reduce_sum')
#       gs = training_util.get_or_create_global_step()
#       opt = adagrad.AdagradOptimizer(0.1)
#       g_v = opt.compute_gradients(loss)
#       train_op = opt.apply_gradients(g_v, gs)
#       saver = saver_module.Saver(sharded=True)
#       init = variables.global_variables_initializer()
#       with self.test_session() as sess:
#         sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
#         sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
#         sess.run([init])
#         rewritten_graph = test_main.grappler_optimize(g, [train_op])
#         sess.run(train_op)
#         emb_ori = sess.run(emb)
#         save_path = saver.save(sess, os.path.join(checkpoint_directory, "model.ckpt"), global_step=12345)
#         print(save_path)
#         for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
#           print('loading... ', name, shape)
  def _init_var_repartition_op(self, tmp):
    op_list = []
    ev_list = ops.get_collection(ops.GraphKeys.EMBEDDING_VARIABLES)
    variable_list = [x for x in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES) if x not in ev_list]
    import_storage_map = defaultdict(list)
    def process_var_name(ev):
      ev_name = ev.name.split(":")[0]
      device_name = ev.device
      idx = device_name.find("task:")
      post_name = device_name[idx:]
      post_idx = post_name.find("/")
      if post_idx == -1:
        device_id = int(device_name[idx+len("task:"):])
      else:
        device_id = int(post_name[len("task:"):post_idx])
      idx = ev_name.find("part_")
      if idx != -1:
        post_idx = ev_name[idx:].find("/")
        part_len = len("part_")
        if (post_idx == -1):
          pre_name = ev_name[:idx-1]
          var_idx = int(ev_name[idx+part_len:])
        else:
          pre_name = ev_name[:idx-1] + ev_name[idx:][post_idx:]
          var_idx = int(ev_name[idx:][part_len:post_idx])
        return True, pre_name, device_id, var_idx
      else:
        new_op = None
        op_name = ev.device
        if "task:2" in op_name:
          with ops.device("/job:ps/task:0/device:CPU:0"):
            # p = variables.VariableV1([1.0], dtype=dtypes.float32, name=ev.name[:-2])
            self.tmp_value_list.append(ev.read_value())
            a = array_ops.placeholder(dtypes.float32, name=ev.name[:-2]+"/placeholder")
            self.init_ph_list.append(a)
            with ops.name_scope("elastic_import"):
              self.init_op_list.append(state_ops.assign(ev, a, validate_shape=False))
        return False, new_op, 0, 0

    for var in variable_list:
      flag, pre_name, device_id, var_idx = process_var_name(var)
      print(pre_name, " -- ", var_idx)
      if flag:
        import_storage_map[pre_name].append((device_id, var_idx, var))
      elif pre_name is not None:
        op_list.append(pre_name)

    graph = ops.get_default_graph()
    for var_name, var_list in import_storage_map.items():
      var_list.sort(key= lambda x: x[0])
      var_read = [var[2] for var in var_list]
      # try:
      #   read_value = graph.get_tensor_by_name(var_name+"/ConcatPartitions/concat:0")
      # except:
      with ops.name_scope("elastic_import"):
        read_value = array_ops.concat(var_read, axis=0) #partition_axis
      for idx, var_meta in enumerate(var_list):
        if var_meta[0] != var_meta[1]:
          if resource_variable_ops.is_resource_variable(var_list[idx][2]):
            op_list.append(gen_kv_variable_ops.re_assign_resource(var_list[idx][2], read_value, self.partition_num_ph, idx, len(var_list)))
          else:
            op_list.append(gen_kv_variable_ops.re_assign(var_list[idx][2]._ref(), read_value, self.partition_num_ph, idx, len(var_list)))
        else:
          # read_value = array_ops.concat(var_read, axis=0) #partition_axis
          if resource_variable_ops.is_resource_variable(var_list[idx][2]):
            op_list.append(gen_kv_variable_ops.re_assign_resource(var_list[idx][2], read_value, self.partition_num_ph, idx, len(var_list)))
          else:
            op_list.append(gen_kv_variable_ops.re_assign(var_list[idx][2]._ref(), read_value, self.partition_num_ph, idx, len(var_list)))
    print(op_list)
    return op_list

  def _init_repartition_op(self, tmp):
    self.partition_num_ph = array_ops.placeholder(dtypes.int32, name='partition_num')
    op_list = []
    ev_list = ops.get_collection(ops.GraphKeys.EMBEDDING_VARIABLES)
    print(ev_list)
    import_storage_map = defaultdict(lambda: defaultdict(list))
    def process_ev_name(ev):
      ev_name = ev.name.split(":")[0]
      idx = ev_name.find("part_")
      post_idx = ev_name[idx:].find("/")
      if (post_idx == -1):
        pre_name = ev_name[:idx]
      else:
          pre_name = ev_name[:idx] + ev_name[idx:][post_idx:]
      return pre_name

    for ev in ev_list:
      is_partitioned_ev = not isinstance(ev._save_slice_info, str)
      save_slice_info = ev._save_slice_info
      partition_num = save_slice_info.full_shape[0] if is_partitioned_ev else 1
      pre_name = process_ev_name(ev)
      import_storage_map[pre_name]["keys"] = [None for _ in range(partition_num)]
      import_storage_map[pre_name]["values"]  = [None for _ in range(partition_num)]
      import_storage_map[pre_name]["versions"] = [None for _ in range(partition_num)]
      import_storage_map[pre_name]["freqs"] = [None for _ in range(partition_num)]

    for ev in ev_list:
      is_partitioned_ev = not isinstance(ev._save_slice_info, str)
      partition_id = 0
      partition_num = 1
      save_slice_info = ev._save_slice_info
      pre_name = process_ev_name(ev)
      key_type = dtypes.as_dtype(ev.handle.op.get_attr("Tkeys"))
      dtype = dtypes.as_dtype(ev.handle.op.get_attr("dtype"))
      if save_slice_info is not None:
        partition_id = save_slice_info.var_offset[0] if is_partitioned_ev else 0
        partition_num = save_slice_info.full_shape[0] if is_partitioned_ev else 1
      unneeded_ids, unneeded_values, unneeded_versions, unneeded_freqs = gen_kv_variable_ops.filter_storage(ev.handle, self.partition_num_ph, key_type, dtype, partition_id=partition_id)
      import_storage_map[pre_name]["keys"][partition_id] = unneeded_ids
      import_storage_map[pre_name]["values"][partition_id] = unneeded_values
      import_storage_map[pre_name]["versions"][partition_id] = unneeded_versions
      import_storage_map[pre_name]["freqs"][partition_id] = unneeded_freqs

    for ev in ev_list:
      pre_name = process_ev_name(ev)
      is_partitioned_ev = not isinstance(ev._save_slice_info, str)
      partition_id = 0
      save_slice_info = ev._save_slice_info
      if save_slice_info is not None:
        partition_id = save_slice_info.var_offset[0] if is_partitioned_ev else 0
      imported_keys = [import_storage_map[pre_name]["keys"][i] for i in range(len(import_storage_map[pre_name]["keys"])) if i != partition_id]
      imported_values = [import_storage_map[pre_name]["values"][i] for i in range(len(import_storage_map[pre_name]["values"])) if i != partition_id]
      imported_versions = [import_storage_map[pre_name]["versions"][i] for i in range(len(import_storage_map[pre_name]["versions"])) if i != partition_id]
      imported_freqs = [import_storage_map[pre_name]["freqs"][i] for i in range(len(import_storage_map[pre_name]["freqs"])) if i != partition_id]
      op_list.append(gen_kv_variable_ops.import_storage(ev.handle, imported_keys, imported_values,
                                                          imported_versions, imported_freqs, partition_id=partition_id))

    return op_list
    
  def test(self):
    checkpoint_directory = self.get_temp_dir()
    os.environ["ENABLE_ELASTIC"] = "True"
    os.environ["TF_CONFIG"] = "{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"ps\", \"index\": 0}}"
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2),
              ev_option = variables.EmbeddingVariableOption(
                  storage_option=variables.StorageOption(
                      storage_type=config_pb2.StorageType.DRAM)))

      emb = embedding_ops.embedding_lookup(var,
                                           math_ops.cast([0,1,2,5,6,7],
                                           dtypes.int64))
      fun = math_ops.multiply(emb, 0.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v, gs)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        tmp = []
        self.op_list = self._init_repartition_op(tmp)
        self.op_list.extend(self._init_var_repartition_op(tmp))
        with ops.control_dependencies(self.op_list):
          self.import_op = [control_flow_ops.no_op("elastic_subgraph_import")]
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        rewritten_graph = test_main.grappler_optimize(g, [train_op])
        sess.run(train_op)
        emb_ori = sess.run(emb)
        save_path = saver.save(sess, os.path.join(checkpoint_directory, "model.ckpt"), global_step=12345)
        print(save_path)
        for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
          print('loading... ', name, shape)

  def test_1(self):
    checkpoint_directory = self.get_temp_dir()
    os.environ["ENABLE_ELASTIC"] = "True"
    os.environ["TF_CONFIG"] = "{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"ps\", \"index\": 0}}"
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2),
              ev_option = variables.EmbeddingVariableOption(
                  storage_option=variables.StorageOption(
                      storage_type=config_pb2.StorageType.DRAM)))

      emb = embedding_ops.embedding_lookup(var,
                                           math_ops.cast([0,1,2,5,6,7],
                                           dtypes.int64))
      fun = math_ops.multiply(emb, 0.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v, gs)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        tmp = []
        self.op_list = self._init_repartition_op(tmp)
        self.op_list.extend(self._init_var_repartition_op(tmp))
        with ops.control_dependencies(self.op_list):
          self.import_op = [control_flow_ops.no_op("elastic_subgraph_import")]
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run(train_op)
        emb_ori = sess.run(emb)
        save_path = saver.save(sess, os.path.join(checkpoint_directory, "model.ckpt"), global_step=12345)
        print(save_path)
        for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
            print('loading... ', name, shape)

if __name__ == "__main__":
  googletest.main()