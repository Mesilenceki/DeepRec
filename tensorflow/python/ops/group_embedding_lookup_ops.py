"""Ops to use variables as resources."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_kv_variable_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

__all__ = ["group_embedding_var_lookup", "_GroupGatherGrad",
           "group_variable_lookup", "_GroupEmbeddingLookup"]

#for GPU EV group_lookup
def group_embedding_var_lookup(params,
                             unique_values,
                             unique_indices,
                             sp_indices,
                             sp_dense_shape,
                             sp_weights,
                             combiners,
                             dimensions,
                             ignore_weights,
                             ev_init_value = None):
  if ev_init_value is not None:
    default_value = ev_init_value
    is_use_default_value_tensor = True
  else:
    default_value = ops.convert_to_tensor(1.0)
    is_use_default_value_tensor = False
  if ignore_weights:
    sp_weight = ops.convert_to_tensor(1.0)
    sp_weights = [sp_weight for _ in range(len(sp_indices))]
  return gen_kv_variable_ops.group_embedding_var_lookup(params,
                                                        unique_values,
                                                        unique_indices,
                                                        sp_indices,
                                                        sp_dense_shape,
                                                        sp_weights,
                                                        default_value,
                                                        combiners,
                                                        dimensions,
                                                        ignore_weights,
                                                        is_use_default_value_tensor)

@ops.RegisterGradient("GroupEmbeddingVarLookup")
def _GroupGatherGrad(op, *grads):
  ev_num = op.get_attr("num_lookups")
  return_grads = []
  combiner = op.get_attr("combiner")
  dimension = op.get_attr("dimension")
  params = op.inputs[:ev_num]
  sp_values = op.inputs[ev_num:2*ev_num]
  sp_values_offset = op.outputs[ev_num:2*ev_num]
  tmp_grads = gen_kv_variable_ops.multi_kv_resource_gather_grad(grads[:ev_num],
                                                                params,
                                                                sp_values,
                                                                sp_values_offset,
                                                                dimension,
                                                                combiner)                                                            
  for i in range(ev_num):
    handle = op.inputs[i]
    while handle.op.type != "KvVarHandleOp":
      handle = handle.op.inputs[0]
    params_shape = ops.convert_to_tensor(
        tensor_shape.TensorShape(handle.op.get_attr("shape")))
    indice = op.inputs[ev_num+i]
    grad = tmp_grads[i]
    size = array_ops.expand_dims(array_ops.size(indice), 0)
    values_shape = array_ops.concat([size, params_shape[0:]], 0)
    grad = array_ops.reshape(grad, values_shape)
    indice = array_ops.reshape(indice, size)
    return_grads.append(ops.IndexedSlices(grad, indice, params_shape))
  for _ in range(ev_num*4 + 1):
    return_grads.append(None)
  return return_grads
  
#for GPU EV group_lookup
def group_variable_lookup(params,
                          unique_values,
                          unique_indices,
                          sp_indices,
                          sp_dense_shape,
                          sp_weights,
                          combiners,
                          dimensions,
                          ignore_weights,
                          default_id=None):
  if default_id is not None:
    default_value = default_id
  else:
    default_value = ops.convert_to_tensor(0.0)

  is_use_default_value_tensor = True

  if ignore_weights:
    sp_weight = ops.convert_to_tensor(1.0)
    sp_weights = [sp_weight for _ in range(len(sp_indices))]
    
  return gen_kv_variable_ops.group_variable_lookup(params,
                                                  unique_values,
                                                  unique_indices,
                                                  sp_indices,
                                                  sp_dense_shape,
                                                  sp_weights,
                                                  default_value,
                                                  combiners,
                                                  dimensions,
                                                  ignore_weights,
                                                  is_use_default_value_tensor)

@ops.RegisterGradient("GroupVariableLookup")
def _GroupEmbeddingLookup(op, *grads):
  ev_num = op.get_attr("num_lookups")
  combiner = op.get_attr("combiner")
  dimension = op.get_attr("dimension")
  return_grads = []
  params = op.inputs[:ev_num]
  sp_values = op.inputs[ev_num:2*ev_num]
  sp_indices = op.inputs[ev_num*3: ev_num*4]
  sp_values_offset = op.outputs[ev_num:2*ev_num]
  tmp_grads = gen_kv_variable_ops.multi_embedding_sparse_look_up_grad(grads[:ev_num],
                                                                      params,
                                                                      sp_values,
                                                                      sp_indices,
                                                                      sp_values_offset,
                                                                      dimension,
                                                                      combiner)
  for i in range(ev_num):
    params = op.inputs[i]
    with ops.colocate_with(params):
      params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
      params_shape = math_ops.cast(params_shape, dtypes.int32)
    unique_indice = op.inputs[ev_num*2+i]
    grad = tmp_grads[i]
    size = array_ops.expand_dims(array_ops.size(unique_indice), 0)
    values_shape = array_ops.concat([size, params_shape[1:]], 0)
    grad = array_ops.reshape(grad, values_shape)
    unique_indice = array_ops.reshape(unique_indice, size)
    return_grads.append(ops.IndexedSlices(grad, unique_indice, params_shape))
  for _ in range(ev_num*5+1):
    return_grads.append(None)
  return return_grads