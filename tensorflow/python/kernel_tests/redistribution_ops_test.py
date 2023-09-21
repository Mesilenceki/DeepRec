"""Tests for tensorflow.ops.tf.ReAssign*."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import gen_kv_variable_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import init_ops
from tensorflow.python.platform import test

class ReAssignOpTest(test.TestCase):
  @test_util.run_deprecated_v1
  def testScaleUp(self):
    with self.cached_session() as sess:
      partitioner = partitioned_variables.fixed_size_partitioner(2)
      var = variable_scope.get_variable("test_var", shape=(12, 8), initializer=init_ops.random_uniform_initializer(),partitioner=partitioner, use_resource=False)
      var_list = var._get_variable_list()
      data_0 = array_ops.fill([6, 8], 0.5)
      data_1 = array_ops.fill([6, 8], 1.5)
      sess.run(var_list[0].assign(data_0))
      print(var_list[0].shape)
      sess.run(var_list[1].assign(data_1))
      assign_value = var.as_tensor()
      reassign_value_0 = gen_kv_variable_ops.re_assign(var_list[0], assign_value, 3, 0, 2)
      reassign_value_1 = gen_kv_variable_ops.re_assign(var_list[1], assign_value, 3, 1, 2)
      sess.run([reassign_value_0, reassign_value_1])
      val_0, val_1 = sess.run([var_list[0], var_list[1]])
      print(val_0)
      for val_list in val_0.tolist():
        for val in val_list:
          self.assertEqual(val, 0.5)
      for val_list in val_1.tolist()[0:2]:
        for val in val_list:
          self.assertEqual(val, 0.5)
      for val_list in val_1.tolist()[2:]:
        print(val_list)
        for val in val_list:
          self.assertEqual(val, 1.5)
  
  @test_util.run_deprecated_v1
  def testScaleDown(self):
    with self.cached_session() as sess:
      partitioner = partitioned_variables.fixed_size_partitioner(3)
      var = variable_scope.get_variable("test_var", shape=(12, 8), initializer=init_ops.random_uniform_initializer(),partitioner=partitioner, use_resource=False)
      var_list = var._get_variable_list()
      data_0 = array_ops.fill([4, 8], 0.5)
      data_1 = array_ops.fill([4, 8], 1.5)
      data_2 = array_ops.fill([4, 8], 2.5)
      sess.run(var_list[0].assign(data_0))
      sess.run(var_list[1].assign(data_1))
      sess.run(var_list[2].assign(data_2))
      assign_value = var.as_tensor()
      reassign_value_0 = gen_kv_variable_ops.re_assign(var_list[0], assign_value, 2, 0, 3)
      reassign_value_1 = gen_kv_variable_ops.re_assign(var_list[1], assign_value, 2, 1, 3)
      reassign_value_2 = gen_kv_variable_ops.re_assign(var_list[2], assign_value, 2, 2, 3)
      sess.run([reassign_value_0, reassign_value_1, reassign_value_2])
      val_0, val_1, val_2 = sess.run([var_list[0], var_list[1], var_list[2]])
      for val_list in val_0.tolist()[0:4]:
        for val in val_list:
          self.assertEqual(val, 0.5)
      for val_list in val_0.tolist()[4:]:
        for val in val_list:
          self.assertEqual(val, 1.5)
      for val_list in val_1.tolist()[0:2]:
        print(val_list)
        for val in val_list:
          self.assertEqual(val, 1.5)
      for val_list in val_1.tolist()[2:]:
        print(val_list)
        for val in val_list:
          self.assertEqual(val, 2.5)
  
  def testResourceScaleUp(self):
    with self.cached_session() as sess:
      partitioner = partitioned_variables.fixed_size_partitioner(2)
      var = variable_scope.get_variable("test_var", shape=(12, 8), initializer=init_ops.random_uniform_initializer(),partitioner=partitioner, use_resource=True)
      var_list = var._get_variable_list()
      data_0 = array_ops.fill([6, 8], 0.5)
      data_1 = array_ops.fill([6, 8], 1.5)
      sess.run(var_list[0].assign(data_0))
      print(var_list[0].shape)
      sess.run(var_list[1].assign(data_1))
      assign_value = var_list[0].sparse_read(math_ops.range(math_ops.cast(var.shape[0].value/3, dtypes.int32), math_ops.cast(var.shape[0].value/2, dtypes.int32)))
      print(sess.run(assign_value))
      reassign_value_0 = var_list[0], assign_value, 3, 0, 2)
      reassign_value_1 = gen_kv_variable_ops.re_assign(var_list[1], assign_value, 3, 1, 2)
      sess.run([reassign_value_0, reassign_value_1])
      val_0, val_1 = sess.run([var_list[0], var_list[1]])
      print(val_0)
      for val_list in val_0.tolist():
        for val in val_list:
          self.assertEqual(val, 0.5)
      for val_list in val_1.tolist()[0:2]:
        for val in val_list:
          self.assertEqual(val, 0.5)
      for val_list in val_1.tolist()[2:]:
        print(val_list)
        for val in val_list:
          self.assertEqual(val, 1.5)


if __name__ == "__main__":
  test.main()