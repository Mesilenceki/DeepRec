#!/usr/bin/env python

# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training import training
from tensorflow._api.v1 import train as train_v1
from tensorflow.python.ops import embedding_ops
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session as _monitored_session
from tensorflow.python.distribute import device_util
from tensorflow.python.training import server_lib
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.client import device_lib
from tensorflow.python.framework import ops

from tensorflow_estimator.python.estimator import estimator as _estimator_lib
from tensorflow_estimator.python.estimator import \
    run_config as run_config_lib
from tensorflow.python.saved_model.model_utils.export_utils import \
  EXPORT_TAG_MAP
from tensorflow.python.saved_model.model_utils.export_utils import \
  get_timestamped_export_dir
from tensorflow.python.saved_model.model_utils.export_utils import \
  SIGNATURE_KEY_MAP
    
import re
import abc
import os
import contextlib
import threading
import random as rn

import numpy as np
import horovod.tensorflow as hvd

class HvdContext(object):
  DEFAULT_DEVICE = '/job:localhost'

  _instance = None

  @classmethod
  def get(cls):
    r'''Get singleton.
    '''
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  @classmethod
  @contextlib.contextmanager
  def scope(cls, **kwargs):
    r'''Update params in context.
    '''
    prev_kwargs = {}
    try:
      c = cls.get()
      prev_kwargs = c.options.update(**kwargs)
      yield c
    finally:
      c.options.update(**prev_kwargs)
      del prev_kwargs

  @classmethod
  def get_tf_config(cls):
    r'''Get configuration from TF_CONFIG environment variable.
    '''
    tf_config = json.loads(os.getenv('TF_CONFIG', '{}'))
    if not tf_config:
      return None
    task = tf_config['task']
    cluster = tf_config['cluster']
    task_type = task['type']
    task_id = int(task['index'])
    tf_config_type = collections.namedtuple(
      'TfConfig', ['task_type', 'task_id', 'cluster'])
    return tf_config_type(task_type, task_id, cluster)
  
  @property
  def cluster_spec(self):
    r'''cluster spec.
    '''
    return self._cluster_spec

  @property
  def task_type(self):
    r'''job name of current server. `localhost` by default.
    '''
    return self._task_type

  @property
  def task_id(self):
    r'''task index of current server. 0 by default.
    '''
    return self._task_id

  @property
  def target(self):
    r'''target of current server.
    '''
    if not self._cluster_spec:
      return ''

    addr = self._cluster_spec.job_tasks(self._task_type)[self._task_id]
    return f'grpc://{addr}'

  @property
  def is_chief(self):
    r'''True if current server is chief worker.
    '''
    return self._is_chief

  @property
  def has_gpu(self):
    r'''True if current server has GPU.
    '''
    return self._num_gpus > 0

  @property
  def num_gpus(self):
    r'''Number of GPUs.
    '''
    return self._num_gpus

  def _update(self, task_type=None, task_id=None, cluster_spec=None,
              num_gpus=None):
    r'''Update parameters from cluster_spec.

    If task_type, task_id or cluster_spec is None, these arguments will not be
    changed.

    Args:
      task_type: (Optional.) name of current job. `localhost` by default.
      task_id: (Optional.) index of current task. 0 by default.
      cluster_spec: (Optional.) ClusterSpec object.
    '''
    tf_config = None
    try:
      tf_config = self.get_tf_config()
    except:  # pylint: disable=bare-except
      pass
    if tf_config:
      self._task_type = tf_config.task_type
      self._task_id = tf_config.task_id
      self._cluster_spec = server_lib.ClusterSpec(tf_config.cluster)
    else:
      self._task_type = 'localhost'
      self._task_id = 0
      self._cluster_spec = None
    if task_type:
      self._task_type = task_type
    if self._task_type not in ('localhost', 'chief', 'worker'):
      logging.info('No valid configuration for non-worker roles')
      return

    if task_id:
      self._task_id = task_id
    if cluster_spec:
      self._cluster_spec = cluster_spec
    if self._cluster_spec:
      self._cluster_spec = multi_worker_util.normalize_cluster_spec(
        self._cluster_spec)
      self._is_chief = False
      try:
        self._is_chief = multi_worker_util.is_chief(
          self._cluster_spec, self._task_type, self._task_id)
      except:  # pylint: disable=bare-except
        pass
    if num_gpus:
      self._num_gpus = num_gpus
    elif not self._num_gpus:
      num_gpus = 0
      num_gpus_config = config_pb2.ConfigProto()
      num_gpus_config.inter_op_parallelism_threads = 1
      num_gpus_config.intra_op_parallelism_threads = 1
      num_gpus_config.gpu_options.allow_growth = True
      for device in device_lib.list_local_devices(num_gpus_config):
        if device.device_type == 'GPU':
          num_gpus += 1
      self._num_gpus = num_gpus
    self._default_device = (
      f'/job:{self._task_type}/replica:0/task:{self._task_id}')
    self._local_cpu_device = device_util.canonicalize(
      '/device:CPU:0', default=self._default_device)
    if self._num_gpus == 0:
      self._local_devices = [self._local_cpu_device]
    else:
      self._local_devices = [
        device_util.canonicalize(
          f'/device:GPU:{d}', default=self._default_device)
        for d in range(self._num_gpus)]

    local_world_size_str = os.getenv('LOCAL_WORLD_SIZE', '')
    if not local_world_size_str:
      self._local_world_size = len(self._local_devices)  # pylint: disable=protected-access
    else:
      self._local_world_size = int(local_world_size_str)

    if not self._cluster_spec:
      self._devices = list(self._local_devices)
      return
    task_indices = []
    try:
      task_defs = dict(enumerate(self._cluster_spec.job_tasks(self._task_type)))
      task_indices = sorted(task_defs)
    except:  # pylint: disable=bare-except
      pass
    worker_indices = []
    try:
      worker_defs = dict(enumerate(self._cluster_spec.job_tasks('worker')))
      worker_indices = sorted(worker_defs)
    except:  # pylint: disable=bare-except
      pass
    chief_indices = []
    try:
      chief_defs = dict(enumerate(self._cluster_spec.job_tasks('chief')))
      chief_indices = sorted(chief_defs)
    except:  # pylint: disable=bare-except
      pass
    self._cpu_devices = [
      device_util.resolve(f'/job:{self._task_type}/task:{t}/device:CPU:0')
      for t in task_indices]
    if self._num_gpus == 0:
      self._devices = self._cpu_devices
      if self._task_type == 'worker':
        chief_devices = [
          device_util.resolve(f'/job:chief/task:{t}/device:CPU:0')
          for t in chief_indices]
        self._devices = chief_devices + self._devices
      elif self._task_type == 'chief':
        self._devices += [
          device_util.resolve(f'/job:worker/task:{t}/device:CPU:0')
          for t in worker_indices]
      return
    self._devices = [
      device_util.resolve(f'/job:{self._task_type}/task:{t}/device:GPU:{g}')
      for t in task_indices for g in range(self._num_gpus)]
    if self._task_type == 'worker':
      chief_devices = [
        device_util.resolve(f'/job:chief/task:{t}/device:GPU:{g}')
        for t in chief_indices for g in range(self._num_gpus)]
      self._devices = chief_devices + self._devices
    elif self._task_type == 'chief':
      self._devices += [
        device_util.resolve(f'/job:worker/task:{t}/device:GPU:{g}')
        for t in worker_indices for g in range(self._num_gpus)]

  def __init__(self):
    r'''Construct a server specification.
    '''
    self._task_type = 'localhost'
    self._task_id = 0
    self._cluster_spec = None
    self._is_chief = True
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '')
    if visible_devices:
      self._num_gpus = len(visible_devices.split(','))
    else:
      self._num_gpus = 1
    self._update()
    self._saving_listener_registry = {}

class GraphRewriting(object):  # pylint: disable=useless-object-inheritance
  r'''Python API rewriting.
  '''
  _lock = threading.Lock()
  _stack_depth = 0
  _registry = {}
  _registry_keys = []

  @classmethod
  def register(cls, rewriting):
    r'''Register implementation.

    Args:
      rewriting: Implementation class to register.
    '''
    if not issubclass(rewriting, cls):
      raise ValueError(f'{rewriting} must be a subclass of GraphRewriting')
    if rewriting.__name__ not in cls._registry:
      cls._registry_keys.append(rewriting.__name__)
    cls._registry[rewriting.__name__] = rewriting()

  @classmethod
  @contextlib.contextmanager
  def scope(cls, **kwargs):
    r'''Context manager that patches Python APIs.
    '''
    seed = kwargs.pop('seed', None)
    if seed is not None:
      rn.seed(seed)
      np.random.seed(seed)
      random_seed.set_random_seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)
      os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    with contextlib.nullcontext() as ctx:
      try:
        with cls._lock:
          cls._stack_depth += 1
          if cls._stack_depth <= 1:
            for name in cls._registry_keys:
              cls._registry[name].begin()
        yield ctx

      finally:
        with cls._lock:
          if cls._stack_depth <= 1:
            for name in reversed(cls._registry_keys):
              cls._registry[name].end()
          cls._stack_depth -= 1

  @abc.abstractmethod
  def begin(self):
    r'''Rewrites API.
    '''

  @abc.abstractmethod
  def end(self):
    r'''Revert API rewriting.
    ''' 

##################### Optimizer ##########################

def wraps_optimizer(
    cls,
    use_locking=False,
    sparse_as_dense=False,
    backward_passes_per_step=1,
    average_aggregated_gradients=False):
  r'''Decorator to create horovod optimizer class.

  Args:
    optimizer_type: The actual optimizer type that will be used to compute and
      apply the gradients. Must be one of the Optimizer classes.
    aggregation: Aggregate gradients inside `compute_gradients` or
      `apply_gradients`.

  Returns:
    hb_optimizer_type: The hybridbackend optimizer type for `optimizer_type`
  '''

  if isinstance(cls, hvd._DistributedOptimizer):
    return cls
  else:
    def horovod_optimizer(learning_rate, **kwargs):
        # horovod_params = {""}
        opt = cls(learning_rate * hvd.size(), **kwargs)
        return hvd.DistributedOptimizer(opt)
    return horovod_optimizer


class OptimizerRewriting(GraphRewriting):
  r'''Rewriting optimizers.
  '''
  def __init__(self):
    super().__init__()
    self._prev_optimizers = {}

  def begin(self):
    r'''Rewrites API.
    '''
    for k, c in training.__dict__.items():
      if (isinstance(c, type)
          and issubclass(c, training.Optimizer)
          and c not in (
            training.Optimizer,
            training.SyncReplicasOptimizer)):
        self._prev_optimizers[k] = c
        wrapped = wraps_optimizer(c)
        setattr(training, k, wrapped)
        setattr(train_v1, k, wrapped)

  def end(self):
    r'''Revert API rewriting.
    '''
    for c, opt in self._prev_optimizers.items():
      setattr(training, c, opt)
      setattr(train_v1, c, opt)


GraphRewriting.register(OptimizerRewriting)


##################### MonitoredTrainingSession ##########################

def wraps_session_config(session_config, *args, **kwargs):
  r'''Wraps ConfigProto for distributed training.
  '''
  if not session_config:
    kwargs.setdefault('allow_soft_placement', True)
    session_config = config_pb2.ConfigProto(*args, **kwargs)
  session_config.gpu_options.allow_growth = True
  session_config.gpu_options.force_gpu_compatible = True
  # Horovod: pin GPU to be used to process local rank (one GPU per process)
  session_config.gpu_options.visible_device_list = str(hvd.local_rank())

  if not session_config.device_filters:
    cluster_spec = HvdContext.get().cluster_spec
    task_type = HvdContext.get().task_type
    task_id = HvdContext.get().task_id
    if cluster_spec is None:
      session_config.isolate_session_state = True
      return session_config
    session_config.isolate_session_state = False
    del session_config.device_filters[:]
    if task_type in ('chief', 'worker'):
      session_config.device_filters.extend([
        '/job:ps', '/job:chief', f'/job:{task_type}/task:{task_id}'])
      session_config.experimental.collective_group_leader = (
        multi_worker_util.collective_leader(cluster_spec, task_type, task_id))
    elif task_type == 'evaluator':
      session_config.device_filters.append(f'/job:{task_type}/task:{task_id}')
  return session_config

def wraps_server(cls):
  r'''Decorator to create hybridbackend server class.
  '''
  if issubclass(cls, HybridBackendServerBase):
    return cls

  class HybridBackendServer(cls, HybridBackendServerBase):
    r'''An in-process TensorFlow server, for use in distributed training.
    '''
    _default = None

    @classmethod
    def get(class_):
      if class_._default is None:
        class_._default = class_(None)
      return class_._default

    def __init__(self, server_or_cluster_def, **kwargs):
      r'''Creates a new server with the given definition.
      '''
      if server_or_cluster_def is None:
        server_or_cluster_def = HvdContext.get().cluster_spec
        kwargs['job_name'] = HvdContext.get().task_type
        kwargs['task_index'] = HvdContext.get().task_id
      if server_or_cluster_def is None:
        self._is_local = True
        return
      self._is_local = False
      kwargs['config'] = wraps_session_config(kwargs.pop('config', None))
      super().__init__(server_or_cluster_def, **kwargs)

    @property
    def target(self):
      r'''Returns the target for asession to connect to this server.
      '''
      if self._is_local:
        return ''
      return super().target

    def monitored_session(self, **kwargs):
      r'''Creates a `MonitoredSession` for training.
      '''
      with scope():
        return _monitored_session.MonitoredTrainingSession(
          master=self.target, **kwargs)

  return HybridBackendServer


Server = wraps_server(server_lib.Server)


def wraps_monitored_training_session(fn):
  r'''Decorator to create wrapped monitored training session.
  '''
  if hasattr(fn, 'wrapped_fn'):
    return fn

  def HorovodMonitoredTrainingSession(*args, **kwargs):  # pylint: disable=invalid-name
    r'''Creates a `MonitoredSession` for training.
    '''
    checkpoint_dir = kwargs.get('checkpoint_dir', None)
    summary_dir = kwargs.get('summary_dir', None)
    summary_dir = summary_dir or checkpoint_dir
    scaffold = kwargs.pop('scaffold', _monitored_session.Scaffold())
    kwargs['scaffold'] = scaffold
    hooks = kwargs.pop('hooks', [])
    # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
    # from rank 0 to all other processes.To ensure consistent
    # initialization of all workers when training is started with random weights
    # or restored from a checkpoint.
    hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    chief_only_hooks = kwargs.pop('chief_only_hooks', [])
    chief_only_hooks = list(chief_only_hooks)
    kwargs['hooks'] = hooks
    kwargs['chief_only_hooks'] = chief_only_hooks
    kwargs['config'] = wraps_session_config(kwargs.pop('config', None))
    kwargs['is_chief'] = True
    args = list(args)
    if args:
      master = args[0]
      if not master:
        master = Server.get().target
      args[0] = master
    else:
      master = kwargs.pop('master', None)
      if not master:
        master = Server.get().target
      kwargs['master'] = master

    prev_monitored_session = _monitored_session.MonitoredSession
    _monitored_session.MonitoredSession = prev_monitored_session
    # wraps_monitored_session(
    #   prev_monitored_session,
    #   keep_checkpoint_max=kwargs.pop('keep_checkpoint_max', 5),
    #   keep_checkpoint_every_n_hours=kwargs.pop(
    #     'keep_checkpoint_every_n_hours', 10000.0))
    sess = fn(*args, **kwargs)
    _monitored_session.MonitoredSession = prev_monitored_session
    return sess

  HorovodMonitoredTrainingSession.wrapped_fn = fn
  return HorovodMonitoredTrainingSession

class SessionRewriting(GraphRewriting):
  r'''Rewriting monitored training session.
  '''
  def __init__(self):
    super().__init__()
    self._prev_sess_fn = None

  def begin(self):
    r'''Rewrites API.
    '''
    self._prev_sess_fn = _monitored_session.MonitoredTrainingSession
    _monitored_session.MonitoredTrainingSession = (
      wraps_monitored_training_session(
        _monitored_session.MonitoredTrainingSession))
    training.MonitoredTrainingSession = (
      _monitored_session.MonitoredTrainingSession)
    train_v1.MonitoredTrainingSession = (
      _monitored_session.MonitoredTrainingSession)

  def end(self):
    r'''Revert API rewriting.
    '''
    train_v1.MonitoredTrainingSession = self._prev_sess_fn
    training.MonitoredTrainingSession = self._prev_sess_fn
    _monitored_session.MonitoredTrainingSession = self._prev_sess_fn


GraphRewriting.register(SessionRewriting)

##################### Saver ##############################
# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

r'''Save and restore replicated and sharded variables.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver
from tensorflow.python.util import nest

try:
  from tensorflow.python.training.saving.saveable_object_util import \
    op_list_to_dict
except ImportError:
  op_list_to_dict = saver.BaseSaverBuilder.OpListToDict

from hybridbackend.tensorflow.framework.ops import GraphKeys
from hybridbackend.tensorflow.framework.ops import ModeKeys
from hybridbackend.tensorflow.framework.rewriting import SessionRunRewriting


class HybridBackendSaverBuilderBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of sharded saver builders.
  '''

def wraps_saver_builder(cls):
  r'''Wraps a saver builder to support hybrid parallelism.
  '''
  if issubclass(cls, HybridBackendSaverBuilderBase):
    return cls

  class HybridBackendSaverBuilder(cls, HybridBackendSaverBuilderBase):
    r'''Wrapped SaverBuilder with support for hybrid parallelism.
    '''
    def __init__(self, *args, **kwargs):
      name = kwargs.pop('name', 'hybrid_backend_saver_builder')
      self._restoreable_saveables = None
      self._rank = HvdContext.get().rank
      self._world_size = HvdContext.get().world_size
      super().__init__(*args, **kwargs)
      with ops.device('/cpu:0'):
        self._local_barrier = data_flow_ops.Barrier(
          [dtypes.bool],
          shared_name=f'{name}_local_barrier')
        self._global_barrier = data_flow_ops.Barrier(
          [dtypes.bool],
          shared_name=f'{name}_global_barrier')

    @property
    def rank(self):
      return self._rank

    @property
    def world_size(self):
      return self._world_size

    def _AddShardedSaveOpsForV2(self, checkpoint_prefix, per_device):
      r'''Add ops to save the params in parallel, for the V2 format.

      Args:
        checkpoint_prefix: scalar String Tensor.  Interpreted *NOT AS A
          FILENAME*, but as a prefix of a V2 checkpoint;
        per_device: A list of (device, BaseSaverBuilder.VarToSave) pairs, as
          returned by _GroupByDevices().

      Returns:
        An op to save the variables, which, when evaluated, returns the prefix
          "<user-fed prefix>" only and does not include the sharded spec suffix.
      '''
      if self._world_size <= 1:
        return super()._AddShardedSaveOpsForV2(checkpoint_prefix, per_device)

      # Filter sharded saveables.
      sharded_saveables = []
      if self._rank != 0:
        sharded_saveables += ops.get_collection_ref(
          GraphKeys.SHARDED_VARIABLES)
        sharded_saveables += ops.get_collection_ref(
          GraphKeys.SHARDED_RESOURCES)
        sharded_saveables = [v for v in sharded_saveables if v is not None]
        sharded_saveables = op_list_to_dict(sharded_saveables).values()
        sharded_saveables = nest.flatten(sharded_saveables)

      # Save local partitions.
      checkpoint_uuid = uuid.uuid4()
      checkpoint_prefix_suffix = f'_temp_{checkpoint_uuid.hex}/part'
      tmp_checkpoint_prefix = string_ops.string_join(
        [checkpoint_prefix, checkpoint_prefix_suffix])

      num_shards = len(per_device)
      num_shards_tensor = constant_op.constant(num_shards)
      local_done = constant_op.constant([True], dtype=dtypes.bool)
      global_done = constant_op.constant(
        [True for _ in range(self._world_size - 1)], dtype=dtypes.bool)
      save_ops = []
      filenames = []
      last_device = None
      empty_filename = ''

      for shard, (device, saveables) in enumerate(per_device):
        # Only sharded saveables need to save for non-chief workers
        if self._rank != 0:
          saveables = [
            s for s in saveables
            if s.op in sharded_saveables or s in sharded_saveables]
        logging.vlog(
          1,
          f'Saving {len(saveables)} saveables for shard {shard} at process '
          f'{self._rank}: {[s.name for s in saveables]}')
        last_device = device
        with ops.device(device):
          with ops.device('/cpu:0'):
            filename = empty_filename
            if saveables:
              filename = self.sharded_filename(
                tmp_checkpoint_prefix, shard, num_shards_tensor)
              save_ops.append(self._AddSaveOps(filename, saveables))
            filenames.append(filename)

      with ops.control_dependencies([x.op for x in save_ops]):
        with ops.device(last_device):
          with ops.device('/cpu:0'):
            notify_local_done = [
              self._local_barrier.insert_many(0, keys=[f], values=local_done)
              for f in filenames]
            _, ready_filenames, _ = self._local_barrier.take_many(
              self._world_size * len(filenames))
            notify_global_done = self._global_barrier.insert_many(
              0,
              keys=[str(i) for i in range(self._world_size - 1)],
              values=global_done)
            _, ready_ranks, _ = self._global_barrier.take_many(1)

            if self._rank == 0:
              ready_filenames_mask = math_ops.logical_not(
                string_ops.regex_full_match(ready_filenames, empty_filename))
              ready_filenames = array_ops.boolean_mask(
                ready_filenames, ready_filenames_mask)
              with ops.control_dependencies(notify_local_done):
                with ops.control_dependencies([ready_filenames]):
                  merge_files = gen_io_ops.merge_v2_checkpoints(
                    ready_filenames, checkpoint_prefix, delete_old_dirs=True)
                with ops.control_dependencies([merge_files]):
                  with ops.control_dependencies([notify_global_done]):
                    return array_ops.identity(checkpoint_prefix)
            with ops.control_dependencies(notify_local_done):
              with ops.control_dependencies([ready_ranks]):
                return array_ops.identity(checkpoint_prefix)

    def _AddShardedRestoreOps(
        self, filename_tensor, per_device, restore_sequentially, reshape):
      r'''Add Ops to restore variables from multiple devices.

      Args:
        filename_tensor: Tensor for the path of the file to load.
        per_device: A list of (device, SaveableObject) pairs, as returned by
          _GroupByDevices().
        restore_sequentially: True if we want to restore variables sequentially
          within a shard.
        reshape: True if we want to reshape loaded tensors to the shape of the
          corresponding variable.

      Returns:
        An Operation that restores the variables.
      '''
      model_dir = Context.get().options.model_dir
      if model_dir is not None:
        latest_path = checkpoint_management.latest_checkpoint(model_dir)  # pylint: disable=protected-access
        if latest_path:
          self._restoreable_saveables, _ = zip(
            *checkpoint_utils.list_variables(model_dir))
      return super()._AddShardedRestoreOps(
        filename_tensor, per_device, restore_sequentially, reshape)

    def restore_op(self, filename_tensor, saveable, preferred_shard):
      r'''Create ops to restore 'saveable'.
      '''
      if (self._restoreable_saveables is not None
          and saveable.name not in self._restoreable_saveables
          and hasattr(saveable, 'initializer')):
        return saveable.initializer.outputs
      return super().restore_op(filename_tensor, saveable, preferred_shard)

  return HybridBackendSaverBuilder


SaverBuilder = wraps_saver_builder(saver.BulkSaverBuilder)


class HybridBackendSaverBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of sharded savers.
  '''


def wraps_saver(cls):
  r'''Wraps a saver to support hybrid parallelism.
  '''
  if issubclass(cls, HybridBackendSaverBase):
    return cls

  class HybridBackendSaver(cls, HybridBackendSaverBase):
    r'''SaverBuilder with support for hybrid parallelism.
    '''
    def __init__(self, *args, **kwargs):
      self._rank = HvdContext.get().rank
      self._world_size = HvdContext.get().world_size
      kwargs['sharded'] = True
      kwargs['allow_empty'] = True
      with ops.device('/cpu:0'):
        super().__init__(*args, **kwargs)

    @property
    def rank(self):
      return self._rank

    @property
    def world_size(self):
      return self._world_size

    def _build(self, *args, **kwargs):
      r'''Builds saver_def.
      '''
      if self._world_size <= 1:
        super()._build(*args, **kwargs)
        return

      if self._builder is None:
        orig_saver_builder = saver.BulkSaverBuilder
        saver.BulkSaverBuilder = wraps_saver_builder(saver.BulkSaverBuilder)
        super()._build(*args, **kwargs)
        saver.BulkSaverBuilder = orig_saver_builder
      else:
        if not isinstance(self._builder, HybridBackendSaverBuilderBase):
          raise ValueError(
            '`SaverBuilder` must decorated by `wraps_saver_builder`')
        super()._build(*args, **kwargs)

    def save(self, *args, **kwargs):
      r'''Saves sharded variables.
      '''
      if self._world_size <= 1:
        super().save(*args, **kwargs)
        return

      write_meta_graph = (
        kwargs.pop('write_meta_graph', True)
        and self._rank == 0)
      kwargs['write_meta_graph'] = write_meta_graph
      write_state = kwargs.pop('write_state', True) and self._rank == 0
      kwargs['write_state'] = write_state
      super().save(*args, **kwargs)

    def export_meta_graph(self, filename=None, **kwargs):
      if self._rank == 0:
        return super().export_meta_graph(filename=filename, **kwargs)
      return None

  return HybridBackendSaver


Saver = wraps_saver(saver.Saver)


def replace_default_saver():
  r'''Try to replace default saver to HybridBackendSaver.
  '''
  rank = HvdContext.get().rank
  savers = ops.get_collection_ref(ops.GraphKeys.SAVERS)

  if not savers:
    default_saver = Saver()
    ops.add_to_collection(ops.GraphKeys.SAVERS, default_saver)
    return
  if len(savers) > 1:
    raise ValueError(f'Multiple items found in collection SAVERS: {savers}')

  default_saver = savers[0]
  if isinstance(default_saver, HybridBackendSaverBase):
    return

  if not default_saver._sharded:  # pylint: disable=protected-access
    raise ValueError('Default saver must be sharded')
  if default_saver._builder is not None:  # pylint: disable=protected-access
    if not isinstance(default_saver._builder, HybridBackendSaverBuilderBase):  # pylint: disable=protected-access
      raise ValueError(
        'builder for default saver must decorated by `wraps_saver_builder`')
  else:
    def _wraps_build(build_fn):
      r'''Decorator to wrap saver build.
      '''
      def wrapped_build(self, *args, **kwargs):
        r'''Builds saver_def.
        '''
        orig_saver_builder = saver.BulkSaverBuilder
        saver.BulkSaverBuilder = wraps_saver_builder(saver.BulkSaverBuilder)
        build_fn(self, *args, **kwargs)
        saver.BulkSaverBuilder = orig_saver_builder
      return wrapped_build
    default_saver._build = _wraps_build(default_saver._build)  # pylint: disable=protected-access

  def _wraps_save(save_fn):
    def wrapped_save(self, *args, **kwargs):
      r'''Saves sharded variables.
      '''
      write_meta_graph = kwargs.pop('write_meta_graph', True) and rank == 0
      kwargs['write_meta_graph'] = write_meta_graph
      write_state = kwargs.pop('write_state', True) and rank == 0
      kwargs['write_state'] = write_state
      save_fn(self, *args, **kwargs)
    return wrapped_save
  default_saver.save = _wraps_save(default_saver.save)

class DefaultSaverRewriting(SessionRunRewriting):
  r'''A SessionRunHook replaces default saver.
  '''
  def begin(self):
    r''' initialize replica variables and enable synchronous dataset wrapper
    '''
    replace_default_saver()


SessionRunRewriting.register(DefaultSaverRewriting)

##################### Estimator ##########################

class RunConfig(run_config_lib.RunConfig):
  r'''RunConfig for estimators.
  '''
  @classmethod
  def build(cls, prototype=None, **kwargs):
    r'''Creates RunConfig from prototype.
    '''
    if prototype is None:
      return cls(**kwargs)
    prototype = prototype.replace(device_fn=device_function)
    prototype._is_chief = True  # pylint: disable=protected-access
    prototype._session_config = wraps_session_config(prototype.session_config)  # pylint: disable=protected-access
    if prototype._evaluation_master == '':  # pylint: disable=protected-access
      prototype._evaluation_master = prototype.master  # pylint: disable=protected-access

    return prototype

  def __init__(self, **kwargs):
    r'''Creates a wrapped RunConfig.
    '''
    kwargs['session_config'] = wraps_session_config(
      kwargs.pop('session_config', None))
    kwargs['device_fn'] = device_function
    super().__init__(**kwargs)
    self._is_chief = True  # pylint: disable=protected-access
    if self.evaluation_master == '':  # pylint: disable=protected-access
      self._evaluation_master = self.master  # pylint: disable=protected-access

class EstimatorRewriting(GraphRewriting):
  r'''Rewriting estimator
  '''
  def __init__(self):
    super().__init__()
    self._prev_estimator = None
  
  def begin(self):
    self._prev_estimator = _estimator_lib.Estimator

    class HorovodEstimator(cls):
      def __init__(self, model_fn):
        pass
      
      def train(self):
        pass
      
      def evaluate(self):
        pass
    
    _estimator_lib.Estimator = HorovodEstimator
  
  def end(self):
    _estimator_lib.Estimator = self._prev_estimator

GraphRewriting.register(EstimatorRewriting)

from tensorflow.python.training import monitored_session
from tensorflow.python.util import compat

from hybridbackend.tensorflow.framework.context import Context
from hybridbackend.tensorflow.framework.ops import ModeKeys

##################### public interface ##########################

@contextlib.contextmanager
def scope(**kwargs):
    with GraphRewriting.scope(**kwargs) as ctx:
        yield ctx


@contextlib.contextmanager
def embedding_scope(**kwargs):
    with GraphRewriting.scope(sharded=True, **kwargs) as ctx:
        yield ctx

def export(export_dir_base,
          checkpoint_path,
          signature_def_fn,
          assets_extra=None,
          as_text=False,
          clear_devices=True,
          strip_default_attrs=True,
          mode=ModeKeys.PREDICT):

  def export_all(
    export_dir_base,
    checkpoint_path,
    signature_defs_and_main_op_fn,
    assets_extra=None,
    as_text=False,
    clear_devices=True,
    strip_default_attrs=True,
    modes=None,
    **kwargs):
    r'''Build a SavedModel from variables in checkpoint.

    Args:
      export_dir_base: A string containing a directory to write the exported
          graph and checkpoints.
      checkpoint_path: A path to a checkpoint.
      signature_defs_and_main_op_fn: Function returns signature defs and main_op.
      assets_extra: A dict specifying how to populate the assets.extra directory
          within the exported SavedModel.  Each key should give the destination
          path (including the filename) relative to the assets.extra directory.
          The corresponding value gives the full path of the source file to be
          copied.  For example, the simple case of copying a single file without
          renaming it is specified as
          `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: Whether or not to write the SavedModel proto in text format.
      clear_devices: Whether or not to clear the device field.
      strip_default_attrs: Whether or not to remove default-valued attributes
          from the NodeDefs.
      modes: List contains PREDICT, TRAIN or TEST.

    Returns:
      Export directory if it's chief.
    '''
    if hvd.rank() != 0:
      return None

    export_dir = get_timestamped_export_dir(export_dir_base)
    with ops.Graph().as_default():
      with Context.scope(mode=ModeKeys.PREDICT, comm_pool_name=ModeKeys.PREDICT):
        # Build graph.
        signature_def_map = signature_defs_and_main_op_fn()
        main_op = None
        if isinstance(signature_def_map, (tuple, list)):
          if len(signature_def_map) > 1:
            main_op = signature_def_map[1]
          signature_def_map = signature_def_map[0]
        if not main_op:
          main_op = monitored_session.Scaffold.default_local_init_op()
        if modes is None:
          modes = [ModeKeys.PREDICT, ModeKeys.TRAIN, ModeKeys.EVAL]
        modes = [
          m for m in modes
          if SIGNATURE_KEY_MAP[m] in signature_def_map]
        signature_def_map = {
          k: signature_def_map[k] for k in signature_def_map
          if k in [SIGNATURE_KEY_MAP[m] for m in modes]}
        signature_tags = [EXPORT_TAG_MAP[m][0] for m in modes]

        b = builder.SavedModelBuilder(export_dir, **kwargs)
        b._has_saved_variables = True  # pylint: disable=protected-access

        # Copy variables.
        saved_model_utils.get_or_create_variables_dir(export_dir)
        export_checkpoint_path = saved_model_utils.get_variables_path(export_dir)
        checkpoint_files = [
          *gfile.Glob(f'{checkpoint_path}.index'),
          *gfile.Glob(f'{checkpoint_path}.data-?????-of-?????')]
        for f in checkpoint_files:
          export_ckpt = re.sub(
            compat.as_text(checkpoint_path),
            compat.as_text(export_checkpoint_path),
            f)
          gfile.Copy(f, export_ckpt)

        # Add MetaGraph.
        b.add_meta_graph(
          tags=signature_tags,
          signature_def_map=signature_def_map,
          assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
          clear_devices=clear_devices,
          main_op=main_op,
          strip_default_attrs=strip_default_attrs)

        # Save model.
        b.save(as_text=as_text)

        # Save extras.
        if assets_extra:
          assets_extra_path = os.path.join(
            export_dir, constants.EXTRA_ASSETS_DIRECTORY)
          for dst, src in assets_extra.items():
            target = os.path.join(assets_extra_path, compat.as_bytes(dst))
            gfile.MakeDirs(os.path.dirname(target))
            gfile.Copy(src, target)

  return export_all(export_dir_base,
                    checkpoint_path,
                    lambda: {SIGNATURE_KEY_MAP[mode]: signature_def_fn()},
                    assets_extra=assets_extra,
                    as_text=as_text,
                    clear_devices=clear_devices,
                    strip_default_attrs=strip_default_attrs,
                    modes=[mode]) 