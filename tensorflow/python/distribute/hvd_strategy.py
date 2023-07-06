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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid
import re
import abc
import os
import contextlib
import threading
import random as rn
import json
import collections
import time
import six
import numpy as np
import horovod.tensorflow as hvd

from tensorflow.python.distribute import estimator_training
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context as _context
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import gfile
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import monitored_session as _monitored_session
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.training import training
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.model_utils.export_utils import \
  EXPORT_TAG_MAP
from tensorflow.python.saved_model.model_utils.export_utils import \
  get_timestamped_export_dir
from tensorflow.python.saved_model.model_utils.export_utils import \
  SIGNATURE_KEY_MAP

from tensorflow._api.v1 import train as train_v1

from tensorflow_estimator.python.estimator import estimator as _estimator_lib
from tensorflow_estimator.python.estimator import \
    run_config as run_config_lib
from tensorflow_estimator.python.estimator.training import _is_google_env
from tensorflow_estimator.python.estimator.training import _TrainingExecutor
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow_estimator.python.estimator.training import _MAX_DELAY_SECS
from tensorflow_estimator.python.estimator.export import export_lib

try:
  from tensorflow.python.training.saving.saveable_object_util import \
    op_list_to_dict
except ImportError:
  op_list_to_dict = saver.BaseSaverBuilder.OpListToDict

from hybridbackend.tensorflow.framework.ops import GraphKeys

##################### HVDSTRATEGY COMMON CODE ##########################

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
      # prev_kwargs = c.options.update(**kwargs)
      yield c
    finally:
      # c.options.update(**prev_kwargs)
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

    with HvdContext.scope() as ctx:
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
      with HvdContext.scope(mode=ModeKeys.PREDICT):
        # Build graph.
        signature_def_map = signature_defs_and_main_op_fn()
        main_op = None
        if isinstance(signature_def_map, (tuple, list)):
          if len(signature_def_map) > 1:
            main_op = signature_def_map[1]
          signature_def_map = signature_def_map[0]
        if not main_op:
          main_op = _monitored_session.Scaffold.default_local_init_op()
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

class CollectiveServerBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of server wrapper.
  '''

def wraps_server(cls):
  if issubclass(cls, CollectiveServerBase):
    return cls

  class CollectiveServer(cls, CollectiveServerBase):
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

  return CollectiveServer

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

class CollectiveSaverBuilderBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of sharded saver builders.
  '''

def wraps_saver_builder(cls):
  r'''Wraps a saver builder to support hybrid parallelism.
  '''
  if issubclass(cls, CollectiveSaverBuilderBase):
    return cls

  class CollectiveSaverBuilder(cls, CollectiveSaverBuilderBase):
    r'''Wrapped SaverBuilder with support for hybrid parallelism.
    '''
    def __init__(self, *args, **kwargs):
      name = kwargs.pop('name', 'collective_saver_builder')
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
          ops.EMBEDDING_VARIABLES)
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
      model_dir = HvdContext.get().options.model_dir
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

  return CollectiveSaverBuilder


SaverBuilder = wraps_saver_builder(saver.BulkSaverBuilder)


class CollectiveSaverBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of sharded savers.
  '''

def wraps_saver(cls):
  r'''Wraps a saver to support hybrid parallelism.
  '''
  if issubclass(cls, CollectiveSaverBase):
    return cls

  class CollectiveSaver(cls, CollectiveSaverBase):
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
        if not isinstance(self._builder, CollectiveSaverBuilderBase):
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

  return CollectiveSaver


Saver = wraps_saver(saver.Saver)


def replace_default_saver():
  rank = HvdContext.get().rank
  savers = ops.get_collection_ref(ops.GraphKeys.SAVERS)

  if not savers:
    default_saver = Saver()
    ops.add_to_collection(ops.GraphKeys.SAVERS, default_saver)
    return
  if len(savers) > 1:
    raise ValueError(f'Multiple items found in collection SAVERS: {savers}')

  default_saver = savers[0]
  if isinstance(default_saver, CollectiveSaverBase):
    return

  if not default_saver._sharded:  # pylint: disable=protected-access
    raise ValueError('Default saver must be sharded')
  if default_saver._builder is not None:  # pylint: disable=protected-access
    if not isinstance(default_saver._builder, CollectiveSaverBuilderBase):  # pylint: disable=protected-access
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

class DefaultSaverRewriting(GraphRewriting):
  r'''A SessionRunHook replaces default saver.
  '''
  def begin(self):
    r''' initialize replica variables and enable synchronous dataset wrapper
    '''
    replace_default_saver()

GraphRewriting.register(DefaultSaverRewriting)

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

def wraps_model_fn(model_fn, model_dir, config):
  r'''Decorator to set params in a model function.
  '''
  def wrapped_model_fn(features, labels, mode, params):
    r'''Wrapped model function.
    '''
    with scope(mode=mode, model_dir=model_dir):
      estimator_spec = model_fn(features, labels, mode, params)
      if estimator_spec.scaffold.saver:
        if not isinstance(
            estimator_spec.scaffold.saver._builder,  # pylint: disable=protected-access
            HorovodSaverBuilderBase):
          raise ValueError(
            'scaffold.saver in EstimatorSpec must be hb.train.Saver, '
            'you can try call hb.train.replace_default_saver() before '
            'creation of the scaffold.')
      else:
        estimator_spec.scaffold._saver = Saver(  # pylint: disable=protected-access
          max_to_keep=config.keep_checkpoint_max,
          keep_checkpoint_every_n_hours=config.keep_checkpoint_every_n_hours,
          defer_build=True,
          save_relative_paths=True)
      training_hooks = list(estimator_spec.training_hooks) or []
      training_chief_hooks = list(estimator_spec.training_chief_hooks) or []
      estimator_spec = estimator_spec._replace(  # pylint: disable=protected-access
        training_hooks=training_hooks,
        training_chief_hooks=training_chief_hooks)
      return estimator_spec
  return wrapped_model_fn

def start_std_server(config):
  r'''Creates, starts, and returns a server_lib.Server.
  '''
  logging.info('Start Tensorflow server.')
  return Server(config.cluster_spec,
                job_name=config.task_type,
                task_index=config.task_id,
                config=wraps_session_config(config.session_config),
                start=True,
                protocol=config.protocol)

EvaluationSpec = collections.namedtuple(
  'EvaluationSpec', ['name', 'hooks', 'update_op', 'eval_dict'])

class TrainingExecutor(_TrainingExecutor):
  r'''The executor to run `Estimator` training and evaluation.
  '''
  def _start_std_server(self, config):
    r'''Creates, starts, and returns a server_lib.Server.'''
    start_std_server(config)

class HorovodEstimatorBase(object):  # pylint: disable=useless-object-inheritance
  r'''Base class of estimator wrapper.
  '''

def wraps_estimator(cls):
  r'''Estimator decorator to train and evaluate in parallel.
  '''
  if issubclass(cls, HorovodEstimatorBase):
    return cls

  class HorovodEstimator(cls, HorovodEstimatorBase):
    r'''Class to train and evaluate TensorFlow models.
    '''
    def __init__(self, model_fn, **kwargs):
      r'''Constructs a wrapped `Estimator` instance.

      Args:
        model_fn: Model function. See
          `tensorflow_estimator/python/estimator/estimator.py#L145`
          for more information.
        kwargs: Estimator arguments.
      '''
      kwargs['config'] = RunConfig.build(prototype=kwargs.pop('config', None))
      model_dir = kwargs.get('model_dir', None)
      self._train_drop_remainder = kwargs.pop('train_drop_remainder', True)
      self._eval_drop_remainder = kwargs.pop('eval_drop_remainder', True)
      self._predict_drop_remainder = kwargs.pop('predict_drop_remainder', True)

      super().__init__(
        wraps_model_fn(model_fn, model_dir, kwargs['config']),
        **kwargs)

    def _assert_members_are_not_overridden(self):
      r'''disable the overridden check here.
      '''

    def train(
        self, input_fn, hooks=None, max_steps=None, saving_listeners=None):
      r'''support sync_dataset in training.
      '''
      if saving_listeners is None:
        saving_listeners = []
      with scope(
          mode=ModeKeys.TRAIN,
          model_dir=self._model_dir,
          data_sync_drop_remainder=self._train_drop_remainder):
        return super().train(
          input_fn, hooks=hooks, max_steps=max_steps,
          saving_listeners=saving_listeners)

    def evaluate(self,
                 input_fn,
                 steps=None,
                 hooks=None,
                 checkpoint_path=None,
                 name=None):
      r'''support standalone evaluation.
      '''
      _estimator_lib._estimator_api_gauge.get_cell('evaluate').set(True)  # pylint: disable=protected-access
      if self.config.cluster_spec:
        if estimator_training.should_run_distribute_coordinator(self.config):
          raise ValueError(
            'Running `evaluate` with Distribute Coordinator '
            'not supported.')
        if not _is_google_env():
          start_std_server(self.config)

        start_delay_secs = 0
        if self.config.task_type == run_config_lib.TaskType.WORKER:
          max_delay_secs = _MAX_DELAY_SECS
          if self.config.experimental_max_worker_delay_secs is not None:
            max_delay_secs = int(self.config.experimental_max_worker_delay_secs)
          start_delay_secs = min(
            max_delay_secs,
            (self.config.task_id + 1) * _DELAY_SECS_PER_WORKER)

        if start_delay_secs > 0:
          logging.info(
            f'Waiting {start_delay_secs} secs before starting evaluation.')
          time.sleep(start_delay_secs)

      return self._actual_eval(
        input_fn,
        strategy=self._eval_distribution,
        steps=steps,
        hooks=hooks,
        checkpoint_path=checkpoint_path,
        name=name)

    def _actual_eval(
        self, input_fn, strategy=None, steps=None, hooks=None,
        checkpoint_path=None, name=None):
      if strategy:
        raise ValueError('DistributionStrategy not supported')

      with _context.graph_mode(), HvdContext.scope(
          mode=ModeKeys.EVAL,
          model_dir=self._model_dir):
        hooks = _estimator_lib._check_hooks_type(hooks)  # pylint: disable=protected-access
        hooks.extend(self._convert_eval_steps_to_hooks(steps))  # pylint: disable=protected-access
        if not checkpoint_path:
          latest_path = checkpoint_management.latest_checkpoint(self._model_dir)  # pylint: disable=protected-access
          if not latest_path:
            raise ValueError(
              f'Could not find trained model in model_dir: {self._model_dir}.')  # pylint: disable=protected-access
          checkpoint_path = latest_path

        with ops.Graph().as_default() as g, g.device(self._device_fn):  # pylint: disable=protected-access
          with ops.name_scope(ModeKeys.EVAL), reuse_variables(vs.AUTO_REUSE):
            (scaffold, update_op, eval_dict, all_hooks) = (
              self._evaluate_build_graph(  # pylint: disable=protected-access
                input_fn,
                hooks, checkpoint_path))
            return self._evaluate_run(  # pylint: disable=protected-access
              checkpoint_path=checkpoint_path,
              scaffold=scaffold,
              update_op=update_op,
              eval_dict=eval_dict,
              all_hooks=all_hooks,
              output_dir=self.eval_dir(name))

    def train_and_evaluate(
        self, train_spec, eval_spec,
        eval_every_n_iter=None,
        eval_history=None):
      r'''Train and evaluate the `estimator`.

      Args:
        eval_every_n_iter: `int`, runs parallel evaluation once every
          N training iteration. If None, disable the evaluation.
        eval_history: History of eval metrics. eval_history should support
          `append` method.
      '''
      train_hooks = []
      if eval_every_n_iter is not None:
        def _eval_fn():
          with scope(
              model_dir=self._model_dir,
              data_sync_drop_remainder=self._eval_drop_remainder):
            (_, evaluation_hooks, input_hooks, update_op, eval_dict) = (
              self._call_model_fn_eval(  # pylint: disable=protected-access
                eval_spec.input_fn, self.config))
            hooks = list(evaluation_hooks) or []
            hooks.extend(list(input_hooks) or [])
            return EvaluationSpec(
              name=EvaluationSpec.__name__,
              hooks=hooks,
              update_op=update_op,
              eval_dict=eval_dict)
        eval_hook = EvaluationHook(
          _eval_fn,
          steps=eval_spec.steps,
          every_n_iter=eval_every_n_iter,
          summary_dir=self.eval_dir(),
          history=eval_history)
        train_hooks.append(eval_hook)

      if self.config.cluster_spec:
        if estimator_training.should_run_distribute_coordinator(self.config):
          raise ValueError(
            'Running `train_and_evaluate` with Distribute Coordinator '
            'not supported.')

        executor = TrainingExecutor(
          estimator=self,
          train_spec=train_spec,
          eval_spec=eval_spec,
          train_hooks=train_hooks)
        return executor.run()

      return self.train(
        train_spec.input_fn,
        hooks=tuple(train_spec.hooks) + tuple(train_hooks),
        max_steps=train_spec.max_steps)

    def export_saved_model(
        self, export_dir_base, serving_input_receiver_fn,
        assets_extra=None,
        as_text=False,
        checkpoint_path=None,
        experimental_mode=ModeKeys.PREDICT,
        **kwargs):
      r'''Exports inference graph as a `SavedModel` into the given dir.
      '''
      if not serving_input_receiver_fn:
        raise ValueError('An input_receiver_fn must be defined.')

      input_receiver_fn_map = {experimental_mode: serving_input_receiver_fn}

      return self._export_all_saved_models(
        export_dir_base,
        input_receiver_fn_map,
        assets_extra=assets_extra,
        as_text=as_text,
        checkpoint_path=checkpoint_path,
        strip_default_attrs=True,
        **kwargs)

    def experimental_export_all_saved_models(
        self, export_dir_base, input_receiver_fn_map,
        assets_extra=None,
        as_text=False,
        checkpoint_path=None,
        **kwargs):
      r'''Exports a `SavedModel` with `tf.MetaGraphDefs` for each requested
        mode.
      '''
      return self._export_all_saved_models(
        export_dir_base, input_receiver_fn_map,
        assets_extra=assets_extra,
        as_text=as_text,
        checkpoint_path=checkpoint_path,
        strip_default_attrs=True,
        **kwargs)

    def _export_all_saved_models(
        self,
        export_dir_base,
        input_receiver_fn_map,
        assets_extra=None,
        as_text=False,
        checkpoint_path=None,
        strip_default_attrs=True,
        **kwargs):
      r'''Exports multiple modes in the model function to a SavedModel.
      '''
      if (input_receiver_fn_map.get(ModeKeys.TRAIN)
          or input_receiver_fn_map.get(ModeKeys.EVAL)
          or not input_receiver_fn_map.get(ModeKeys.PREDICT)):
        raise ValueError('Only PREDICT mode is supported.')
      mode = ModeKeys.PREDICT

      if HvdContext.get().rank != 0:
        return None

      if not checkpoint_path:
        checkpoint_path = checkpoint_management.latest_checkpoint(
          self._model_dir)
      if not checkpoint_path:
        if self._warm_start_settings:
          checkpoint_path = self._warm_start_settings.ckpt_to_initialize_from
          if gfile.IsDirectory(checkpoint_path):
            checkpoint_path = checkpoint_management.latest_checkpoint(
              checkpoint_path)
        else:
          raise ValueError(
            f'Couldn\'t find trained model at {self._model_dir}.')

      def _fn():
        random_seed.set_random_seed(self._config.tf_random_seed)

        input_receiver_fn = input_receiver_fn_map[mode]
        input_receiver = input_receiver_fn()
        estimator_spec = self._call_model_fn(
          features=input_receiver.features,
          labels=getattr(input_receiver, 'labels', None),
          mode=mode,
          config=self.config)
        export_outputs = export_lib.export_outputs_for_mode(
          mode=estimator_spec.mode,
          serving_export_outputs=estimator_spec.export_outputs,
          predictions=estimator_spec.predictions,
          loss=estimator_spec.loss,
          metrics=estimator_spec.eval_metric_ops)
        signature_def_map = export_lib.build_all_signature_defs(
          input_receiver.receiver_tensors,
          export_outputs,
          getattr(input_receiver, 'receiver_tensors_alternatives', None),
          serving_only=(mode == ModeKeys.PREDICT))
        main_op = None
        if estimator_spec.scaffold.local_init_op is not None:
          main_op = estimator_spec.scaffold.local_init_op
        return signature_def_map, main_op

      return export_all(
        export_dir_base,
        checkpoint_path,
        _fn,
        assets_extra=assets_extra,
        as_text=as_text,
        clear_devices=True,
        strip_default_attrs=strip_default_attrs,
        modes=[mode],
        **kwargs)

    def _actual_predict(
        self, input_fn,
        predict_keys=None,
        hooks=None,
        checkpoint_path=None,
        yield_single_examples=True):
      r'''Predict method of estimator in HB.
      '''
      with _context.graph_mode(), HvdContext.scope(
          mode=ModeKeys.PREDICT,
          model_dir=self._model_dir,
          comm_pool_capacity=1,
          comm_pool_name=ModeKeys.PREDICT):
        hooks = _estimator_lib._check_hooks_type(hooks)  # pylint: disable=protected-access
        # Check that model has been trained.
        if not checkpoint_path:
          checkpoint_path = checkpoint_management.latest_checkpoint(
            self._model_dir)
        if not checkpoint_path:
          logging.info(
            f'Could not find trained model in model_dir: {self._model_dir},'
            f'running initialization to predict.')
        with ops.Graph().as_default() as g, g.device(self._device_fn):
          with ops.name_scope(ModeKeys.PREDICT):
            random_seed.set_random_seed(self._config.tf_random_seed)
            self._create_and_assert_global_step(g)
            features, input_hooks = self._get_features_from_input_fn(
              input_fn, ModeKeys.PREDICT)
            estimator_spec = self._call_model_fn(
              features, None, ModeKeys.PREDICT, self.config)

          # Call to warm_start has to be after model_fn is called.
          self._maybe_warm_start(checkpoint_path)

          predictions = self._extract_keys(estimator_spec.predictions,
                                           predict_keys)
          all_hooks = list(input_hooks)
          all_hooks.extend(hooks)
          all_hooks.extend(list(estimator_spec.prediction_hooks or []))
          with _monitored_session.MonitoredSession(
              session_creator=_monitored_session.ChiefSessionCreator(
                checkpoint_filename_with_path=checkpoint_path,
                master=self._config.master,
                scaffold=estimator_spec.scaffold,
                config=self._session_config),
              hooks=all_hooks) as mon_sess:
            while not mon_sess.should_stop():
              preds_evaluated = mon_sess.run(predictions)
              if not yield_single_examples:
                yield preds_evaluated
              elif not isinstance(predictions, dict):
                for pred in preds_evaluated:
                  yield pred
              else:
                for i in range(self._extract_batch_length(preds_evaluated)):
                  yield {
                    key: value[i]
                    for key, value in six.iteritems(preds_evaluated)
                  }

    def predict(
        self, input_fn,
        predict_keys=None,
        hooks=None,
        checkpoint_path=None,
        yield_single_examples=True):
      r'''Predict method of estimator in HB.
      '''
      _estimator_lib._estimator_api_gauge.get_cell('predict').set(True)  # pylint: disable=protected-access
      if self.config.cluster_spec:
        if estimator_training.should_run_distribute_coordinator(self.config):
          raise ValueError(
            'Running `evaluate` with Distribute Coordinator '
            'not supported.')
        if not _is_google_env():
          start_std_server(self.config)

      return self._actual_predict(
        input_fn,
        predict_keys=predict_keys,
        hooks=hooks,
        checkpoint_path=checkpoint_path,
        yield_single_examples=yield_single_examples)

  return HorovodEstimator


Estimator = wraps_estimator(_estimator_lib.Estimator)

class EstimatorRewriting(GraphRewriting):
  r'''Rewriting estimator
  '''
  def __init__(self):
    super().__init__()
    self._prev_estimator = None
  
  def begin(self):
    self._prev_estimator = _estimator_lib.Estimator

    _estimator_lib.Estimator = Estimator
  
  def end(self):
    _estimator_lib.Estimator = self._prev_estimator

GraphRewriting.register(EstimatorRewriting)