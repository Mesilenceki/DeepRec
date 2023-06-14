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
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.framework import ops

from tensorflow_estimator.python.estimator import estimator as _estimator_lib
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
    cluster_spec = Context.get().cluster_spec
    task_type = Context.get().task_type
    task_id = Context.get().task_id
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
    hooks.extend(hvd.BroadcastGlocalVariablesHook(0))

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