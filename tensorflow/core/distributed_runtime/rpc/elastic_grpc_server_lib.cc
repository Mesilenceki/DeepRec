/* Copyright 2023 The DeepRec Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#include "tensorflow/core/distributed_runtime/rpc/elastic_grpc_server_lib.h"
#include "tensorflow/core/distributed_runtime/rpc/elastic_service.h"

#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "include/json/json.h"
#include "grpc/support/alloc.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"
#include "tensorflow/core/util/env_var.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/local_master.h"
#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_cache_wrapper.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/worker_resource.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/protobuf/cluster.pb.h"

namespace tensorflow {

namespace {

// Define an option subclass in order to disable SO_REUSEPORT for the
// server socket.
class NoReusePortOption : public ::grpc::ServerBuilderOption {
 public:
  void UpdateArguments(::grpc::ChannelArguments* args) override {
    args->SetInt(GRPC_ARG_ALLOW_REUSEPORT, 1);
  }

  void UpdatePlugins(std::vector<std::unique_ptr<::grpc::ServerBuilderPlugin>>*
                         plugins) override {}
};

// static utility function
RendezvousMgrInterface* NewRpcRendezvousMgr(const WorkerEnv* env) {
  return new RpcRendezvousMgr(env);
}

}  // namespace

ElasticGrpcServer::ElasticGrpcServer(const ServerDef& server_def, Env* env)
    : server_def_(server_def), env_(env), state_(NEW) {}

ElasticGrpcServer::~ElasticGrpcServer() {
  TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());

  delete master_service_;
  delete worker_service_;
  delete eager_service_;
  delete elastic_service_;

  // TODO(mrry): Refactor the *Env classes so that it is less fiddly
  // to destroy them.

  // Shut down all outstanding rendezvous.
  delete worker_env_.rendezvous_mgr;

  // We must delete graph_mgr before device_mgr, due to shared
  // ownership of OpKernels in the executors. (The graph_mgr will
  // free all stateless OpKernels, and pass over borrowed stateful
  // OpKernels, which are also held in their respective devices'
  // OpSegments.)
  if (worker_env_.session_mgr != nullptr) {
    delete worker_env_.session_mgr;  // Deletes graph_mgr's.
  } else {
    // Note: session_mgr's legacy_session_ deletes device_mgr now.
    delete worker_env_.device_mgr;
  }

  // Do not delete (as these are not owned by the server):
  // - master_env_.env
  // - worker_env_.env
  // - worker_env_.compute_pool
}

void ElasticGrpcServer::MaybeMutateBuilder(::grpc::ServerBuilder* builder) {}

// Look up the port that has been requested for this task in `server_def_`.
Status ElasticGrpcServer::GetPort(int* port) const {
  *port = -1;
  for (const auto& job : server_def_.cluster().job()) {
    if (job.name() == server_def_.job_name()) {
      auto iter = job.tasks().find(server_def_.task_index());
      if (iter == job.tasks().end()) {
        return errors::InvalidArgument("Task ", server_def_.task_index(),
                                       " was not defined in job \"",
                                       server_def_.job_name(), "\"");
      }
      auto colon_index = iter->second.find_last_of(':');
      if (!strings::safe_strto32(iter->second.substr(colon_index + 1), port)) {
        return errors::InvalidArgument(
            "Could not parse port for local server from \"", iter->second,
            "\".");
      }
      break;
    }
  }
  if (*port == -1) {
    return errors::Internal("Job \"", server_def_.job_name(),
                            "\" was not defined in cluster");
  }

  return Status::OK();
}

Status ElasticGrpcServer::UpdateServerDef(const string& cluster_def_str, int& before_part_num, int& after_part_num) {
  std::string tf_config;
  ReadStringFromEnvVar("TF_CONFIG", "", &tf_config);
  if (!tf_config.empty()) {
    Json::Reader reader;
    Json::Value tf_config_json;
    if(!reader.parse(tf_config, tf_config_json)) {
      return errors::Internal("PARSE TF_CONFIG ERROR");
    }
    if ((tf_config_json["cluster"].isNull()) ||
        (tf_config_json["cluster"]["ps"].isNull())) {
      return errors::Internal("PARSE PS FROM TF_CONFIG ERROR");
    }

    Json::Value cluster_json;
    if (!reader.parse(cluster_def_str, cluster_json)) {
      LOG(ERROR) << "cluster_def is not correct with " << cluster_def_str;
      return errors::Internal("PARSE TF_CONFIG/cluster ERROR");
    }

    std::unordered_set<string> ps_addrs_vec;
    after_part_num = cluster_json["cluster"]["ps"].size();
    for (auto& value: cluster_json["cluster"]["ps"]) {
      ps_addrs_vec.emplace(value.asString());
    }

    int job_size = server_def_.cluster().job_size();
    for (int j = 0; j < job_size; ++j) {
      auto* job = server_def_.mutable_cluster()->mutable_job(j);
      if (job->name() == "ps") {
        before_part_num = job->tasks_size();
        if (before_part_num == after_part_num) {
          return Status::OK();
        } else if (after_part_num > before_part_num) {
          int idx = before_part_num;
          LOG(INFO) << "JUNQI Scaling up ===============> " << after_part_num;
          std::unordered_set<string> target_string_set;
          for (auto& value: tf_config_json["cluster"]["ps"]) {
            target_string_set.emplace(value.asString());
          }
          for (auto ps_addr: ps_addrs_vec) {
            if (target_string_set.find(ps_addr) == target_string_set.end()) {
              job->mutable_tasks()->insert({idx, ps_addr});
              tf_config_json["cluster"]["ps"].append(ps_addr);
            }
          } 
          break;
        } else {
          LOG(INFO) << "JUNQI Scaling down ===============> " << after_part_num;
          for (int i = 0; i < before_part_num; ++i) {
            string tmp_string = tf_config_json["cluster"]["ps"][i].asString();
            if (ps_addrs_vec.find(tmp_string) == ps_addrs_vec.end()) {
              Json::Value ps_addr;
              tf_config_json["cluster"]["ps"].removeIndex(i, &ps_addr);
              job->mutable_tasks()->erase(i);
            }
          }
        }
      }
    }
    Json::FastWriter writer;
    std::string new_tf_config = writer.write(tf_config_json);
    LOG(INFO) << "new TF_CONFIG " << new_tf_config;
    setenv("TF_CONFIG", new_tf_config.c_str(), 1);
  }
  return Status::OK();
}

Status ElasticGrpcServer::Update(const string& cluster_def_str) {
  for (auto* device: worker_env_.device_mgr->ListDevices()) {
    LOG(INFO) << device->resource_manager()->DebugString();
  }
  int before_part_num, after_part_num;
  Status s = UpdateServerDef(cluster_def_str, before_part_num, after_part_num);
  if (!s.ok()) {
    LOG(ERROR) << s.error_message();
    return Status::OK();
  }

  if (after_part_num == before_part_num) {
    return Status::OK();
  }

  WorkerCacheInterface* worker_cache;
  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  TF_RETURN_IF_ERROR(
      UpdateWorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  CHECK_NE(nullptr, worker_cache);
  LOG(INFO) << " =============== ";
  for (auto* device: worker_env_.device_mgr->ListDevices()) {
    LOG(INFO) << device->resource_manager()->DebugString();
  }
  ConfigProto config = server_def_.default_session_config();
  string unused;
  string default_worker_name;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                        &default_worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }
  std::unique_ptr<DeviceResolverDistributed> dev_resolver(
      new DeviceResolverDistributed(worker_env_.device_mgr, worker_cache,
                                    default_worker_name));
  std::unique_ptr<CollectiveParamResolverDistributed> param_resolver(
      new CollectiveParamResolverDistributed(config, worker_env_.device_mgr,
                                              dev_resolver.get(), worker_cache,
                                              default_worker_name));
  worker_env_.collective_executor_mgr = new RpcCollectiveExecutorMgr(
      config, worker_env_.device_mgr, std::move(dev_resolver),
      std::move(param_resolver), worker_cache, default_worker_name);

  if (worker_env_.session_mgr != nullptr) {
    delete worker_env_.session_mgr;  // Deletes graph_mgr's.
  }

  // Set up worker environment.
  worker_env_.session_mgr = new SessionMgr(
      &worker_env_, SessionMgr::WorkerNameFromServerDef(server_def_),
      std::unique_ptr<WorkerCacheInterface>(worker_cache),
      [this](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
        WorkerCacheFactoryOptions options(server_def);
        return WorkerCacheFactory(options, worker_cache);
      });
  master_env_.worker_cache = worker_cache;
  // Finish setting up master environment.
  
  StatsPublisherFactory stats_factory = opts_.stats_factory;
  master_env_.master_session_factory =
      [config, stats_factory](
          SessionOptions options, const MasterEnv* env,
          std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
          std::unique_ptr<WorkerCacheInterface> worker_cache,
          std::unique_ptr<DeviceSet> device_set,
          std::vector<string> filtered_worker_list) {
        options.config.MergeFrom(config);
        return new MasterSession(options, env, std::move(remote_devs),
                                 std::move(worker_cache), std::move(device_set),
                                 std::move(filtered_worker_list),
                                 stats_factory);
      };
  master_env_.worker_cache_factory =
      [this](const WorkerCacheFactoryOptions& options,
             WorkerCacheInterface** worker_cache) {
        return WorkerCacheFactory(options, worker_cache);
      };
  return Status::OK();
}

Status ElasticGrpcServer::Init(const GrpcServerOptions& opts) {
  opts_ = opts;
  mutex_lock l(mu_);
  CHECK_EQ(state_, NEW);
  master_env_.env = env_;
  worker_env_.env = env_;

  // Check parameters before DeviceFactory::AddDevices,
  // otherwise if 'task_index=-1' the program will abort.

  int requested_port;
  TF_RETURN_IF_ERROR(GetPort(&requested_port));

  SessionOptions sess_opts;
  ConfigProto config = server_def_.default_session_config();
  sess_opts.config = config;

  // Configure shared devices between master and worker.
  string name_prefix =
      strings::StrCat("/job:", server_def_.job_name(), "/replica:0",
                      "/task:", server_def_.task_index());
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(
      DeviceFactory::AddDevices(sess_opts, name_prefix, &devices));
  worker_env_.device_mgr = new DeviceMgr(std::move(devices));
  master_env_.local_devices = worker_env_.device_mgr->ListDevices();
  worker_env_.local_devices = worker_env_.device_mgr->ListDevices();
  worker_env_.rendezvous_mgr = opts.rendezvous_mgr_func == nullptr
                                   ? new RpcRendezvousMgr(&worker_env_)
                                   : opts.rendezvous_mgr_func(&worker_env_);
  string unused;
  string default_worker_name;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                        &default_worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }

  // N.B. The order of initialization here is intricate, because we
  // wish to allow `requested_port == 0` (for choosing any port,
  // mostly for testing). Therefore, the construction of the channel
  // and worker caches depends on `bound_port_`, which is not set
  // until we call `builder.BuildAndStart()`. We must create the
  // service objects before calling `builder.BuildAndStart()`, but
  // `master_env_` and `worker_env_` are only partially
  // configured. However, this is not dangerous, because we do not
  // start serving requests until `this->Start()` is called, which
  // happens after this method returns.
  //
  // TODO(mrry): Provide a general mechanism for dynamically setting
  // the identities of tasks in the worker pool after the service is
  // running.
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port),
                           GetServerCredentials(server_def_), &bound_port_);
  builder.SetMaxMessageSize(std::numeric_limits<int32>::max());

  builder.SetOption(
      std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption));
  // Allow subclasses to specify more args to pass to the gRPC server.
  MaybeMutateBuilder(&builder);
  master_impl_ = CreateMaster(&master_env_);
  master_service_ = NewGrpcMasterService(master_impl_.get(), config, &builder);
  worker_impl_ = opts.worker_func ? opts.worker_func(&worker_env_, config)
                                  : NewGrpcWorker(&worker_env_, config);
  worker_service_ = NewGrpcWorkerService(worker_impl_.get(), &builder,
                                         opts.worker_service_options)
                        .release();
  eager_service_ = new eager::GrpcEagerServiceImpl(&worker_env_, &builder);
  elastic_service_ = NewGrpcElasticService(this, config, &builder);

  // extra service:
  if (opts.service_func != nullptr) {
    opts.service_func(&worker_env_, &builder);
  }
  server_ = builder.BuildAndStart();

  if (!server_) {
    return errors::Unknown("Could not start gRPC server");
  }

  WorkerCacheInterface* worker_cache;
  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  TF_RETURN_IF_ERROR(
      WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  CHECK_NE(nullptr, worker_cache);

  if (opts.collective_mgr_func) {
    worker_env_.collective_executor_mgr =
        opts.collective_mgr_func(config, &worker_env_, worker_cache);
    if (!worker_env_.collective_executor_mgr) {
      return errors::Internal(
          "collective_mgr_func did not return CollectiveExecutorMgr");
    }
  } else {
    std::unique_ptr<DeviceResolverDistributed> dev_resolver(
        new DeviceResolverDistributed(worker_env_.device_mgr, worker_cache,
                                      default_worker_name));
    std::unique_ptr<CollectiveParamResolverDistributed> param_resolver(
        new CollectiveParamResolverDistributed(config, worker_env_.device_mgr,
                                               dev_resolver.get(), worker_cache,
                                               default_worker_name));
    worker_env_.collective_executor_mgr = new RpcCollectiveExecutorMgr(
        config, worker_env_.device_mgr, std::move(dev_resolver),
        std::move(param_resolver), worker_cache, default_worker_name);
  }

  // Set up worker environment.
  worker_env_.session_mgr = new SessionMgr(
      &worker_env_, SessionMgr::WorkerNameFromServerDef(server_def_),
      std::unique_ptr<WorkerCacheInterface>(worker_cache),
      [this](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
        WorkerCacheFactoryOptions options(server_def);
        return WorkerCacheFactory(options, worker_cache);
      });
  worker_env_.compute_pool = ComputePool(sess_opts);

  // Finish setting up master environment.
  master_env_.ops = OpRegistry::Global();
  master_env_.worker_cache = worker_cache;
  master_env_.collective_executor_mgr = worker_env_.collective_executor_mgr;
  StatsPublisherFactory stats_factory = opts.stats_factory;
  master_env_.master_session_factory =
      [config, stats_factory](
          SessionOptions options, const MasterEnv* env,
          std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
          std::unique_ptr<WorkerCacheInterface> worker_cache,
          std::unique_ptr<DeviceSet> device_set,
          std::vector<string> filtered_worker_list) {
        options.config.MergeFrom(config);
        return new MasterSession(options, env, std::move(remote_devs),
                                 std::move(worker_cache), std::move(device_set),
                                 std::move(filtered_worker_list),
                                 stats_factory);
      };
  master_env_.worker_cache_factory =
      [this](const WorkerCacheFactoryOptions& options,
             WorkerCacheInterface** worker_cache) {
        return WorkerCacheFactory(options, worker_cache);
      };

  // Provide direct access to the master from in-process clients.
  LocalMaster::Register(target(), master_impl_.get(),
                        config.operation_timeout_in_ms());

  return Status::OK();
}

Status ElasticGrpcServer::ParseChannelSpec(const WorkerCacheFactoryOptions& options,
                                    GrpcChannelSpec* channel_spec) {
  for (const auto& job : options.cluster_def->job()) {
    std::map<int, string> host_ports;
    for (const auto& task : job.tasks()) {
      string& host_port = host_ports[task.first];
      if (!host_port.empty()) {
        return errors::InvalidArgument("JobDef for job \"", job.name(),
                                       "\" specified two addresses for task \"",
                                       task.first, "\": ", host_port, " and ",
                                       task.second);
      }
      if (job.name() == *options.job_name && task.first == options.task_index) {
        host_port = strings::StrCat("localhost:", bound_port_);
      } else {
        host_port = task.second;
      }
    }
    TF_RETURN_IF_ERROR(channel_spec->AddHostPortsJob(job.name(), host_ports));
  }
  return Status::OK();
}

Status ElasticGrpcServer::WorkerCacheFactory(const WorkerCacheFactoryOptions& options,
                                      WorkerCacheInterface** worker_cache) {
  if (options.job_name == nullptr || options.job_name->empty()) {
    Status s = errors::InvalidArgument(
        "The master (current machine) is not included in the provided "
        "cluster_def. ",
        options.cluster_def->DebugString());
    LOG(WARNING) << s;
    return s;
  }

  GrpcChannelSpec channel_spec;
  TF_RETURN_IF_ERROR(ParseChannelSpec(options, &channel_spec));

  std::shared_ptr<GrpcChannelCache> channel_cache(
      NewGrpcChannelCache(channel_spec, GetChannelCreationFunction()));

  string name_prefix = strings::StrCat("/job:", *options.job_name, "/replica:0",
                                       "/task:", options.task_index);

  const string host_port = channel_cache->TranslateTask(name_prefix);
  int requested_port;

  auto colon_index = host_port.find_last_of(':');
  if (!strings::safe_strto32(host_port.substr(colon_index + 1),
                             &requested_port)) {
    return errors::Internal("Could not parse port for local server from \"",
                            host_port, "\".");
  }

  if (requested_port != bound_port_) {
    return errors::InvalidArgument("Requested port ", requested_port,
                                   " differs from expected port ", bound_port_);
  }

  *worker_cache = NewGrpcWorkerCacheWithLocalWorker(channel_cache,
                                                    worker_impl(), name_prefix);

  for (auto device : master_env_.local_devices) {
    ResourceMgr *rm = device->resource_manager();
    WorkerResource *worker_resource = new WorkerResource();
    worker_resource->worker_cache = *worker_cache;
    TF_RETURN_IF_ERROR(rm->Create("worker_resource", "worker_resource", worker_resource));
  }

  return Status::OK();
}

Status ElasticGrpcServer::UpdateWorkerCacheFactory(const WorkerCacheFactoryOptions& options,
                                      WorkerCacheInterface** worker_cache) {
  if (options.job_name == nullptr || options.job_name->empty()) {
    Status s = errors::InvalidArgument(
        "The master (current machine) is not included in the provided "
        "cluster_def. ",
        options.cluster_def->DebugString());
    LOG(WARNING) << s;
    return s;
  }

  GrpcChannelSpec channel_spec;
  TF_RETURN_IF_ERROR(ParseChannelSpec(options, &channel_spec));

  std::shared_ptr<GrpcChannelCache> channel_cache(
      NewGrpcChannelCache(channel_spec, GetChannelCreationFunction()));

  string name_prefix = strings::StrCat("/job:", *options.job_name, "/replica:0",
                                       "/task:", options.task_index);

  const string host_port = channel_cache->TranslateTask(name_prefix);
  int requested_port;

  auto colon_index = host_port.find_last_of(':');
  if (!strings::safe_strto32(host_port.substr(colon_index + 1),
                             &requested_port)) {
    return errors::Internal("Could not parse port for local server from \"",
                            host_port, "\".");
  }

  if (requested_port != bound_port_) {
    return errors::InvalidArgument("Requested port ", requested_port,
                                   " differs from expected port ", bound_port_);
  }

  *worker_cache = NewGrpcWorkerCacheWithLocalWorker(channel_cache,
                                                    worker_impl(), name_prefix);

  for (auto device : master_env_.local_devices) {
    ResourceMgr *rm = device->resource_manager();
    WorkerResource *worker_resource;
    TF_RETURN_IF_ERROR(rm->Lookup("worker_resource", "worker_resource", &worker_resource));
    worker_resource->worker_cache = *worker_cache;
  }

  return Status::OK();
}

Status ElasticGrpcServer::Start() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW: {
      master_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_master_service",
                            [this] { master_service_->HandleRPCsLoop(); }));
      worker_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_worker_service",
                            [this] { worker_service_->HandleRPCsLoop(); }));
      eager_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_eager_service",
                            [this] { eager_service_->HandleRPCsLoop(); }));
      update_server_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_elastic_service",
                            [this] { elastic_service_->HandleRPCsLoop(); }));
      state_ = STARTED;
      LOG(INFO) << "Started server with target: " << target();
      return Status::OK();
    }
    case STARTED:
      LOG(INFO) << "Server already started (target: " << target() << ")";
      return Status::OK();
    case STOPPED:
      return errors::FailedPrecondition("Server has stopped.");
    default:
      LOG(FATAL);
  }
}

Status ElasticGrpcServer::AddMasterEagerContextToEagerService(
    const tensorflow::uint64 context_id, tensorflow::EagerContext* context) {
  auto* eager_service =
      static_cast<eager::GrpcEagerServiceImpl*>(eager_service_);
  return eager_service->CreateMasterContext(context_id, context);
}

Status ElasticGrpcServer::Stop() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
      return errors::Unimplemented(
          "Clean shutdown is not currently implemented");
    case STOPPED:
      LOG(INFO) << "Server already stopped (target: " << target() << ")";
      return Status::OK();
    default:
      LOG(FATAL);
  }
}

Status ElasticGrpcServer::Join() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      // Prevent the server from being started subsequently.
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
    case STOPPED:
      master_thread_.reset();
      worker_thread_.reset();
      eager_thread_.reset();
      return Status::OK();
    default:
      LOG(FATAL);
  }
}

const string ElasticGrpcServer::target() const {
  return strings::StrCat("grpc://localhost:", bound_port_);
}

std::shared_ptr<::grpc::ServerCredentials> ElasticGrpcServer::GetServerCredentials(
    const ServerDef& server_def) const {
  return ::grpc::InsecureServerCredentials();
}

ChannelCreationFunction ElasticGrpcServer::GetChannelCreationFunction() const {
  // We can do this because SparseGrpcChannelCache is robust to nullptr being
  // returned by the channel creation function
  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
}

std::unique_ptr<Master> ElasticGrpcServer::CreateMaster(MasterEnv* master_env) {
  return std::unique_ptr<Master>(new Master(master_env, 0.0));
}

/* static */
Status ElasticGrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<ElasticGrpcServer> ret(
      new ElasticGrpcServer(server_def, env == nullptr ? Env::Default() : env));
  ServiceInitFunction service_func = nullptr;
  GrpcServerOptions options;
  options.rendezvous_mgr_func = NewRpcRendezvousMgr;
  Status s = ret->Init(options);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return s;
  }
  *out_server = std::move(ret);
  return Status::OK();
}

/* static */
Status ElasticGrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ElasticGrpcServer>* out_server) {
  std::unique_ptr<ElasticGrpcServer> ret(
      new ElasticGrpcServer(server_def, env == nullptr ? Env::Default() : env));
  GrpcServerOptions options;
  options.rendezvous_mgr_func = NewRpcRendezvousMgr;
  Status s = ret->Init(options);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return s;
  }
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {

class ElasticGrpcServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "elastic-grpc";
  }

  Status NewServer(const ServerDef& server_def,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return ElasticGrpcServer::Create(server_def, Env::Default(), out_server);
  }
};

// Registers a `ServerFactory` for `ElasticGrpcServer` instances.
class ElasticGrpcServerRegistrar {
 public:
  ElasticGrpcServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    memset(&alloc_fns, 0, sizeof(alloc_fns));
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("ELASTIC_GRPC_SERVER", new ElasticGrpcServerFactory());
  }
};
static ElasticGrpcServerRegistrar registrar;

}  // namespace
}  // namespace tensorflow
