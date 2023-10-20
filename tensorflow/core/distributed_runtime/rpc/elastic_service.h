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

#ifndef DYNAMIC_EMBEDDING_SERVER_INCLUDE_FRAMEWORK_ELASTIC_SERVICE_H_
#define DYNAMIC_EMBEDDING_SERVER_INCLUDE_FRAMEWORK_ELASTIC_SERVICE_H_


#include <memory>
#include "grpcpp/server_builder.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
class ElasticGrpcServer;

namespace tensorflow {

class AsyncServiceInterface;
AsyncServiceInterface* NewGrpcElasticService(ElasticGrpcServer* elastic_grpc_server,
    ::grpc::ServerBuilder* builder);

} // namespace tensorflow

#endif // DYNAMIC_EMBEDDING_SERVER_INCLUDE_FRAMEWORK_ELASTIC_SERVICE_H_