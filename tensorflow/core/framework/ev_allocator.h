/* Copyright 2021 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_EV_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EV_ALLOCATOR_H_

#include <atomic>
#include <list>
#include <vector>
#include <readerwriterqueue.h>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/core/spin_lock.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define ARENA_ARRAY_SIZE 128

namespace tensorflow {

// If true, ev allocator collects more stats.
static bool ev_allocator_collect_stats = false;

static const int kMaxTotalAllocationWarnings = 1;

static const int kMaxSingleAllocationWarnings = 5;

// If ev_allocator_collect_stats is true, warn when the total allocated memory
// exceeds this threshold.
static const double kTotalAllocationWarningThreshold = 0.5;

// Individual allocations large than this amount will trigger a warning.
static const double kLargeAllocationWarningThreshold = 0.1;

// The max num of ptr that ThreadLocalBin can cache.
static const int kThreadLocalBinMaxPtrNum = 16;

namespace {
constexpr size_t kChunkSize = (1 << 22);  // 4MB chunk size

constexpr size_t kLargeChunkSize = (25 << 22);  // 100MB chunk size

constexpr int kAddressBits = (sizeof(void*) < 8 ? (8 * sizeof(void*)) : 48);

template <typename ChunkType>
class Chunk;

template <typename ChunkType>
class Bin;

template <typename ChunkType>
class PageMap {
 public:
  PageMap()
      : root_{},
        bytes_used_(0),
        page_shift_(0),
        npages_(0),
        bits_(0),
        root_bits_(0),
        root_length_(0) {}

  ~PageMap() { delete root_; }

  void Init();

  void InitInternal() {
    bits_ = kAddressBits - page_shift_;
    root_bits_ = (bits_ >= kLeafBits) ? (bits_ - kLeafBits) : 0;
    // (1<<root_bits_) must not overflow an "int"
    assert(root_bits_ < sizeof(int) * 8 - 1 && "root_bits is too large");
    root_length_ = 1 << root_bits_;
    root_ = new Leaf*[root_length_]();
  }

  Chunk<ChunkType>* GetChunk(const void* ptr) const {
    const auto k = reinterpret_cast<std::uintptr_t>(ptr) >> page_shift_;
    const auto i1 = k >> kLeafBits;
    const auto i2 = k & (kLeafLength - 1);
    if ((k >> bits_) > 0 || root_[i1] == nullptr) {
      return nullptr;
    }
    return root_[i1]->bin[i2];
  }

  void SetChunk(const void* ptr, Chunk<ChunkType>* b) {
    const auto start = reinterpret_cast<std::uintptr_t>(ptr) >> page_shift_;
    std::lock_guard<spin_lock> l(lock_);
    for (auto key = start; key < start + npages_; ++key) {
      const auto i1 = key >> kLeafBits;
      const auto i2 = key & (kLeafLength - 1);

      CHECK(i1 < root_length_);
      if (root_[i1] == nullptr) {
        Leaf* leaf = new Leaf;
        CHECK(leaf != nullptr);
        memset(leaf, 0, sizeof(*leaf));
        bytes_used_ += sizeof(Leaf);
        root_[i1] = leaf;
      }
      root_[i1]->bin[i2] = b;
    }
  }

 private:
  static constexpr int kLeafBits = 15;
  static constexpr int kLeafLength = 1 << kLeafBits;

  struct Leaf {
    Chunk<ChunkType>* bin[kLeafLength];
  };

  mutable spin_lock lock_;
  Leaf** root_;  // Top-level node
  size_t bytes_used_;
  size_t page_shift_;
  size_t npages_;
  int bits_;
  int root_bits_;
  int root_length_;
};

class FreeChunk {
 public:
  FreeChunk(size_t slot_size) : nfree(0), slot_size_(slot_size), ptrs(nullptr) {
    ptrs = new void*[kChunkSize / slot_size_];
  }

  FreeChunk(const FreeChunk& other) {
    nfree = other.nfree;
    slot_size_ = other.slot_size_;
    ptrs = new void*[kChunkSize / slot_size_];
    for (int i = 0; i < other.nfree; ++i) {
      ptrs[i] = other.ptrs[i];
    }
  }

  FreeChunk(FreeChunk&& other) {
    nfree = other.nfree;
    slot_size_ = other.slot_size_;
    ptrs = other.ptrs;
    other.ptrs = nullptr;
  }

  FreeChunk& operator=(const FreeChunk& other) {
    nfree = other.nfree;
    slot_size_ = other.slot_size_;
    if (this != &other) {
      for (int i = 0; i < other.nfree; ++i) {
        ptrs[i] = other.ptrs[i];
      }
    }

    return *this;
  }

  FreeChunk& operator=(FreeChunk&& other) {
    nfree = other.nfree;
    slot_size_ = other.slot_size_;
    if (this != &other) {
      if (ptrs != nullptr) {
        delete[] ptrs;
      }
      ptrs = other.ptrs;
      other.ptrs = nullptr;
    }

    return *this;
  }

  ~FreeChunk() {
    if (ptrs != nullptr) {
      delete[] ptrs;
    }
  }

  size_t nfree;
  size_t slot_size_;
  void** ptrs;
};

class FreeQueue {
 public:
  FreeQueue(size_t bin_size) : bin_size_(bin_size) {
    q_.reset(new moodycamel::ReaderWriterQueue<FreeChunk>);
  }

  int Size() { return q_->size_approx(); }

  // PushBatch and PopBatch do not guarantee an ordering.
  void Push(FreeChunk& free_chunk) { q_->emplace(std::move(free_chunk)); }

  int Pop(FreeChunk& ret) { return q_->try_dequeue(ret); }

  // PushBatch and PopBatch do not guarantee an ordering.
  void PushBatch(std::vector<FreeChunk>& free_chunk) {
    for (auto& fchunk : free_chunk) {
      q_->emplace(std::move(fchunk));
    }
  }

  int PopBatch(int N, std::vector<FreeChunk>& ret) {
    int pop_count = 0;
    while (pop_count < N) {
      bool succeeded = q_->try_dequeue(ret[pop_count]);
      if (!succeeded) {
        break;
      }
      ++pop_count;
    }
    return pop_count;
  }

 private:
  // NOTE(TODO): Consider to use concurrentqueue instead,
  // that we can delete mutex in Bin.
  std::unique_ptr<moodycamel::ReaderWriterQueue<FreeChunk>> q_;
  size_t bin_size_;
};

template <typename ChunkType>
class Chunk {
 public:
  Chunk(size_t chunk_size, size_t slot_size)
      : chunk_size_(chunk_size),
        slot_size_(slot_size),
        allocated_count_(0),
        slot_count_(chunk_size / slot_size) {}

  virtual ~Chunk() {}

  virtual void GetMemBlock() { start_ = nullptr; }

  virtual void ReleaseMemBlock() {}

  void Init(Bin<ChunkType>* bin, PageMap<ChunkType>* pm) {
    GetMemBlock();
    if (start_ == nullptr) {
      LOG(FATAL) << "OOM, can't create new Chunk for EVAllocator, "
                 << "please check free memory.";
    }
    bin_ = bin;
    pm->SetChunk(start_, this);
    current_ = start_;
    end_ = start_ + chunk_size_;
  }

  void* Allocate() {
    if (current_ + slot_size_ <= end_) {
      auto ret = current_;
      allocated_count_++;
      current_ += slot_size_;
      return ret;
    }
    return nullptr;
  }

  size_t BatchAllocate(size_t num, void** ret) {
    for (int i = 0; i < num; ++i) {
      if (current_ + slot_size_ > end_) {
        return i;
      }
      ret[i] = current_;
      allocated_count_++;
      current_ += slot_size_;
    }
    return num;
  }

  size_t FullAllocate(void** ret) {
    for (int i = 0; i < slot_count_; ++i) {
      ret[i] = current_;
      current_ += slot_size_;
      allocated_count_++;
    }
    return slot_count_;
  }

  size_t Count() { return slot_count_; }

  size_t AllocatedSlot() { return allocated_count_; }

  void DecreSlot(void* ptr) { allocated_count_--; }

  Bin<ChunkType>* AssignedBin() { return bin_; }

 protected:
  size_t chunk_size_;
  void* start_ = nullptr;

 private:
  void* current_ = nullptr;
  void* end_ = nullptr;
  size_t slot_size_;
  size_t slot_count_;
  size_t allocated_count_;
  Bin<ChunkType>* bin_;
};

template <typename ChunkType>
class Bin {
 public:
  Bin(size_t s, PageMap<ChunkType>* pm)
      : bin_size_(s), page_map_(pm), buffer_(s), free_queue_(s) {
    current_chunk_ = CreateChunk();
  }

  ~Bin() {
    for (auto it : chunks_) {
      delete it;
    }
  }

  size_t Allocate(FreeChunk& ret) {
    mutex_lock l(mu_);
    auto allocated_chunk = free_queue_.Pop(ret);
    auto remains = 1 - allocated_chunk;
    if (remains == 0) {
      return 1;
    }

    auto& cur = ret.ptrs;
    int allocated = 0;

    const int full_size = kChunkSize / bin_size_;
    allocated = current_chunk_->BatchAllocate(full_size, cur);
    ret.nfree = allocated;
    if (allocated == full_size) {
      return 1;
    }

    cur += allocated;
    current_chunk_ = CreateChunk();
    allocated = current_chunk_->BatchAllocate(full_size - allocated, cur);
    ret.nfree += allocated;
    return 1;
  }

  size_t BatchAllocate(size_t num, std::vector<FreeChunk>& ret) {
    mutex_lock l(mu_);
    auto allocated_chunk = free_queue_.PopBatch(num, ret);
    auto remain_chunk_num = num - allocated_chunk;
    if (remain_chunk_num == 0) {
      return num;
    }
    if (remain_chunk_num > 1) {
      for (int i = 0; i < remain_chunk_num - 1; ++i) {
        auto& free_chunk = ret[allocated_chunk];
        auto& cur = free_chunk.ptrs;
        current_chunk_ = CreateChunk();
        auto allocated = current_chunk_->FullAllocate(cur);
        free_chunk.nfree = allocated;
        allocated_chunk++;
      }
      current_chunk_ = CreateChunk();
    }

    auto& free_chunk = ret[allocated_chunk];
    auto& cur = free_chunk.ptrs;
    int allocated = current_chunk_->BatchAllocate(kChunkSize / bin_size_, cur);
    free_chunk.nfree = allocated;
    if (allocated == kChunkSize / bin_size_) {
      return num;
    }

    cur += allocated;
    current_chunk_ = CreateChunk();
    allocated =
        current_chunk_->BatchAllocate(kChunkSize / bin_size_ - allocated, cur);
    free_chunk.nfree += allocated;
    return num;
  }

  void Deallocate(FreeChunk& free_chunk) {
    mutex_lock l(mu_);
    int free_ptr_num = free_chunk.nfree;
    if (free_ptr_num == kChunkSize / bin_size_) {
      free_queue_.Push(free_chunk);
      return;
    }
    int ptr_to_move = kChunkSize / bin_size_ - buffer_.nfree;
    if (ptr_to_move > free_ptr_num) {
      for (int i = 0; i < free_ptr_num; ++i) {
        buffer_.ptrs[buffer_.nfree++] = free_chunk.ptrs[i];
      }
      auto tmp_free_chunk = std::move(free_chunk);
    } else {
      for (int i = 0; i < ptr_to_move; ++i) {
        buffer_.ptrs[buffer_.nfree++] = free_chunk.ptrs[i];
      }
      free_chunk.nfree -= ptr_to_move;
      free_queue_.Push(buffer_);
      buffer_ = std::move(free_chunk);
    }
  }

  void ReleaseFreeMemory(FreeChunk& free_chunk) {
    mutex_lock l(mu_);
    for (int i = 0; i < free_chunk.nfree; ++i) {
      Chunk<ChunkType>* chunk = page_map_->GetChunk(free_chunk.ptrs[i]);
      chunk->DecreSlot(free_chunk.ptrs[i]);
    }
    free_chunk.nfree = 0;
    for (int i = 0; i < buffer_.nfree; ++i) {
      Chunk<ChunkType>* chunk = page_map_->GetChunk(buffer_.ptrs[i]);
      chunk->DecreSlot(buffer_.ptrs[i]);
    }
    buffer_.nfree = 0;

    int queue_size = free_queue_.Size();
    {
      std::vector<FreeChunk> free_chunk_list(queue_size, bin_size_);
      auto allocated_chunk = free_queue_.PopBatch(queue_size, free_chunk_list);
      for (int i = 0; i < queue_size; ++i) {
        auto& tmp_free_chunk = free_chunk_list[i];
        for (int j = 0; j < tmp_free_chunk.nfree; ++j) {
          Chunk<ChunkType>* chunk = page_map_->GetChunk(tmp_free_chunk.ptrs[j]);
          chunk->DecreSlot(tmp_free_chunk.ptrs[j]);
        }
      }
    }
    auto should_remove = [](Chunk<ChunkType>* chunk) {
      return chunk->AllocatedSlot() == 0;
    };

    for (auto* chunk : chunks_) {
      if (chunk == current_chunk_) continue;
      size_t allocated_count = chunk->AllocatedSlot();
      LOG(INFO) << "allocated_count: " << allocated_count;
      if (allocated_count == 0) {
        chunk->ReleaseMemBlock();
      }
    }
    chunks_.erase(std::remove_if(chunks_.begin(), chunks_.end(), should_remove),
                  chunks_.end());
  }

  size_t BinSize() const { return bin_size_; }

 private:
  Chunk<ChunkType>* CreateChunk() {
    auto c = new ChunkType(kChunkSize, bin_size_);
    c->Init(this, page_map_);
    chunks_.emplace_back(c);
    return c;
  }

 private:
  mutex mu_;
  size_t bin_size_;
  PageMap<ChunkType>* page_map_ = nullptr GUARDED_BY(mu_);
  Chunk<ChunkType>* current_chunk_ = nullptr GUARDED_BY(mu_);

  FreeChunk buffer_ GUARDED_BY(mu_);
  FreeQueue free_queue_ GUARDED_BY(mu_);
  std::vector<Chunk<ChunkType>*> chunks_ GUARDED_BY(mu_);
};

template <typename ChunkType>
class Arena {
 public:
  Arena(PageMap<ChunkType>* pm) : page_map_(pm) {}

  ~Arena() {
    for (auto it = bins_.begin(); it != bins_.end(); ++it) {
      delete it->second;
    }
    bins_.clear();
  }

  size_t Allocate(size_t bin_size, FreeChunk& ret) {
    auto* bin = GetOrCreate(bin_size);
    return bin->Allocate(ret);
  }

  size_t BatchAllocate(size_t num, size_t bin_size,
                       std::vector<FreeChunk>& ret) {
    auto* bin = GetOrCreate(bin_size);
    return bin->BatchAllocate(num, ret);
  }

  void Deallocate(size_t bin_size, FreeChunk& ptrs) {
    auto* bin = GetOrCreate(bin_size);
    return bin->Deallocate(ptrs);
  }

 private:
  Bin<ChunkType>* GetOrCreate(size_t bin_size) {
    Bin<ChunkType>* bin = nullptr;
    mutex_lock l(mu_);
    auto it = bins_.find(bin_size);
    if (it == bins_.end()) {
      bin = new Bin<ChunkType>(bin_size, page_map_);
      bins_.emplace(bin_size, bin);
    } else {
      bin = it->second;
    }
    return bin;
  }

 private:
  mutex mu_;
  std::unordered_map<size_t, Bin<ChunkType>*> bins_ GUARDED_BY(mu_);
  PageMap<ChunkType>* page_map_ = nullptr;
};

template <typename ChunkType>
class ThreadLocalBin {
 public:
  ThreadLocalBin(size_t t_bin_size, PageMap<ChunkType>* pm,
                 Arena<ChunkType>* arena)
      : t_bin_size_(t_bin_size),
        chunk_slot_num_(kChunkSize / t_bin_size_),
        page_map_(pm),
        arena_(arena),
        free_chunk_(t_bin_size_) {}

  ~ThreadLocalBin() { FlushBackToArena(); }

  void* Allocate() {
    FetchFromArena();

    if (likely(free_chunk_.nfree > 0)) {
      void* ret = free_chunk_.ptrs[--free_chunk_.nfree];
      return ret;
    }

    return nullptr;
  }

  size_t BatchAllocate(size_t num, void** ret) {
    FetchFromArena();

    if (likely(free_chunk_.nfree >= num)) {
      for (int i = 0; i < num; i++) {
        ret[i] = free_chunk_.ptrs[--free_chunk_.nfree];
      }
      return num;
    }

    int prev_num = free_chunk_.nfree;
    int remain_num = num - prev_num;
    for (int i = 0; i < prev_num; i++) {
      ret[i] = free_chunk_.ptrs[--free_chunk_.nfree];
    }

    size_t chunk_num = remain_num / chunk_slot_num_;
    size_t remain_chunk_num = remain_num % chunk_slot_num_;
    if (remain_chunk_num > 0) {
      chunk_num += 1;
    }

    if (chunk_num == 1) {
      FreeChunk free_chunk(t_bin_size_);
      int allocate = arena_->Allocate(t_bin_size_, free_chunk);
      if (remain_chunk_num == 0) {
        int loop_num = free_chunk.nfree;
        for (int j = 0; j < loop_num; ++j) {
          ret[prev_num + j] = free_chunk.ptrs[--free_chunk.nfree];
        }
      } else {
        for (int j = 0; j < remain_chunk_num; ++j) {
          ret[prev_num + j] = free_chunk.ptrs[--free_chunk.nfree];
        }
      }
      return num;
    }

    std::vector<FreeChunk> chunk_list(chunk_num, t_bin_size_);
    int allocate = arena_->BatchAllocate(chunk_num, t_bin_size_, chunk_list);
    if (remain_chunk_num == 0) {
      for (int i = 0; i < chunk_num; ++i) {
        auto& chunk = chunk_list[i];
        for (int j = 0; j < chunk_slot_num_; ++j) {
          ret[i * chunk_slot_num_ + prev_num + j] = chunk.ptrs[--chunk.nfree];
        }
      }
    } else {
      for (int i = 0; i < chunk_num - 1; ++i) {
        auto& chunk = chunk_list[i];
        for (int j = 0; j < chunk_slot_num_; ++j) {
          ret[i * chunk_slot_num_ + prev_num + j] = chunk.ptrs[--chunk.nfree];
        }
      }

      auto& chunk = chunk_list[chunk_num - 1];
      for (int j = 0; j < remain_chunk_num; ++j) {
        ret[(chunk_num - 1) * chunk_slot_num_ + prev_num + j] =
            chunk.ptrs[--chunk.nfree];
      }
    }

    return num;
  }

  void Deallocate(void* ptr) {
    if (likely(free_chunk_.nfree < chunk_slot_num_)) {
      free_chunk_.ptrs[free_chunk_.nfree++] = ptr;
      return;
    }
    FlushBackToArena();
    free_chunk_ = FreeChunk(t_bin_size_);
    free_chunk_.nfree = 1;
    free_chunk_.ptrs[0] = ptr;
    return;
  }

  void ReleaseFreeMemory() {
    if (free_chunk_.nfree > 0) {
      Bin<ChunkType>* bin =
          page_map_->GetChunk(free_chunk_.ptrs[0])->AssignedBin();
      bin->ReleaseFreeMemory(free_chunk_);
      return;
    }
  }

 private:
  void FetchFromArena() {
    if (free_chunk_.nfree == 0) {
      size_t ptrs_num = arena_->Allocate(t_bin_size_, free_chunk_);
    }
  }

  void FlushBackToArena() {
    if (free_chunk_.nfree > 0) {
      Bin<ChunkType>* bin =
          page_map_->GetChunk(free_chunk_.ptrs[0])->AssignedBin();
      bin->Deallocate(free_chunk_);
      return;
    }
  }

 private:
  size_t t_bin_size_;
  size_t chunk_slot_num_;
  PageMap<ChunkType>* page_map_ = nullptr;  // not owned
  Arena<ChunkType>* arena_ = nullptr;       // not owned
  FreeChunk free_chunk_;
};

template <typename ChunkType>
class ThreadLocalCache {
 public:
  ThreadLocalCache(PageMap<ChunkType>* pm, Arena<ChunkType>* arena)
      : page_map_(pm), arena_(arena) {}

  ~ThreadLocalCache() {
    for (auto it = t_bins_.begin(); it != t_bins_.end(); ++it) {
      delete it->second;
    }
    t_bins_.clear();
  }

  void* Allocate(size_t num_bytes) {
    auto* t_bin = GetOrCreate(num_bytes);
    return t_bin->Allocate();
  }

  size_t BatchAllocate(size_t num, size_t num_bytes, void** ret) {
    auto* t_bin = GetOrCreate(num_bytes);
    return t_bin->BatchAllocate(num, ret);
  }

  void Deallocate(size_t num_bytes, void* ptr) {
    auto* t_bin = GetOrCreate(num_bytes);
    t_bin->Deallocate(ptr);
  }

  void ReleaseFreeMemory() {
    for (auto& it : t_bins_) {
      it.second->ReleaseFreeMemory();
    }
  }

 private:
  ThreadLocalBin<ChunkType>* GetOrCreate(size_t num_bytes) {
    auto it = t_bins_.find(num_bytes);
    if (it != t_bins_.end()) {
      return it->second;
    }
    auto bin = new ThreadLocalBin<ChunkType>(num_bytes, page_map_, arena_);
    t_bins_.emplace(num_bytes, bin);
    return bin;
  }

 private:
  PageMap<ChunkType>* page_map_ = nullptr;  // not owned
  Arena<ChunkType>* arena_ = nullptr;       // not owned
  std::unordered_map<size_t, ThreadLocalBin<ChunkType>*> t_bins_;
};

template <typename ChunkType>
class EVAllocatorImpl {
 public:
  EVAllocatorImpl() {
    pthread_key_create(&key_, ThreadLocalCacheCleanup);
    page_map_ = new PageMap<ChunkType>();
    page_map_->Init();

    int64 arena_array_size = ARENA_ARRAY_SIZE;
    Status s = ReadInt64FromEnvVar("ARENA_ARRAY_SIZE", ARENA_ARRAY_SIZE,
                                   &arena_array_size);
    if (!s.ok()) {
      LOG(ERROR) << "Read ARENA_ARRAY_SIZE env error: " << s.error_message();
    }
    LOG(INFO) << "EVAllocator set arena array size: " << arena_array_size;

    arenas_ = new std::vector<Arena<ChunkType>>(arena_array_size, page_map_);
    arena_cur_index = 0;
  }

  ~EVAllocatorImpl() {
    pthread_key_delete(key_);
    delete arenas_;
    delete page_map_;
  }

  void* Allocate(size_t num_bytes) {
    return GetThreadLocalCache()->Allocate(num_bytes);
  }

  size_t BatchAllocate(size_t num, size_t num_bytes, void** ret) {
    return GetThreadLocalCache()->BatchAllocate(num, num_bytes, ret);
  }

  void Deallocate(void* ptr) {
    GetThreadLocalCache()->Deallocate(AllocatedSize(ptr), ptr);
  }

  void ReleaseFreeMemory() { GetThreadLocalCache()->ReleaseFreeMemory(); }

  size_t AllocatedSize(const void* ptr) const {
    auto bin = page_map_->GetChunk(ptr)->AssignedBin();
    if (bin != nullptr) {
      return bin->BinSize();
    }
    return 0;
  }

 private:
  ThreadLocalCache<ChunkType>* GetThreadLocalCache() {
    ThreadLocalCache<ChunkType>* tCache =
        static_cast<ThreadLocalCache<ChunkType>*>(pthread_getspecific(key_));
    if (tCache == nullptr) {
      Arena<ChunkType>* arena = GetNewArena();
      tCache = new ThreadLocalCache<ChunkType>(page_map_, arena);
      pthread_setspecific(key_, tCache);
    }
    return tCache;
  }

  Arena<ChunkType>* GetNewArena() {
    Arena<ChunkType>* ret = nullptr;
    {
      mutex_lock l(mu_arena_index_);
      ret = &((*arenas_)[arena_cur_index]);
      arena_cur_index = (arena_cur_index + 1) % ARENA_ARRAY_SIZE;
    }

    return ret;
  }

  static void ThreadLocalCacheCleanup(void* ptr) {
    auto t_ptr = (ThreadLocalCache<ChunkType>*)ptr;
    delete t_ptr;
  }

 private:
  pthread_key_t key_;
  mutex mu_arena_index_;
  PageMap<ChunkType>* page_map_ = nullptr;
  std::vector<Arena<ChunkType>>* arenas_ = nullptr;
  int arena_cur_index GUARDED_BY(mu_arena_index_);
};

template <typename ChunkType>
class EVAllocator : public Allocator {
 public:
  EVAllocator()
      : single_allocation_warning_count_(0),
        total_allocation_warning_count_(0) {}

  ~EVAllocator() override = default;

  string Name() override { return ""; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    num_bytes = AlignedSize(num_bytes);

    if (num_bytes > kChunkSize) {
      LOG(FATAL) << "Allocation of " << num_bytes << " exceeds " << kChunkSize
                 << " in EVAllocator.";
    }

    void* p = impl_.Allocate(num_bytes);
    if (ev_allocator_collect_stats) {
      const std::size_t alloc_size = impl_.AllocatedSize(p);
      mutex_lock l(mu_);
      ++stats_.num_allocs;
      stats_.bytes_in_use += alloc_size;
      stats_.peak_bytes_in_use =
          std::max<int64>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
      stats_.largest_alloc_size =
          std::max<int64>(stats_.largest_alloc_size, alloc_size);
    }
    return p;
  }

  size_t BatchAllocateRaw(size_t num, size_t alignment, size_t num_bytes,
                          void** ret) override {
    num_bytes = AlignedSize(num_bytes);

    if (num_bytes > kChunkSize) {
      LOG(FATAL) << "Allocation of " << num_bytes << " exceeds " << kChunkSize
                 << " in EVAllocator.";
    }

    auto allocated_num = impl_.BatchAllocate(num, num_bytes, ret);
    if (allocated_num == 0) {
      LOG(WARNING) << "Can't allocate num:" << num
                   << ", num_bytes:" << num_bytes;
      return 0;
    }

    if (ev_allocator_collect_stats) {
      auto p = ret[0];
      const std::size_t alloc_size = impl_.AllocatedSize(p);
      mutex_lock l(mu_);
      stats_.num_allocs += allocated_num;
      stats_.bytes_in_use += alloc_size * allocated_num;
      stats_.peak_bytes_in_use =
          std::max<int64>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
      stats_.largest_alloc_size =
          std::max<int64>(stats_.largest_alloc_size, alloc_size);
    }
    return allocated_num;
  }

  void DeallocateRaw(void* ptr) override {
    if (ev_allocator_collect_stats) {
      const std::size_t alloc_size = impl_.AllocatedSize(ptr);

      mutex_lock l(mu_);
      stats_.bytes_in_use -= alloc_size;
    }

    impl_.Deallocate(ptr);
  }

  void ReleaseFreeMemory() { impl_.ReleaseFreeMemory(); }

  absl::optional<AllocatorStats> GetStats() override {
    mutex_lock l(mu_);
    return stats_;
  }

  void ClearStats() override {
    mutex_lock l(mu_);
    stats_.num_allocs = 0;
    stats_.peak_bytes_in_use = stats_.bytes_in_use;
    stats_.largest_alloc_size = 0;
  }

  size_t AllocatedSizeSlow(const void* ptr) const override {
    return impl_.AllocatedSize(ptr);
  }

 protected:
// Return the smallest alignment multiple that is >= s.
#define ALIGNMENT_CEILING(s, alignment) \
  (((s) + (alignment - 1)) & ((~(alignment)) + 1))

  size_t AlignedSize(size_t num_bytes) {
    // small allocation no need alignment here.
    if (num_bytes <= 4 * sizeof(float)) {
      return num_bytes;
    }

    // Use _mm_load_ps instructions need aligned address.
    return ALIGNMENT_CEILING(num_bytes, 4 * sizeof(float));
  }

 protected:
  mutex mu_;
  AllocatorStats stats_ GUARDED_BY(mu_);

  // Use <atomic> for single allocations to avoid mutex contention when
  // statistics are disabled.
  std::atomic<int> single_allocation_warning_count_;
  int total_allocation_warning_count_ GUARDED_BY(mu_);

  EVAllocatorImpl<ChunkType> impl_;

  TF_DISALLOW_COPY_AND_ASSIGN(EVAllocator);
};

}  // namespace

}  // namespace tensorflow

#endif  // _TENSORFLOW_CORE_FRAMEWORK_EV_ALLOCATOR_H_