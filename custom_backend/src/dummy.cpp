#include "dummy.hpp"
#include <iostream>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace c10d {


bool WorkDummy::isCompleted() {
  return true;
}

bool WorkDummy::isSuccess() const {
  return true;
}

bool WorkDummy::wait(std::chrono::milliseconds /* unused */) {
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkDummy::getFuture() {
  return future_;
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
BackendDummy::BackendDummy(int rank, int size)
    : Backend(rank, size) {}

// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendDummy::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  for (auto& outputTensorVec : outputTensors) {
      for (auto& outputTensor : outputTensorVec) {
          outputTensor.zero_();
      }
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::_allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendDummy::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  for (auto& tensor : tensors) {
      tensor.zero_();
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::barrier(
    const BarrierOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Backend> BackendDummy::createBackendDummy(
    const c10::intrusive_ptr<::c10d::Store>& /* unused */,
    int rank,
    int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<BackendDummy>(rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createBackendDummy", &BackendDummy::createBackendDummy);
}

#define IMPL_ALLREDUCE(DEV)                                                   \
  std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>               \
      allreduce_##DEV(                                                        \
          at::TensorList tensors,                                             \
          const c10::intrusive_ptr<ProcessGroup>& process_group,              \
          const c10::intrusive_ptr<ReduceOp>& reduce_op,                      \
          const c10::optional<at::Tensor>& sparse_indices,                    \
          int64_t timeout) {                                                  \
    std::cout << "My custom c++ meta allreduce!" << std::endl;                \
    auto tensor_vec = tensors.vec();                                          \
    auto work = process_group->getBackend(c10::DeviceType::DEV) -> allreduce( \
        tensor_vec,                                                           \
        AllreduceOptions{                                                     \
            *reduce_op.get(), std::chrono::milliseconds(timeout)});           \
    return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(     \
        std::move(tensor_vec), work);                                         \
  }

IMPL_ALLREDUCE(CPU)

TORCH_LIBRARY_IMPL(c10d, Meta, m) {
    m.impl("allreduce_", allreduce_CPU);
}

// #define IMPL_BROADCAST(DEV)                                                   \
//   std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>               \
//       broadcast_##DEV(                                                        \
//           at::TensorList tensors,                                             \
//           const c10::intrusive_ptr<ProcessGroup>& process_group,              \
//           int64_t root_rank,                                                  \
//           int64_t root_tensor,                                                \
//           bool asyncOp,                                                       \
//           int64_t timeout) {                                                  \
//     auto tensor_vec = tensors.vec();                                          \
//     auto work = process_group->getBackend(c10::DeviceType::DEV) -> broadcast( \
//         tensor_vec,                                                           \
//         BroadcastOptions{                                                     \
//             root_rank,                                                        \
//             root_tensor,                                                      \
//             std::chrono::milliseconds(timeout),                               \
//             asyncOp});                                                        \
//     return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(     \
//         std::move(tensor_vec), work);                                         \
//   }

// IMPL_BROADCAST(CPU)

// TORCH_LIBRARY_IMPL(c10d, Meta, m) {
//     m.impl("broadcast_", broadcast_CPU);
// }

} // namespace c10d
