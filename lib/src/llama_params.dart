import 'dart:io' show Platform;

import 'package:llama_cpp/native_llama_cpp.dart' show llama_print_system_info;
import 'ffi.dart';

/// Params holder like `gpt_params` in `common/common.h`
final class LlamaParams {
  final int seed;
  final int nThread;
  final int nThreadBatch;
  final int nPredict;
  final int nCtx;
  final int nBatch;
  final int? nGpuLayers;
  final int? mainGpu;
  final bool numa;

  const LlamaParams(
    this.seed,
    this.nThread,
    this.nThreadBatch,
    this.nPredict,
    this.nCtx,
    this.nBatch,
    this.nGpuLayers,
    this.mainGpu,
    this.numa,
  );

  String get systemInfo {
    final batch =
        nThreadBatch != -1 ? ' (n_threads_batch = $nThreadBatch)' : '';
    return 'system_info: n_threads = $nThread$batch '
        '/ ${Platform.numberOfProcessors} '
        '| ${NativeString.fromNative(llama_print_system_info())}';
  }
}
