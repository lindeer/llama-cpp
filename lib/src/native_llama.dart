import 'dart:ffi' as ffi;

import 'ffi.dart';
import '../native_llama_cpp.dart' as llama_cpp;

final class LlamaParams {
  final int nCtx;
  final int nGpuLayers;
  final int seed;
  final int nThread;

  const LlamaParams(
    this.nCtx,
    this.nGpuLayers,
    this.seed,
    this.nThread,
  );
}

final class NativeLLama {
  static const engTag = '__end__';
  static const closeTag = '__close__';

  NativeLLama._(
  );

  factory NativeLLama(String path, LlamaParams params) {
    return NativeLLama._(
    );
  }

  void dispose() {
  }

  Stream<String> generate(String prompt) async* {
  }
}
