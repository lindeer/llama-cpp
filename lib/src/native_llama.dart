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

  final ffi.Pointer<llama_cpp.llama_model> model;
  final ffi.Pointer<llama_cpp.llama_context> ctx;
  final llama_cpp.llama_batch batch;

  NativeLLama._(
    this.model,
    this.ctx,
    this.batch,
  );

  factory NativeLLama(String path, LlamaParams params) {
    final cStr = NativeString();
    final modelParams = llama_cpp.llama_model_default_params()
      ..n_gpu_layers = params.nGpuLayers;
    final model = llama_cpp.llama_load_model_from_file(path.into(cStr), modelParams);
    cStr.dispose();

    final ctxParams = llama_cpp.llama_context_default_params()
      ..seed = params.seed
      ..n_ctx = params.nCtx
      ..n_threads = params.nThread
      ..n_threads_batch = 4;
    final ctx = llama_cpp.llama_new_context_with_model(model, ctxParams);
    llama_cpp.llama_backend_init(false);
    final batch = llama_cpp.llama_batch_init(512, 0, 1);

    return NativeLLama._(
      model,
      ctx,
      batch,
    );
  }

  void dispose() {
    llama_cpp.llama_batch_free(batch);
    llama_cpp.llama_free(ctx);
    llama_cpp.llama_free_model(model);
    llama_cpp.llama_backend_free();
  }

  Stream<String> generate(String prompt) async* {
    llama_cpp.llama_reset_timings(ctx);

    llama_cpp.llama_print_timings(ctx);
  }
}
