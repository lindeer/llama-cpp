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
  final NativeString cStr;
  final tokenBuf = TokenArray(size: 64);
  final array = TokenDataArray(512);

  NativeLLama._(
    this.model,
    this.ctx,
    this.batch,
    this.cStr,
  );

  factory NativeLLama(String path, LlamaParams params) {
    final cStr = NativeString();
    final modelParams = llama_cpp.llama_model_default_params()
      ..n_gpu_layers = params.nGpuLayers;
    final model = llama_cpp.llama_load_model_from_file(path.into(cStr), modelParams);

    final ctxParams = llama_cpp.llama_context_default_params()
      ..seed = params.seed
      ..n_ctx = params.nCtx
      ..n_threads = params.nThread
      ..n_threads_batch = 4;
    final ctx = llama_cpp.llama_new_context_with_model(model, ctxParams);
    llama_cpp.llama_backend_init(false);
    llama_cpp.llama_kv_cache_clear(ctx);
    final batch = llama_cpp.llama_batch_init(512, 0, 1);

    return NativeLLama._(
      model,
      ctx,
      batch,
      cStr,
    );
  }

  void dispose() {
    array.dispose();
    tokenBuf.dispose();
    cStr.dispose();

    llama_cpp.llama_batch_free(batch);
    llama_cpp.llama_free(ctx);
    llama_cpp.llama_free_model(model);
    llama_cpp.llama_backend_free();
  }

  Stream<String> generate(String prompt) async* {
    prompt.into(cStr);
    tokenBuf.pavedBy(model, cStr);
    final nVocab = llama_cpp.llama_n_vocab(model);
    final eosToken = llama_cpp.llama_token_eos(model);

    var num = batch.n_tokens;
    var code = 0;

    llama_cpp.llama_reset_timings(ctx);
    while ((code = _decodeBatch(num, num == 0)) == 0) {
      final logits = llama_cpp.llama_get_logits_ith(ctx, batch.n_tokens - 1);
      array.pavedBy(logits, nVocab);
      final tokenId = llama_cpp.llama_sample_token_greedy(ctx, array.pointer);
      if (tokenId == eosToken) {
        code = 3;
        break;
      }
      final token = cStr.fromToken(model, tokenId);
      yield token;

      batch.n_tokens = 0;
      tokenBuf
        ..clear()
        ..add(tokenId);
      num++;
    }
    llama_cpp.llama_print_timings(ctx);
    print("sample llama logits finished with '$code'.");
    yield engTag;
  }

  int _decodeBatch(int count, bool init) {
    final tokenNum = tokenBuf.length;
    // evaluate the initial prompt
    for (var i = 0; i < tokenNum; i++) {
      _addLlamaBatch(tokenBuf[i], count + i, !init);
    }
    if (init) {
      batch.logits[batch.n_tokens - 1] = 1;
    }
    return llama_cpp.llama_decode(ctx, batch);
  }

  void _addLlamaBatch(int id, int pos, bool logits) {
    final n = batch.n_tokens;
    batch.token[n] = id;
    batch.pos[n] = pos;
    batch.n_seq_id[n] = 1;
    batch.seq_id[n][0] = 0;
    batch.logits[n] = logits ? 1 : 0;

    batch.n_tokens++;
  }
}
