import 'dart:ffi' as ffi;
import 'dart:io' show Platform;
import 'dart:math' as m;

import 'ffi.dart';
import 'lib_llama_cpp.dart' as llama_cpp;
import 'llama_params.dart';

void _addLlamaBatch(
  llama_cpp.llama_batch batch,
  int id,
  int pos,
  int seq,
  bool logits,
) {
  final n = batch.n_tokens;
  batch.token[n] = id;
  batch.pos[n] = pos;
  batch.n_seq_id[n] = 1;
  batch.seq_id[n][0] = seq;
  batch.logits[n] = logits ? 1 : 0;

  batch.n_tokens++;
}

/// Append single token to batch with given position and logit.
void addBatchSingle(llama_cpp.llama_batch batch, int t, int pos, bool logit) {
  _addLlamaBatch(batch, t, pos, 0, logit);
}

void _addBatchTokens(
  llama_cpp.llama_batch batch,
  List<int> tokens,
  int pos,
  int seq,
  bool logit,
) {
  for (final (i, t) in tokens.indexed) {
    _addLlamaBatch(batch, t, pos + i, seq, logit);
  }
}

/// Add multiple tokens to batch with given [seq] from start.
void addBatchSeq(llama_cpp.llama_batch batch, List<int> tokens, int seq) {
  _addBatchTokens(batch, tokens, 0, seq, false);
}

/// Add multiple tokens to batch from [pos] with given [logit].
void addBatchPos(
  llama_cpp.llama_batch batch,
  List<int> tokens,
  int pos,
  bool logit,
) {
  _addBatchTokens(batch, tokens, pos, 0, logit);
}

void _normalize(ffi.Pointer<ffi.Float> vec, ffi.Pointer<ffi.Float> out, int n) {
  var norm = 0.0;
  for (var i = 0; i < n; i++) {
    final v = vec[i];
    norm += v * v;
  }
  norm = m.sqrt(norm);
  for (var i = 0; i < n; i++) {
    out[i] = vec[i] / norm;
  }
}

/// Decode batch only for embedding.
void decodeEmbeddingBatch(
  ffi.Pointer<llama_cpp.llama_context> ctx,
  llama_cpp.llama_batch batch,
  ffi.Pointer<ffi.Float> output,
  int seq,
  int dimens,
) {
  llama_cpp.llama_kv_cache_clear(ctx);
  print('decodeEmbeddingBatch: n_tokens = ${batch.n_tokens}, n_seq = $seq');
  if (llama_cpp.llama_decode(ctx, batch) < 0) {
    throw Exception('decodeEmbeddingBatch: failed to decode');
  }
  for (var k = 0; k < seq; k++) {
    final emb = llama_cpp.llama_get_embeddings_ith(ctx, k);
    final out = output + k * dimens;
    _normalize(emb, out, dimens);
  }
}

int get _physicalCores {
  final n = Platform.numberOfProcessors;
  return n > 4
      ? n ~/ 2
      : n > 0
          ? n
          : 4;
}

String _systemInfo(LlamaParams lp, llama_cpp.llama_context_params params) {
  final n = lp.nThreadBatch;
  final batch = n != null ? ' (n_threads_batch = $n)' : '';
  return 'system_info: n_threads = ${params.n_threads}$batch '
      '/ ${Platform.numberOfProcessors} '
      '| ${CharArray.toDartString(llama_cpp.llama_print_system_info())}';
}

/// Load a model from a given path, it could be a LLM also a embedding model.
/// return both model and context.
(ffi.Pointer<llama_cpp.llama_model>, ffi.Pointer<llama_cpp.llama_context>)
    loadModel(CharArray path, LlamaParams params) {
  final ctxSize = params.nCtx ?? 512;
  final s = params.seed ?? 0;
  final seed = s > 0 ? s : DateTime.now().millisecondsSinceEpoch ~/ 1000;
  print('seed = $seed');
  print('llama backend init');
  llama_cpp.llama_backend_init();
  llama_cpp.llama_numa_init(params.numa);
  final modelParams = llama_cpp.llama_model_default_params();
  final nGpuLayers = params.nGpuLayers;
  if (nGpuLayers != null) {
    modelParams.n_gpu_layers = nGpuLayers > 0 ? nGpuLayers : 0;
  }
  final mainGpu = params.mainGpu;
  if (mainGpu != null) {
    modelParams.main_gpu = mainGpu;
  }

  final model = llama_cpp.llama_load_model_from_file(path.pointer, modelParams);
  if (model.address == 0) {
    throw Exception("Load model from '${path.dartString}' failed");
  }

  final ctxParams = llama_cpp.llama_context_default_params()..seed = seed;
  if (ctxSize > 0) {
    ctxParams.n_ctx = ctxSize;
  }
  final nBatch = params.nBatch ?? -1;
  if (nBatch > 0) {
    ctxParams.n_batch = nBatch;
  }
  final t = params.nThread ?? 0;
  ctxParams.n_threads = t > 0 ? t : _physicalCores;
  final tb = params.nThreadBatch ?? 0;
  ctxParams.n_threads_batch = tb > 0 ? tb : ctxParams.n_threads;

  final ctx = llama_cpp.llama_new_context_with_model(model, ctxParams);
  if (ctx.address == 0) {
    throw Exception("Create llama context failed");
  }
  final nCtxTrain = llama_cpp.llama_n_ctx_train(model);
  final nCtx = llama_cpp.llama_n_ctx(ctx);
  print('n_ctx: $nCtx, train=$nCtxTrain');
  if (nCtx > nCtxTrain) {
    print('warning: model was trained on only $nCtxTrain context tokens '
        '($nCtx specified)');
  }
  print(_systemInfo(params, ctxParams));
  _warmup(model, ctx, ctxParams.n_batch);

  return (model, ctx);
}

void _warmup(ffi.Pointer<llama_cpp.llama_model> model,
    ffi.Pointer<llama_cpp.llama_context> ctx, int batchSize) {
  print('warming up the model with an empty run');
  final tokens = TokenArray(size: 2);
  tokens.add(llama_cpp.llama_token_bos(model));
  tokens.add(llama_cpp.llama_token_eos(model));
  final batch = llama_cpp.llama_batch_get_one(
    tokens.pointerAt(0),
    m.min(tokens.length, batchSize),
    0,
    0,
  );
  llama_cpp.llama_decode(ctx, batch);
  llama_cpp.llama_kv_cache_clear(ctx);
  llama_cpp.llama_reset_timings(ctx);
  tokens.dispose();
}
