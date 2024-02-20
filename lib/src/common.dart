import 'dart:ffi' as ffi;
import 'dart:math' as m;

import 'package:llama_cpp/native_llama_cpp.dart' as llama_cpp;

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
