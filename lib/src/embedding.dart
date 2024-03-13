import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart' show calloc;

import 'common.dart' as c;
import 'ffi.dart';
import 'lib_llama_cpp.dart' as llama_cpp;
import 'llama_params.dart';

/// Embedding runs in current isolate.
/// Place it in another isolate if you want async embeddings.
final class Embedding {
  final ffi.Pointer<llama_cpp.llama_model> model;
  final ffi.Pointer<llama_cpp.llama_context> ctx;
  final CharArray cStr;
  final bool verbose;
  final tokenBuf = TokenArray(size: 64);

  Embedding._(
    this.model,
    this.ctx,
    this.cStr,
    this.verbose,
  );

  factory Embedding(
    String path, {
    int? nThread,
    int? nThreadBatch,
    int? nCtx,
    int? nBatch,
    int? nGpuLayers,
    bool verbose = false,
  }) {
    final cStr = CharArray.from(path);
    final (model, ctx) = c.loadModel(
      cStr,
      LlamaParams(
        nThread: nThread,
        nThreadBatch: nThreadBatch,
        nCtx: nCtx,
        nBatch: nBatch,
        nGpuLayers: nGpuLayers,
        embedding: true,
      ),
    );

    return Embedding._(
      model,
      ctx,
      cStr,
      verbose,
    );
  }

  /// Embedding multiple prompts at one time.
  List<List<double>> embedBatch(List<String> prompts) => _embed(prompts);

  /// Embedding one prompt at one time.
  List<double> embedSingle(String prompt) => _embed([prompt]).first;

  List<List<double>> _embed(List<String> prompts) {
    llama_cpp.llama_reset_timings(ctx);

    final batchSize = llama_cpp.llama_n_batch(ctx);
    final batch = llama_cpp.llama_batch_init(batchSize, 0, prompts.length);
    final arrayList = prompts.map((p) {
      cStr.pavedBy(p);
      tokenBuf.pavedBy(model, cStr, addBos: true);
      final l = tokenBuf.toList();
      return l.length > batchSize ? l.sublist(0, batchSize) : l;
    });
    if (verbose) {
      for (final (i, l) in arrayList.indexed) {
        print("main: prompt $i: '${prompts[i]}'");
        print("main: number of tokens in prompt = ${l.length}");
        for (final t in l) {
          print("${'$t'.padLeft(6)} -> '${cStr.tokenString(model, t)}'");
        }
      }
    }

    final dimens = llama_cpp.llama_n_embd(model);
    final row = arrayList.length;
    final data =
        calloc.allocate<ffi.Float>(ffi.sizeOf<ffi.Float>() * row * dimens);
    var out = data;
    var s = 0;
    for (final tokens in arrayList) {
      final len = tokens.length;
      if (batch.n_tokens + len > batchSize) {
        c.decodeEmbeddingBatch(ctx, batch, out, s, dimens);
        batch.n_tokens = 0;
        out += s * dimens;
        s = 0;
      }
      c.addBatchSeq(batch, tokens, s);
      s++;
    }
    c.decodeEmbeddingBatch(ctx, batch, out, s, dimens);

    final result = List<List<double>>.generate(row, (r) {
      final p = data + r * dimens;
      return List<double>.generate(
        dimens,
        (i) => (p[i] * 1000000).round() / 1000000,
        growable: false,
      );
    }, growable: false);
    llama_cpp.llama_print_timings(ctx);

    calloc.free(data);
    llama_cpp.llama_batch_free(batch);
    return result;
  }

  /// Free context, model and memory objects in C world.
  void dispose() {
    tokenBuf.dispose();
    cStr.dispose();

    llama_cpp.llama_free(ctx);
    llama_cpp.llama_free_model(model);
    llama_cpp.llama_backend_free();
    print('Embedding.dispose: done.');
  }
}
