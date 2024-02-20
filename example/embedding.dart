import 'dart:ffi' as ffi;
import 'dart:io' show stderr, stdout, Platform;
import 'dart:math' as m;

import 'package:ffi/ffi.dart' show calloc;
import 'package:llama_cpp/src/ffi.dart';
import 'package:llama_cpp/native_llama_cpp.dart' as llama_cpp;

int main(List<String> argv) {
  if (argv.isEmpty || argv[0].startsWith('-')) {
    print("usage: ${Platform.script.path} MODEL_PATH [PROMPT]");
    return 1;
  }
  final path = argv[0];
  final prompt = argv.length > 1 ? argv[1] : 'Hello my name is';
  llama_cpp.llama_backend_init(false);

  final cStr = NativeString();
  final modelParams = llama_cpp.llama_model_default_params();
  final model = llama_cpp.llama_load_model_from_file(path.into(cStr), modelParams);
  final ctxParams = llama_cpp.llama_context_default_params()
    ..embedding = true
    ..seed = 1234
    ..n_threads = 4
    ..n_threads_batch = 4;
  final ctx = llama_cpp.llama_new_context_with_model(model, ctxParams);
  const batchSize = 512;
  final batch = llama_cpp.llama_batch_init(batchSize, 0, 1);

  final nCtxTrain = llama_cpp.llama_n_ctx_train(model);
  final nCtx = llama_cpp.llama_n_ctx(ctx);
  print("model was trained on only $nCtxTrain context tokens ($nCtx specified)");
  final prompts = prompt.split('\n').map((e) => e.trim())
      .where((e) => e.isNotEmpty).toList(growable: false);
  llama_cpp.llama_reset_timings(ctx);
  final maxTokenSize = prompts.map((e) => e.length).reduce(m.max);
  final tokens = TokenArray(size: maxTokenSize);
  final tokenList = prompts.map((p) {
    p.into(cStr);
    tokens.pavedBy(model, cStr, addBos: true);
    final l = tokens.toList();
    return l.length > batchSize ? l.sublist(0, batchSize) : l;
  });

  for (final (i, l) in tokenList.indexed) {
    print("main: prompt $i: '${prompts[i]}'");
    print("main: number of tokens in prompt = ${l.length}");
    for (final t in l) {
      print("${'$t'.padLeft(6)} -> '${cStr.tokenString(model, t)}'");
    }
  }

  final dimens = llama_cpp.llama_n_embd(model);
  final row = tokenList.length;
  final bytes = ffi.sizeOf<ffi.Float>() * row * dimens;
  final data = calloc.allocate<ffi.Float>(bytes);
  var out = data;
  var s = 0;
  for (final tokens in tokenList) {
    final len = tokens.length;
    if (batch.n_tokens + len > batchSize) {
      _decodeBatch(ctx, batch, out, s, dimens);
      batch.n_tokens = 0;
      out += s * dimens;
      s = 0;
    }
    addBatchSeq(batch, tokens, s);
    s++;
  }
  _decodeBatch(ctx, batch, out, s, dimens);
  for (var j = 0, pos = 0; j < row; j++, pos += dimens) {
    stdout.write("embedding $j: [");
    final p = data + pos;
    for (var i = 0; i < dimens; i++) {
      final v = (p + i).value;
      stdout.write("${v.toStringAsFixed(6)}, ");
    }
    stdout.writeln("]\n");
  }

  calloc.free(data);
  cStr.dispose();
  tokens.dispose();
  llama_cpp.llama_print_timings(ctx);

  llama_cpp.llama_batch_free(batch);
  llama_cpp.llama_free(ctx);
  llama_cpp.llama_free_model(model);
  llama_cpp.llama_backend_free();
  return 0;
}
