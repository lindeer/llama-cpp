import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

import 'native_llama_cpp.dart' as llama_cpp;

String _fromNative(ffi.Pointer<ffi.Char> pointer, int len) => pointer.cast<Utf8>().toDartString(length: len);

ffi.Pointer<ffi.Char> _toNative(String text) => text.toNativeUtf8().cast<ffi.Char>();

void loadLlama(String path) {
  llama_cpp.llama_backend_init(false);
  final charPointer = path.toNativeUtf8().cast<ffi.Char>();
  final params = llama_cpp.llama_model_default_params();
  final model = llama_cpp.llama_load_model_from_file(charPointer, params);
  final ctxParams = llama_cpp.llama_context_default_params();
  final ctx = llama_cpp.llama_new_context_with_model(model, ctxParams);

  final contextSize = ctxParams.n_ctx;
  final batchSize = ctxParams.n_batch;
  final maxSize = contextSize - 4;

  final question = "人生的意义是什么？";
  final nq = question.toNativeUtf8(allocator: calloc);
  final buf = calloc.allocate<llama_cpp.llama_token>(maxSize * ffi.sizeOf<llama_cpp.llama_token>());
  final num = llama_cpp.llama_tokenize(
    model,
    nq.cast<ffi.Char>(), nq.length,
    buf, contextSize,
    false, false,
  );
  const bufSize = 20;
  final strBuf = calloc.allocate<ffi.Char>(bufSize * ffi.sizeOf<ffi.Char>());

  final nVocab = llama_cpp.llama_n_vocab(model);
  print("llama params: tokenSize=$num, maxSize=$maxSize, nVocab=$nVocab, "
      "contextSize=$contextSize, batchSize=$batchSize");
  if (num > maxSize) {
    return;
  }
  final tmp = StringBuffer();
  for (int i = 0; i < num; i++) {
    final len = llama_cpp.llama_token_to_piece(model, buf[i], strBuf, bufSize);
    final str = strBuf.cast<Utf8>().toDartString(length: len);
    tmp.write(str);
  }
  final s = tmp.toString().substring(1);
  print("question is '$tmp', len is [${tmp.length}], actual is ='$s'");

  final logits = llama_cpp.llama_get_logits(ctx);
  final candidates = calloc.allocate<llama_cpp.llama_token_data>(
      nVocab * ffi.sizeOf<llama_cpp.llama_token_data>());
  final array = calloc.allocate<llama_cpp.llama_token_data_array>(
      ffi.sizeOf<llama_cpp.llama_token_data>());

  llama_cpp.llama_reset_timings(ctx);

  var count = 0;
  while ((count = llama_cpp.llama_get_kv_cache_token_count(ctx)) < maxSize) {
    final ret = llama_cpp.llama_eval(ctx, buf, num, count);
    if (ret != 0) {
      print("Error: 'llama_eval' return $ret!");
      return;
    }
    for (int i = 0; i < nVocab; i++) {
      final c = candidates[i];
      c.id = i;
      c.logit = logits[i];
      c.p = 0.0;
    }
    array.ref.data = candidates;
    array.ref.size = nVocab;
    array.ref.sorted = false;

    final tokenId = llama_cpp.llama_sample_token_greedy(ctx, array);
    if (tokenId == llama_cpp.llama_token_eos(model)) {
      break;
    }
    final len = llama_cpp.llama_token_to_piece(model, tokenId, strBuf, bufSize);
    final str = strBuf.cast<Utf8>().toDartString(length: len);
    print(str);
  }

  llama_cpp.llama_print_timings(ctx);

  calloc.free(array);
  calloc.free(candidates);

  calloc.free(strBuf);
  calloc.free(nq);

  llama_cpp.llama_free(ctx);
  llama_cpp.llama_free_model(model);
  llama_cpp.llama_backend_free();
}
