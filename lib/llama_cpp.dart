import 'dart:convert' show utf8;
import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart';

import 'native_llama_cpp.dart' as llama_cpp;

String _fromNative(ffi.Pointer<ffi.Char> pointer, int len) => pointer.cast<Utf8>().toDartString(length: len);

int _toNative(String text, ffi.Pointer<ffi.Char> buf, int bufSize) {
  final units = utf8.encode(text);
  final size = units.length + 1;
  if (size > bufSize) {
    return -1;
  }
  final pointer = buf.cast<ffi.Uint8>();
  final nativeString = pointer.asTypedList(size);
  nativeString.setAll(0, units);
  nativeString[size - 1] = 0;
  return size - 1;
}

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
  const bufSize = 64;
  final strBuf = calloc.allocate<ffi.Char>(bufSize * ffi.sizeOf<ffi.Char>());
  var len = _toNative(question, strBuf, bufSize);
  if (len < 0) {
    print('_toNative failed!');
    exit(-1);
  }
  final tokenBuf = calloc.allocate<llama_cpp.llama_token>(maxSize * ffi.sizeOf<llama_cpp.llama_token>());
  var num = llama_cpp.llama_tokenize(
    model,
    strBuf, len,
    tokenBuf, maxSize,
    false, true,
  );

  final nVocab = llama_cpp.llama_n_vocab(model);
  print("llama params: tokenSize=$num, maxSize=$maxSize, len=$len, "
      "nVocab=$nVocab, contextSize=$contextSize, batchSize=$batchSize");
  if (num > maxSize) {
    return;
  }
  final tmp = StringBuffer();
  for (int i = 1; i < num; i++) {
    final len = llama_cpp.llama_token_to_piece(model, tokenBuf[i], strBuf, bufSize);
    final str = strBuf.cast<Utf8>().toDartString(length: len);
    tmp.write(str);
  }
  print("The original question is '$tmp'");

  final candidates = calloc.allocate<llama_cpp.llama_token_data>(
      nVocab * ffi.sizeOf<llama_cpp.llama_token_data>());
  final array = calloc.allocate<llama_cpp.llama_token_data_array>(
      ffi.sizeOf<llama_cpp.llama_token_data>());
  array.ref.data = candidates;
  array.ref.size = nVocab;
  array.ref.sorted = false;

  llama_cpp.llama_reset_timings(ctx);
  final eos = 2; // llama_cpp.llama_token_eos(ctx);

  var count = 0;
  while ((count = llama_cpp.llama_get_kv_cache_token_count(ctx)) < maxSize) {
    final ret = llama_cpp.llama_eval(ctx, tokenBuf, num, count);
    print("count=$count, num=$num, ret=$ret, maxSize=$maxSize");

    if (ret != 0) {
      print("Error: 'llama_eval' return $ret!");
      return;
    }
    final logits = llama_cpp.llama_get_logits(ctx);
    for (int i = 0; i < nVocab; i++) {
      candidates[i]
        ..id = i
        ..logit = logits[i]
        ..p = 0.0;
    }

    final tokenId = llama_cpp.llama_sample_token_greedy(ctx, array);
    if (tokenId == eos) { // llama_cpp.llama_token_eos(ctx.cast())
      print("-----------");
      break;
    }
    len = llama_cpp.llama_token_to_piece(model, tokenId, strBuf, bufSize);
    final str = _fromNative(strBuf, len);
    print("tokenId=$tokenId, str='$str'");

    tokenBuf[0] = tokenId;
    num = 1;
  }

  llama_cpp.llama_print_timings(ctx);

  calloc.free(array);
  calloc.free(candidates);

  calloc.free(strBuf);

  llama_cpp.llama_free(ctx);
  llama_cpp.llama_free_model(model);
  llama_cpp.llama_backend_free();
}
