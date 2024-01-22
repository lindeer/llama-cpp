import 'dart:convert' show utf8;
import 'dart:ffi' as ffi;
import 'dart:io' show stderr, stdout, Platform;
import 'package:ffi/ffi.dart';
import 'package:llama_cpp/native_llama_cpp.dart' as llama_cpp;

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


void _addLlamaBatch(
    llama_cpp.llama_batch batch,
    int id,
    int pos,
    List<int> seqIds,
    bool logits,
) {
  final n = batch.n_tokens;
  final m = seqIds.length;
  batch.token[n] = id;
  batch.pos[n] = pos;
  batch.n_seq_id[n] = m;
  for (var i = 0; i < m; i++) {
    batch.seq_id[n][i] = seqIds[i];
  }
  batch.logits[n] = logits ? 1 : 0;

  batch.n_tokens++;
}

int main(List<String> argv) {
  if (argv.isEmpty || argv[0].startsWith('-')) {
    print("usage: ${Platform.script.path} MODEL_PATH [PROMPT]");
    return 1;
  }
  final path = argv[0];
  final prompt = argv.length > 1 ? argv[1] : 'Hello my name is';
  // total length of the sequence including the prompt
  const nLen = 32;
  llama_cpp.llama_backend_init(false);

  const bufSize = 1024;
  final strBuf = calloc.allocate<ffi.Char>(bufSize * ffi.sizeOf<ffi.Char>());
  var strLen = _toNative(path, strBuf, bufSize);

  final modelParams = llama_cpp.llama_model_default_params();
  final model = llama_cpp.llama_load_model_from_file(strBuf, modelParams);
  final ctxParams = llama_cpp.llama_context_default_params()
    ..seed = 1234
    ..n_ctx = 1024
    ..n_threads = 4
    ..n_threads_batch = 4;
  final ctx = llama_cpp.llama_new_context_with_model(model, ctxParams);
  if (ctx.address == 0) {

  }

  final ctxSize = llama_cpp.llama_n_ctx(ctx);
  final tokenCapacity = prompt.length + 1;
  strLen = _toNative(prompt, strBuf, bufSize);
  final tokenBuf = calloc.allocate<llama_cpp.llama_token>(tokenCapacity * ffi.sizeOf<llama_cpp.llama_token>());
  var tokenNum = llama_cpp.llama_tokenize(
    model,
    strBuf, strLen,
    tokenBuf, tokenCapacity,
    false, true,
  );
  if (tokenNum < 0) {
    stderr.writeln("':llama_tokenize' return err: $tokenNum");
    return 1;
  }
  final kvReq = tokenNum + (nLen - tokenNum);
  print("\nn_len = $nLen, n_ctx = $ctxSize, n_kv_req = $kvReq, token_n = $tokenNum, len = $strLen");
  stderr.write("User prompt is:");
  for (var i = 0; i < tokenNum; i++) {
    strLen = llama_cpp.llama_token_to_piece(model, tokenBuf[i], strBuf, bufSize);
    if (strLen > 0) {
      stderr.write(_fromNative(strBuf, strLen));
    }
  }
  stderr.writeln();
  stderr.flush();

  // create a llama_batch with size 512
  // we use this object to submit token data for decoding
  final batch = llama_cpp.llama_batch_init(512, 0, 1);
  // evaluate the initial prompt
  for (var i = 0; i < tokenNum; i++) {
    _addLlamaBatch(batch, tokenBuf[i], i, [0], false);
  }
  batch.logits[batch.n_tokens - 1] = 1;

  if (llama_cpp.llama_decode(ctx, batch) != 0) {
    return 1;
  }

  llama_cpp.llama_reset_timings(ctx);
  var count = batch.n_tokens;
  var dataCapacity = llama_cpp.llama_n_vocab(model);
  var candidates = calloc.allocate<llama_cpp.llama_token_data>(dataCapacity * ffi.sizeOf<llama_cpp.llama_token_data>());
  final array = calloc.allocate<llama_cpp.llama_token_data_array>(ffi.sizeOf<llama_cpp.llama_token_data>());
  while (count <= nLen) {
    final nVocab = llama_cpp.llama_n_vocab(model);
    final logits = llama_cpp.llama_get_logits_ith(ctx, batch.n_tokens - 1);
    if (dataCapacity < nVocab) {
      calloc.free(candidates);
      dataCapacity = nVocab;
      candidates = calloc.allocate<llama_cpp.llama_token_data>(dataCapacity * ffi.sizeOf<llama_cpp.llama_token_data>());
    }
    for (var id = 0; id < nVocab; id++) {
      candidates[id]
          ..id = id
          ..logit = logits[id]
          ..p = 0;
    }
    array.ref
      ..data = candidates
      ..size = nVocab
      ..sorted = false;

    final tokenId = llama_cpp.llama_sample_token_greedy(ctx, array);
    if (tokenId == 0 || tokenId == 2 || count == nLen) {
      break;
    }
    strLen = llama_cpp.llama_token_to_piece(model, tokenId, strBuf, bufSize);
    stdout.write(_fromNative(strBuf, strLen));
    // `stdout.flush()` cause 'Bad state: StreamSink is bound to a stream' error in Dart 3.1.5
    // stdout.flush();

    // prepare the next batch
    batch.n_tokens = 0;
    // push this new token for next evaluation
    _addLlamaBatch(batch, tokenId, count, [0], true);

    count++;

    // evaluate the current batch with the transformer model
    if (llama_cpp.llama_decode(ctx, batch) != 0) {
      return 2;
    }
  }

  calloc.free(candidates);
  calloc.free(array);

  calloc.free(tokenBuf);
  calloc.free(strBuf);

  llama_cpp.llama_print_timings(ctx);

  llama_cpp.llama_batch_free(batch);
  llama_cpp.llama_free(ctx);
  llama_cpp.llama_free_model(model);
  llama_cpp.llama_backend_free();

  return 0;
}
