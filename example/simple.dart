import 'dart:ffi' as ffi;
import 'dart:io' show stderr, stdout, Platform;

import 'package:llama_cpp/src/common.dart' as c;
import 'package:llama_cpp/src/ffi.dart';
import 'package:llama_cpp/src/lib_llama_cpp.dart' as llama_cpp;
import 'package:llama_cpp/src/llama_params.dart';

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

  final cStr = NativeString();
  path.into(cStr);
  final (model, ctx) = c.loadModel(
    cStr,
    LlamaParams(
      seed: 1234,
      nCtx: 1024,
      nThread: 4,
      nThreadBatch: 4,
    ),
  );

  final ctxSize = llama_cpp.llama_n_ctx(ctx);
  final tokenCapacity = prompt.length + 1;
  prompt.into(cStr);
  final tokenBuf = TokenArray(size: tokenCapacity);
  tokenBuf.pavedBy(model, cStr);
  final tokenNum = tokenBuf.length;
  final kvReq = tokenNum + (nLen - tokenNum);
  print("\nn_len = $nLen, n_ctx = $ctxSize, n_kv_req = $kvReq, "
      "token_n = $tokenNum, len = ${cStr.length}");
  stderr.write("User prompt is:");
  for (var i = 0; i < tokenNum; i++) {
    final text = cStr.tokenString(model, tokenBuf[i]);
    stderr.write(text);
  }
  stderr.writeln();
  stderr.flush();

  // create a llama_batch with size 512
  // we use this object to submit token data for decoding
  final batch = llama_cpp.llama_batch_init(512, 0, 1);
  // evaluate the initial prompt
  c.addBatchSeq(batch, tokenBuf.toList(), 0);
  batch.logits[batch.n_tokens - 1] = 1;

  if (llama_cpp.llama_decode(ctx, batch) != 0) {
    return 1;
  }

  llama_cpp.llama_reset_timings(ctx);
  var count = batch.n_tokens;
  final nVocab = llama_cpp.llama_n_vocab(model);
  final array = TokenDataArray(nVocab);
  final eosToken = llama_cpp.llama_token_eos(model);
  while (count <= nLen) {
    final logits = llama_cpp.llama_get_logits_ith(ctx, batch.n_tokens - 1);
    array.pavedBy(logits, nVocab);

    final tokenId = llama_cpp.llama_sample_token_greedy(ctx, array.pointer);
    if (tokenId == eosToken || count == nLen) {
      break;
    }
    final word = cStr.tokenString(model, tokenId);
    stdout.write(word);
    // `stdout.flush()` cause 'Bad state: StreamSink is bound to a stream' error in Dart 3.1.5
    // stdout.flush();

    // prepare the next batch
    batch.n_tokens = 0;
    // push this new token for next evaluation
    c.addBatchSingle(batch, tokenId, count, true);

    count++;

    // evaluate the current batch with the transformer model
    if (llama_cpp.llama_decode(ctx, batch) != 0) {
      return 2;
    }
  }

  array.dispose();
  tokenBuf.dispose();
  cStr.dispose();

  llama_cpp.llama_print_timings(ctx);

  llama_cpp.llama_batch_free(batch);
  llama_cpp.llama_free(ctx);
  llama_cpp.llama_free_model(model);
  llama_cpp.llama_backend_free();

  return 0;
}
