import 'dart:ffi' as ffi;
import 'dart:io' show stderr, stdout, Platform;
import 'package:llama_cpp/native_llama_cpp.dart' as llama_cpp;
import 'package:llama_cpp/src/ffi.dart';

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

  final cStr = NativeString();
  final modelParams = llama_cpp.llama_model_default_params();
  final model = llama_cpp.llama_load_model_from_file(path.into(cStr), modelParams);
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
  prompt.into(cStr);
  final tokenBuf = TokenArray(size: tokenCapacity);
  tokenBuf.pavedBy(model, cStr);
  final tokenNum = tokenBuf.length;
  final kvReq = tokenNum + (nLen - tokenNum);
  print("\nn_len = $nLen, n_ctx = $ctxSize, n_kv_req = $kvReq, token_n = $tokenNum, len = ${cStr.length}");
  stderr.write("User prompt is:");
  for (var i = 0; i < tokenNum; i++) {
    final text = cStr.fromToken(model, tokenBuf[i]);
    stderr.write(text);
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
  final array = TokenDataArray(llama_cpp.llama_n_vocab(model));
  while (count <= nLen) {
    final nVocab = llama_cpp.llama_n_vocab(model);
    final logits = llama_cpp.llama_get_logits_ith(ctx, batch.n_tokens - 1);
    array.pavedBy(logits, nVocab);

    final tokenId = llama_cpp.llama_sample_token_greedy(ctx, array.pointer);
    if (tokenId == 0 || tokenId == 2 || count == nLen) {
      break;
    }
    final word = cStr.fromToken(model, tokenId);
    stdout.write(word);
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
