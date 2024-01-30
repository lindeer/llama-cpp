import 'dart:ffi' as ffi;
import 'dart:math' show max, min;

import 'ffi.dart';
import '../native_llama_cpp.dart' as llama_cpp;
import 'sampling.dart';

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
    final model =
        llama_cpp.llama_load_model_from_file(path.into(cStr), modelParams);

    final t = params.nThread;
    final ctxParams = llama_cpp.llama_context_default_params()
      ..seed = params.seed
      ..n_ctx = params.nCtx
      ..n_threads = t
      ..n_threads_batch = t;
    final ctx = llama_cpp.llama_new_context_with_model(model, ctxParams);
    llama_cpp.llama_backend_init(false);
    final batch = llama_cpp.llama_batch_init(ctxParams.n_batch, 0, 1);
    llama_cpp.llama_set_n_threads(ctx, t, t);

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

    var num = 0;
    var code = 0;

    llama_cpp.llama_reset_timings(ctx);
    llama_cpp.llama_kv_cache_clear(ctx);
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

      num += batch.n_tokens;
      batch.n_tokens = 0;
      tokenBuf
        ..clear()
        ..add(tokenId);
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

  int _sampleSampling(SamplingContext ctxSampling, int idx,
      [bool isResampling = false]) {
    final params = ctxSampling.params;
    final model = llama_cpp.llama_get_model(ctx);
    final nVocab = llama_cpp.llama_n_vocab(model);
    final temp = params.temperature;
    final penaltyLastN = params.penaltyLastN < 0 ? params.nPrev
        : params.penaltyLastN;
    final penaltyRepeat = params.penaltyRepeat;
    final penaltyFrequency = params.penaltyFrequency;
    final penaltyPresent = params.penaltyPresent;
    final mirostat = params.mirostat;
    final mirostatTau = params.mirostatTau;
    final mirostatEta = params.mirostatEta;
    final penalizeNewline = params.penalizeNewline;

    final logits = llama_cpp.llama_get_logits_ith(ctx, idx);
    final logitBias = params.logitBias?.entries;
    logitBias?.forEach((e) {
      logits[e.key] += e.value;
    });
    array.pavedBy(logits, nVocab);
    // apply penalties
    final penaltyTokens = params.usePenaltyPromptTokens
        ? params.penaltyPromptTokens : ctxSampling.prev;
    final usedSize = min(penaltyTokens.length, penaltyLastN);
    if (usedSize > 0) {
      final nl = llama_cpp.llama_token_nl(model);
      final logit = logits[nl];
      llama_cpp.llama_sample_repetition_penalties(
        ctx,
        array.pointer,
        penaltyTokens,
        usedSize,
        penaltyRepeat,
        penaltyFrequency,
        penaltyPresent,
      );
      if (!penalizeNewline) {
        for (var i = 0; i < array.length; i++) {
          final data = array[i];
          if (data.id == nl) {
            final old = data.logit;
            array.setLogit(i, logit);
            final v = array[i].logit;
            print("$i: $old -> $v");
            break;
          }
        }
      }
    }

    final grammar = ctxSampling.grammar;
    if (isResampling && grammar != null) {
      llama_cpp.llama_sample_grammar(ctx, array.pointer, grammar);
    }
    var id = 0;
    if (temp < 0.0) {
      llama_cpp.llama_sample_softmax(ctx, array.pointer);
      id = array[0].id;
    } else if (temp == 0.0) {
      id = llama_cpp.llama_sample_token_greedy(ctx, array.pointer);
    } else {
      if (mirostat == 1) {
        const mirostatM = 100;
        llama_cpp.llama_sample_temp(ctx, array.pointer, temp);
        id = llama_cpp.llama_sample_token_mirostat(ctx, array.pointer,
            mirostatTau, mirostatEta, mirostatM, ctxSampling.mirostatMu);
      } else if (mirostat == 2) {
        llama_cpp.llama_sample_temp(ctx, array.pointer, temp);
        id = llama_cpp.llama_sample_token_mirostat_v2(ctx, array.pointer,
            mirostatTau, mirostatEta, ctxSampling.mirostatMu);
      } else {
        final minKeep = max(1, params.nProbs);
        _samplerQueue(params, nVocab, minKeep);
        id = llama_cpp.llama_sample_token(ctx, array.pointer);
      }
      print("sampled token($mirostat): ${'$id'.padLeft(5)}: "
          "'${cStr.fromToken(model, id)}'");
    }

    if (grammar != null && !isResampling) {
    }

    return id;
  }

  void _samplerQueue(SamplingParams params, int capacity, int minKeep) {
    final topK = params.topK <= 0 ? capacity : params.topK;
    for (final i in params.samplersSequence.codeUnits) {
      switch (i) {
        case _kChar:
          llama_cpp.llama_sample_top_k(ctx, array.pointer, topK, minKeep);
          break;
        case _fChar:
          llama_cpp.llama_sample_tail_free(ctx, array.pointer,
              params.tfsZ, minKeep);
          break;
        case _yChar:
          llama_cpp.llama_sample_typical(ctx, array.pointer,
              params.typicalP, minKeep);
          break;
        case _pChar:
          llama_cpp.llama_sample_top_p(ctx, array.pointer,
              params.topP, minKeep);
          break;
        case _nChar:
          llama_cpp.llama_sample_min_p(ctx, array.pointer,
              params.minP, minKeep);
          break;
        case _tChar:
          llama_cpp.llama_sample_temp(ctx, array.pointer, params.temperature);
          break;
        default:
          break;
      }
    }
  }
}

const _fChar = 0x66;
const _kChar = 0x6b;
const _yChar = 0x79;
const _pChar = 0x70;
const _nChar = 0x6e;
const _tChar = 0x74;
