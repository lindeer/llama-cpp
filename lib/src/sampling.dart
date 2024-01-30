import 'dart:ffi' as ffi;
import 'dart:math';

import 'package:ffi/ffi.dart';

import '../native_llama_cpp.dart' as llama_cpp;
import 'ffi.dart' show NativeString;

extension _FloatEx on double {
  String get str => toStringAsFixed(3);
}

class SamplingParams {
  /// number of previous tokens to remember
  final int nPrev;

  /// if greater than 0, output the probabilities of top n_probs tokens.
  final int nProbs;

  /// <= 0 to use vocab size
  final int topK;
  final double topP;
  final double minP;
  final double tfsZ;
  final double typicalP;
  final double temperature;
  final int penaltyLastN;
  final double penaltyRepeat;
  final double penaltyFrequency;
  final double penaltyPresent;
  final int mirostat;
  final double mirostatTau;
  final double mirostatEta;
  final bool penalizeNewline;
  final String samplersSequence;

  /// optional BNF-like grammar to constrain sampling
  final String? grammar;

  /// string to help guidance
  final String? cfgNegativePrompt;

  /// how strong is guidance
  final double cfgScale;
  final Map<int, double>? logitBias;
  final penaltyPromptTokens = <int>[];
  final bool usePenaltyPromptTokens;

  SamplingParams({
    this.nPrev = 64,
    this.nProbs = 0,
    this.topK = 40,
    this.topP = 0.95,
    this.minP = 0.05,
    this.tfsZ = 1.00,
    this.typicalP = 1.0,
    this.temperature = 0.80,
    this.penaltyLastN = 64,
    this.penaltyRepeat = 1.10,
    this.penaltyFrequency = 0.00,
    this.penaltyPresent = 0.00,
    this.mirostat = 0,
    this.mirostatTau = 5.00,
    this.mirostatEta = 0.10,
    this.penalizeNewline = true,
    // top_k, tail_free, typical_p, top_p, min_p, temp
    this.samplersSequence = "kfypmt",
    this.grammar,
    this.cfgNegativePrompt,
    this.cfgScale = 1.0,
    this.logitBias,
    this.usePenaltyPromptTokens = false,
  });

  String get samplingString =>
      "\trepeat_last_n = $penaltyLastN, repeat_penalty = ${penaltyRepeat.str}, frequency_penalty = "
      "${penaltyFrequency.str}, presence_penalty = ${penaltyPresent.str}\n"
      "\ttop_k = $topK, tfs_z = ${tfsZ.str}, top_p = ${topP.str}, min_p = ${minP.str}, "
      "typical_p = ${typicalP.str}, temp = ${temperature.str}\n"
      "\tmirostat = $mirostat, mirostat_lr = ${mirostatEta.str}, mirostat_ent = ${mirostatTau.str}";
}

class SamplingContext {
  final SamplingParams params;
  final ffi.Pointer<ffi.Float> mirostatMu;
  final ffi.Pointer<llama_cpp.llama_grammar>? grammar;
  final List<int> prev;
  final List<llama_cpp.llama_token_data> cur;

  SamplingContext._(
    this.params,
    this.mirostatMu,
    this.grammar,
    this.prev,
    this.cur,
  );

  factory SamplingContext.from(SamplingParams params) {
    ffi.Pointer<llama_cpp.llama_grammar>? grammar;
    if (params.grammar != null) {}
    final mu = calloc.allocate<ffi.Float>(ffi.sizeOf<ffi.Float>());
    return SamplingContext._(
      params,
      mu,
      grammar,
      <int>[],
      <llama_cpp.llama_token_data>[],
    );
  }

  void free() {
    final g = grammar;
    if (g != null) {
      llama_cpp.llama_grammar_free(g);
    }
    calloc.free(mirostatMu);
  }

  void reset() {
    final g = grammar;
    if (g != null) {
      llama_cpp.llama_grammar_free(g);
    }
    prev.fillRange(0, prev.length, 0);
    cur.clear();
  }

  int get lastSampledToken => prev.last;

  String lastSampledTokenString(
      ffi.Pointer<llama_cpp.llama_context> ctx, int n, NativeString cStr) {
    final size = prev.length;
    n = min(n, size);
    final model = llama_cpp.llama_get_model(ctx);
    final result =
        prev.sublist(size - n).map((t) => cStr.fromToken(model, t)).join('');
    return result;
  }
}
