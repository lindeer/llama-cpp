import 'dart:ffi' as ffi;
import 'dart:math';

import 'package:ffi/ffi.dart';

import 'lib_llama_cpp.dart' as llama_cpp;

extension _FloatEx on double {
  String get str => toStringAsFixed(3);
}

const fChar = 0x66;
const kChar = 0x6b;
const yChar = 0x79;
const pChar = 0x70;
const mChar = 0x6d;
const tChar = 0x74;

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
  final List<int>? penaltyPromptTokens;
  final bool usePenaltyPromptTokens;

  const SamplingParams({
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
    this.penaltyPromptTokens,
    this.usePenaltyPromptTokens = false,
  });

  String get samplingString =>
      "\trepeat_last_n = $penaltyLastN, repeat_penalty = ${penaltyRepeat.str}, frequency_penalty = "
      "${penaltyFrequency.str}, presence_penalty = ${penaltyPresent.str}\n"
      "\ttop_k = $topK, tfs_z = ${tfsZ.str}, top_p = ${topP.str}, min_p = ${minP.str}, "
      "typical_p = ${typicalP.str}, temp = ${temperature.str}\n"
      "\tmirostat = $mirostat, mirostat_lr = ${mirostatEta.str}, mirostat_ent = ${mirostatTau.str}";

  String get samplingOrder {
    final buf = StringBuffer('CFG -> Penalties ');
    if (mirostat == 0) {
      for (final c in samplersSequence.codeUnits) {
        final seq = _samplersSeq[c];
        if (seq != null) {
          buf.write(seq);
        }
      }
    } else {
      buf.write('-> mirostat ');
    }
    return buf.toString();
  }

  @override
  String toString() => samplingString;
}

const _samplersSeq = {
  fChar: '-> tfs_z ',
  kChar: '-> top_k ',
  yChar: '-> typical_p ',
  pChar: '-> top_p ',
  mChar: '-> min_p ',
  tChar: '-> temp ',
};

class SamplingContext {
  final SamplingParams params;
  final ffi.Pointer<ffi.Float> mirostatMu;
  final ffi.Pointer<llama_cpp.llama_grammar>? grammar;
  final ffi.Pointer<llama_cpp.llama_token> _prev;
  final int prevSize;
  final int usedSize;

  SamplingContext._(
    this.params,
    this.mirostatMu,
    this.grammar,
    this._prev,
    this.prevSize,
    this.usedSize,
  );

  factory SamplingContext.from(SamplingParams params) {
    ffi.Pointer<llama_cpp.llama_grammar>? grammar;
    if (params.grammar != null) {}
    final mu = calloc.allocate<ffi.Float>(ffi.sizeOf<ffi.Float>());
    final (p, len) = _createNativeTokens(params);
    final lastN = params.penaltyLastN;
    final penaltyLastN = lastN < 0 ? params.nPrev : lastN;
    final usedSize = min(len, penaltyLastN);

    return SamplingContext._(
      params,
      mu,
      grammar,
      p,
      len,
      usedSize,
    );
  }

  void free() {
    final g = grammar;
    if (g != null) {
      llama_cpp.llama_grammar_free(g);
    }
    calloc.free(mirostatMu);
    calloc.free(_prev);
  }

  void reset() {
    final g = grammar;
    if (g != null) {
      llama_cpp.llama_grammar_free(g);
    }
    for (var i = 0; i < prevSize; i++) {
      _prev[i] = 0;
    }
  }

  int get lastSampledToken => _prev[prevSize - 1];

  ffi.Pointer<llama_cpp.llama_token> get penaltyPointer =>
      _prev + prevSize - usedSize;

  /// A ring buffer to append a list of tokens
  void acceptSampling(
    ffi.Pointer<llama_cpp.llama_context> ctx,
    List<int> ids,
    bool applyGrammar,
  ) {
    final n = min(ids.length, prevSize);
    for (var i = 0; i < prevSize - n; i++) {
      (_prev + i).value = _prev[i + n];
    }
    for (var i = 0; i < n; i++) {
      (_prev + i + prevSize - n).value = ids[i];
    }

    if (grammar != null && applyGrammar) {
      // TODO: consider grammar
    }
  }
}

(ffi.Pointer<llama_cpp.llama_token>, int) _createNativeTokens(
    SamplingParams params) {
  final promptTokens = params.penaltyPromptTokens ?? [];
  final tokens = params.usePenaltyPromptTokens && promptTokens.isNotEmpty
      ? promptTokens
      : List.filled(params.nPrev, 0);
  final p = calloc.allocate<llama_cpp.llama_token>(
      ffi.sizeOf<llama_cpp.llama_token>() * tokens.length);
  for (var i = 0; i < tokens.length; i++) {
    p[i] = tokens[i];
  }
  return (p, tokens.length);
}
