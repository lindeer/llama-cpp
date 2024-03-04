import 'dart:convert' show json, utf8;
import 'dart:io' show stdout;
import 'dart:isolate' show Isolate, ReceivePort, SendPort;

import 'src/llama_params.dart';
import 'src/native_llama.dart';

/// A brief overview of inter-ops among main classes:
///
/// ```
/// +----------------+---------------------+--------------------+
/// |  main Isolate  |    llama Isolate    |    native world    |
/// +----------------+---------------------+--------------------+
/// |   LlamaCpp     |     NativeLlama     |    llama_cpp       |
/// |                |                     |                    |
/// |          send --> incoming          -->       +           |
/// |                |                     |        |           |
/// |                |                    ffi       |           |
/// |                |                     |        |           |
/// |     receiving <-- outgoing          <--       +           |
/// |                |                     |                    |
/// +---------------+---------------------+---------------------+
/// ```
class LlamaCpp {
  final ReceivePort _recv;
  final Isolate _isolate;
  final SendPort _send;
  final Stream<String> _receiving;
  final bool verbose;

  const LlamaCpp._(
    this._recv,
    this._isolate,
    this._send,
    this._receiving,
    this.verbose,
  );

  /// Async create LlamaCpp by given params.
  static Future<LlamaCpp> load(
    String path, {
    int? seed,
    int? nThread,
    int? nThreadBatch,
    int? nPredict,
    int? nCtx,
    int? nBatch,
    int? nKeep,
    int? nGpuLayers,
    int? mainGpu,
    int numa = 0,
    bool verbose = true,
  }) async {
    final recv = ReceivePort('main.incoming');
    final params = LlamaParams(
      seed: seed,
      nThread: nThread,
      nThreadBatch: nThreadBatch,
      nPredict: nPredict,
      nCtx: nCtx,
      nBatch: nBatch,
      nGpuLayers: nGpuLayers,
      mainGpu: mainGpu,
      numa: numa,
    );
    final isolate = await Isolate.spawn<(SendPort, String, LlamaParams)>(
      _llamaIsolate,
      (recv.sendPort, path, params),
      errorsAreFatal: true,
      debugName: '_llamaIsolate',
    );
    final receiving = recv.asBroadcastStream();
    final send = (await receiving.first) as SendPort;
    return LlamaCpp._(recv, isolate, send, receiving.cast<String>(), verbose);
  }

  static const _finish = <String, dynamic>{'cmd': NativeLLama.closeTag};

  /// Notify isolate to free native resources, after that, finish this isolate.
  Future<void> dispose() async {
    print("LlamaCpp.dispose: disposing native llama ...");
    _send.send(_finish);
    await _receiving.first;
    print("LlamaCpp.dispose: native llama disposed.");
    _recv.close();
    _isolate.kill();
  }

  /// Generate text stream by given params.
  /// [params] json string with params, e.g.:
  /// ai.answerWith({
  ///   "prompt": "my question is",
  ///   "min_p": 20,
  /// });
  Stream<String> answerWith(String params) {
    final request = json.decode(params);
    if ((request['prompt'] ?? '').isEmpty) {
      throw Exception("Json body without 'prompt'!");
    }
    return _requestAnswer(request);
  }

  /// Generate text stream by given prompt.
  /// @question The prompt passed by user who want model to generate an answer.
  Stream<String> answer(
    String question, {
    int? nPrev,
    int? nProbs,
    int? topK,
    double? topP,
    double? minP,
    double? tfsZ,
    double? typicalP,
    double? temperature,
    int? penaltyLastN,
    double? penaltyRepeat,
    double? penaltyFrequency,
    double? penaltyPresent,
    int? mirostat,
    double? mirostatTau,
    double? mirostatEta,
    bool? penalizeNewline,
    String? samplersSequence,
  }) {
    final request = {
      'prompt': question,
      if (nPrev != null) 'n_prev': nPrev,
      if (nProbs != null) 'n_probs': nProbs,
      if (topK != null) 'top_k': topK,
      if (topP != null) 'top_p': topP,
      if (minP != null) 'min_p': minP,
      if (tfsZ != null) 'tfs_z': tfsZ,
      if (typicalP != null) 'typical_p': typicalP,
      if (temperature != null) 'temperature': temperature,
      if (penaltyLastN != null) 'penalty_last_n': penaltyLastN,
      if (penaltyRepeat != null) 'penalty_repeat': penaltyRepeat,
      if (penaltyFrequency != null) 'penalty_frequency': penaltyFrequency,
      if (penaltyPresent != null) 'penalty_present': penaltyPresent,
      if (mirostat != null) 'mirostat': mirostat,
      if (mirostatTau != null) 'mirostat_tau': mirostatTau,
      if (mirostatEta != null) 'mirostat_eta': mirostatEta,
      if (penalizeNewline != null) 'penalize_newline': penalizeNewline,
      if (samplersSequence != null) 'samplers_sequence': samplersSequence,
    };

    return _requestAnswer(request);
  }

  Stream<String> _requestAnswer(Map<String, dynamic> request) async* {
    if (verbose) {
      stdout.writeln("<<<<<<<<<<<<<<<");
      stdout.writeln("$request\n---------------");
    }
    _send.send(request);
    await for (final msg in _receiving) {
      if (msg == NativeLLama.engTag) {
        break;
      } else {
        yield msg;
        if (verbose) {
          stdout.write(msg);
        }
      }
    }
    if (verbose) {
      stdout.writeln("\n>>>>>>>>>>>>>>>");
    }
  }

  // run in the isolate, relative main.
  static _llamaIsolate((SendPort, String, LlamaParams) r) async {
    final (outgoing, path, params) = r;
    final incoming = ReceivePort('_runIsolate.incoming');

    final llama = NativeLLama(path, params);
    outgoing.send(incoming.sendPort);
    final requests = incoming.cast<Map<String, dynamic>>();
    await for (final r in requests) {
      if (r['cmd'] == NativeLLama.closeTag) {
        print("Isolate received '$r', start closing ...");
        break;
      }
      final params = r;
      final prompt = params['prompt'] as String;
      final rawStream = llama.generate(
        prompt,
        nPrev: params['n_prev'],
        nProbs: params['n_probs'],
        topK: params['top_k'],
        topP: params['top_p'],
        minP: params['min_p'],
        tfsZ: params['tfs_z'],
        typicalP: params['typical_p'],
        temperature: params['temperature'],
        penaltyLastN: params['penalty_last_n'],
        penaltyRepeat: params['penalty_repeat'],
        penaltyFrequency: params['penalty_frequency'],
        penaltyPresent: params['penalty_present'],
        mirostat: params['mirostat'],
        mirostatTau: params['mirostat_tau'],
        mirostatEta: params['mirostat_eta'],
        penalizeNewline: params['penalize_newline'],
        samplersSequence: params['samplers_sequence'],
      );
      final s = rawStream.transform(utf8.decoder);
      await for (final str in s) {
        outgoing.send(str);
        if (str == NativeLLama.engTag) {
          break;
        }
      }
    }
    llama.dispose();
    outgoing.send(NativeLLama.closeTag);
  }
}
