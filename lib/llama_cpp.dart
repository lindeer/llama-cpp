import 'dart:io' show Platform, stdout;
import 'dart:isolate' show Isolate, ReceivePort, SendPort;

import 'src/native_llama.dart';

int get _physicalCores {
  final n = Platform.numberOfProcessors;
  return n > 4 ? n ~/ 2 : n > 0 ? n : 4;
}

/// A brief overview of inter-ops among main classes:
///
/// +----------------+---------------------+--------------------+
/// |  main Isolate  |    llama Isolate    |    native world    |
/// |----------------|---------------------|--------------------|
/// |   LlamaCpp     |     NativeLlama     |    llama_cpp       |
/// |                |                     |                    |
/// |          send --> incoming          -->       +           |
/// |                |                     |        |           |
/// |                |                    ffi       |           |
/// |                |                     |        |           |
/// |     receiving <-- outgoing          <--       +           |
/// |                |                     |                    |
/// +---------------+---------------------+---------------------+
class LlamaCpp {
  final ReceivePort _recv;
  final Isolate _isolate;
  final SendPort _send;
  final Stream<String> _receiving;
  final bool verbose;

  const LlamaCpp._(this._recv, this._isolate, this._send, this._receiving, this.verbose);

  /// Async create LlamaCpp by given params.
  static Future<LlamaCpp> load(
    String path, {
    int seed = -1,
    int nThread = 0,
    int nThreadBatch = -1,
    int nPredict = -1,
    int nCtx = 512,
    int nBatch = 512,
    int nKeep = 0,
    int? nGpuLayers,
    int? mainGpu,
    bool numa = false,
    bool verbose = true,
  }) async {
    final recv = ReceivePort('main.incoming');
    final params = LlamaParams(
      seed,
      nThread > 0 ? nThread : _physicalCores,
      nThreadBatch,
      nPredict,
      nCtx,
      nBatch,
      nGpuLayers,
      mainGpu,
      numa,
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

  static const _finish = NativeLLama.closeTag;

  /// Notify isolate to free native resources, after that, finish this isolate.
  Future<void> dispose() async {
    print("LlamaCpp.dispose: disposing native llama ...");
    _send.send(_finish);
    await _receiving.first;
    print("LlamaCpp.dispose: native llama disposed.");
    _recv.close();
    _isolate.kill();
  }

  /// Generate text stream by given prompt.
  /// @question The prompt passed by user who want model to generate an answer.
  Stream<String> answer(String question) async* {
    if (verbose) {
      stdout.writeln("<<<<<<<<<<<<<<<");
      stdout.writeln("$question\n---------------");
    }
    _send.send(question);
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
    final questions = incoming.cast<String>();
    await for (final q in questions) {
      if (q == _finish) {
        print("Isolate received '$q', start closing ...");
        break;
      }
      final s = llama.generate(q);
      await for (final str in s) {
        outgoing.send(str);
        if (str == NativeLLama.engTag) {
          break;
        }
      }
    }
    llama.dispose();
    outgoing.send(_finish);
  }
}
