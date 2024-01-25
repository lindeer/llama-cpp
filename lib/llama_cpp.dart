import 'dart:io' show Platform, stdout;
import 'dart:isolate' show Isolate, ReceivePort, SendPort;
import 'dart:math' show min;

import 'src/native_llama.dart';

class LlamaCpp {
  final ReceivePort _recv;
  final Isolate _isolate;
  final SendPort _send;
  final Stream<String> _receiving;
  final bool verbose;

  const LlamaCpp._(this._recv, this._isolate, this._send, this._receiving, this.verbose);

  static Future<LlamaCpp> load(
    String path, {
    int nCtx = 512,
    int nGpuLayers = 0,
    int seed = 1234,
    int nThread = 8,
    bool verbose = true,
  }) async {
    final recv = ReceivePort('main.incoming');
    final params = LlamaParams(
      nCtx,
      nGpuLayers,
      seed,
      min(nThread, Platform.numberOfProcessors),
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

  Future<void> dispose() async {
    print("LlamaCpp.dispose: disposing native llama ...");
    _send.send(_finish);
    await _receiving.first;
    print("LlamaCpp.dispose: native llama disposed.");
    _recv.close();
    _isolate.kill();
  }

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
