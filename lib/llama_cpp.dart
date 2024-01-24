import 'dart:isolate' show Isolate, ReceivePort, SendPort;

import 'src/native_llama.dart';

class LlamaCpp {
  final ReceivePort _recv;
  final Isolate _isolate;
  final SendPort _send;
  final Stream<String> _receiving;

  const LlamaCpp._(this._recv, this._isolate, this._send, this._receiving);

  static Future<LlamaCpp> load(
    String path, {
    int nCtx = 512,
    int nGpuLayers = 0,
    int seed = 1234,
    int nThread = 8,
  }) async {
    final recv = ReceivePort('main.incoming');
    final params = LlamaParams(nCtx, nGpuLayers, seed, nThread);
    final isolate = await Isolate.spawn<(SendPort, String, LlamaParams)>(
      _llamaIsolate,
      (recv.sendPort, path, params),
      errorsAreFatal: true,
      debugName: '_llamaIsolate',
    );
    final receiving = recv.asBroadcastStream();
    final send = (await receiving.first) as SendPort;
    return LlamaCpp._(recv, isolate, send, receiving.cast<String>());
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
    _send.send(question);
    await for (final msg in _receiving) {
      if (msg == NativeLLama.engTag) {
        break;
      } else {
        yield msg;
      }
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
