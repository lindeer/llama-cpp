import 'dart:convert' show utf8;
import 'dart:io' show HttpServer;

import 'package:llama_cpp/llama_cpp.dart' show LlamaCpp;

const _defaultPort = 8080;
void main(List<String> argv) async {
  if (argv.isEmpty) {
    print("usage: dart server.dart MODEL_PATH [PORT]");
    return;
  }
  final path = argv[0];
  final port = (argv.length > 1 ? int.tryParse(argv[1]) : null) ?? _defaultPort;
  final ai = await LlamaCpp.load(path);

  final server = await HttpServer.bind('localhost', port);
  print('Serving at http://${server.address.host}:${server.port}');
  await for (final request in server) {
    final body = await request
        .map((e) => List<int>.from(e))
        .transform(utf8.decoder)
        .join();
    final response = request.response;
    response.headers
      ..set('Content-Type', 'application/octet-stream; charset=utf-8')
      ..add("Transfer-Encoding", "chunked");
    response.bufferOutput = false;
    final answer = ai.answerWith(body);
    // curl should run with `--no-buffer` param
    await response.addStream(answer.transform(utf8.encoder));
    await response.close();
  }
}
