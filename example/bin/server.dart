import 'dart:convert' show json, utf8;

import 'package:shelf_router/shelf_router.dart' show Router;
import 'package:shelf/shelf.dart' show Request, Response;
import 'package:shelf/shelf_io.dart' as io;
import 'package:llama_cpp/llama_cpp.dart' show LlamaCpp;

const _defaultPort = 8080;
void main(List<String> argv) async {
  if (argv.isEmpty) {
    print("usage: dart server.dart MODEL_PATH [PROMPT]");
    return;
  }
  final path = argv[0];
  final port = argv.length > 1 ? (int.tryParse(argv[1]) ?? _defaultPort) : _defaultPort;
  final ai = await LlamaCpp.load(path);

  final app = Router();
  app.post('/generate', (Request request) async {
    final body = await request.readAsString();
    final obj = json.decode(body);
    final prompt = obj['prompt'];
    final response = ai.answer(prompt);
    return Response.ok(
      response.transform(utf8.encoder),
      headers: {
        'Content-Type': 'application/octet-stream; charset=utf-8',
      },
    );
  });
  final server = await io.serve(app, 'localhost', port);
  print('Serving at http://${server.address.host}:${server.port}');
}
