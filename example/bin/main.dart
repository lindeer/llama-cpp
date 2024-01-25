import 'dart:io';

import 'package:llama_cpp/llama_cpp.dart';

Future<int> main(List<String> argv) async {

  if (argv.isEmpty || argv[0].startsWith('-')) {
    print("usage: ${Platform.script.path} MODEL_PATH [PROMPT]");
    return 1;
  }
  final path = argv[0];
  final prompt = argv.length > 1 ? argv[1] : 'Hello my name is';
  final llama = await LlamaCpp.load(path, verbose: false);

  await for (final s in llama.answer(prompt)) {
    stdout.write(s);
  }
  stdout.writeln();

  await llama.dispose();
  return 0;
}
