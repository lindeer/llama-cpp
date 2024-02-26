import 'dart:convert' show json;
import 'dart:io' show stdin, stdout;
import 'package:llama_cpp/llama_cpp.dart';
import 'package:rag/chroma.dart';
import 'package:rag/common.dart' as c;

String _makePrompt(String question, List<ChromaItem> items) {
  return '''根据以下信息：

${items.map((e) => e.doc).join("\n\n")}

请回答：$question''';
}

String? get _readLine {
  stdout.write('> ');
  return stdin.readLineSync()?.trim();
}

void main(List<String> argv) async {
  final config = c.appConfig;
  final chroma = await c.setupChroma(config);

  final path = config['gpt_model'] as String;
  final gpt = await LlamaCpp.load(path, verbose: false);
  late String question;
  while ((question = (_readLine ?? 'exit')) != 'exit') {
    if (question.isEmpty) {
      continue;
    }
    final items = await chroma.query(question, nResults: 2);
    final prompt = _makePrompt(question, items);
    final answer = gpt.answer(prompt);
    stdout.write('< ');
    await for (final str in answer) {
      stdout.write(str);
    }
    stdout.writeln();
  }
  await gpt.dispose();
  chroma.dispose();
}
