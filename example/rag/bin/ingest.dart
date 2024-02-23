import 'dart:convert' show json;
import 'dart:io';

import 'package:rag/chroma.dart';
import 'package:rag/common.dart' as c;

List<ChromaDoc> _processDocuments(String dir, List<String> ignored) {
  return [];
}

void main(List<String> argv) async {
  final uri = Directory.current.uri.resolve('config.json');
  final config = json.decode(await File.fromUri(uri).readAsString());

  final chroma = await c.setupChroma(config);
  final all = await chroma.allItems;
  final ignored = all
      .map((d) => d.metadata?['source'] as String?)
      .whereType<String>()
      .toList(growable: false);
  final dir = config['source_dir'] ?? 'sources';

  final at = DateTime.now().millisecondsSinceEpoch;
  final docs = _processDocuments(dir, ignored);
  await chroma.add(docs);
  final cost = DateTime.now().millisecondsSinceEpoch - at;
  print("${docs.length} documents cost $cost ms.");
}
