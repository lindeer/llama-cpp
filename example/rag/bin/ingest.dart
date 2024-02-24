import 'dart:io';

import 'package:rag/chroma.dart';
import 'package:rag/common.dart' as c;

const chunkSize = 500;
const overlapSize = 10;

List<ChromaDoc> _processDocuments(String dir, List<String> ignored) {
  bool isValidFile(String path) {
    return FileSystemEntity.isFileSync(path) && !ignored.contains(path);
  }

  final docs = Directory(dir)
      .listSync(recursive: true)
      .where((f) => isValidFile(f.path))
      .expand((e) {
    final file = File.fromUri(e.uri);
    final lines = file.readAsLinesSync().where((l) => l.trim().isNotEmpty);
    return _processLines(e.uri, lines);
  }).toList(growable: false);
  return docs;
}

List<ChromaDoc> _processLines(Uri file, Iterable<String> lines) {
  final result = <ChromaDoc>[];
  final filepath = file.path;
  for (final line in lines) {
    final len = line.length;
    if (len > chunkSize) {
      for (var i = 0; i < len; i += chunkSize) {
        final enough = len - i > chunkSize;
        final str = enough ? line.substring(i, chunkSize) : line.substring(i);
        final delta = i > overlapSize ? line.substring(i - overlapSize, i) : '';
        result.add(ChromaDoc('$delta$str', filepath));
      }
    } else {
      result.add(ChromaDoc(line, filepath));
    }
  }
  return result;
}

void main(List<String> argv) async {
  final config = c.appConfig;
  final chroma = await c.setupChroma(config);
  final all = await chroma.allItems;
  final ignored = all
      .map((d) => d.metadata?['source'] as String?)
      .whereType<String>()
      .toList(growable: false);
  final dir = config['source_dir'] ?? 'sources';

  final docs = _processDocuments(dir, ignored);
  if (docs.isNotEmpty) {
    final at = DateTime.now().millisecondsSinceEpoch;
    await chroma.add(docs);
    final cost = DateTime.now().millisecondsSinceEpoch - at;
    print("Save [${docs.length}] documents cost $cost ms.");
  }
  chroma.dispose();
}
