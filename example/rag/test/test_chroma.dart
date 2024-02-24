import 'dart:io';

import 'package:rag/common.dart' as c;
import 'package:rag/chroma.dart';
import 'package:test/test.dart';

void main() async {
  // start a chroma server:
  // `uvicorn chromadb.app:app --reload --workers 1 --host 0.0.0.0 --port 8000`

  final config = {
    'embedding_model': Platform.environment['EMBEDDING_MODEL_PATH'] ??
        (throw Exception("Model path 'EMBEDDING_MODEL_PATH' not specified!")),
    'db_server_url': 'http://0.0.0.0:8000',
    'database': 'ecr',
    'collection': 'retail',
  };
  final chroma = await c.setupChroma(config);

  test('chroma save', () async {
    final docs = ['Hello world!', 'Hello Moto!', 'Hi world!', 'Hey world']
        .indexed
        .map((r) {
      final (i, d) = r;
      return ChromaDoc(d, 'file${i % 2}.txt');
    }).toList();
    await chroma.add(docs);
    final items = await chroma.query('hello, world~', nResults: 2);
    expect(items.length, 2);
    final str = items.map((e) => "'${e.id}':'${e.doc}'").join(',');
    expect(str.contains('Hello world!'), true);
    final list = await Future.wait<int>(
      [chroma.allItems.then((v) => v.length), chroma.collection.count()],
    );
    expect(list, [4, 4]);
  });

  tearDownAll(() {
    chroma.dispose();
  });
}
