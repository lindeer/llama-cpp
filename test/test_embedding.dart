import 'dart:io';

import 'package:llama_cpp/embedding.dart';
import 'package:test/test.dart';

const _url = 'https://hf-mirror.com/CompendiumLabs/bge-base-zh-v1.5-gguf/'
    'blob/main/bge-base-zh-v1.5-q4_k_m.gguf';
const _embedModelPath = 'test/data/bge-base-zh-v1.5.gguf';

void _compareList(List<double> d, List<double> v) {
  expect(d.length, v.length);
  for (var j = 0; j < d.length; j++) {
    final v1 = d[j];
    final v2 = v[j];
    expect((v1 - v2).abs() < 0.0000015, true, reason: "[$j]: '$v1' != '$v2'");
  }
}

void main() {
  final embedding = Embedding(_embedModelPath);
  final prompts = File('test/data/text.txt').readAsLinesSync();
  final values = File('test/data/values.txt').readAsLinesSync().map((l) {
    final v = l
        .replaceAll('[', '')
        .replaceAll(']', '')
        .split(',')
        .where((e) => e.trim().isNotEmpty)
        .indexed
        .map((idx) {
      final (j, e) = idx;
      try {
        return double.parse(e.trim());
      } catch (x) {
        print("parse [$j]: '$e' failed!");
        rethrow;
      }
    }).toList();
    return v;
  }).toList();

  setUp(() {
    expect(
      File(_embedModelPath).existsSync(),
      true,
      reason: "Download model from $_url in '$_embedModelPath' before testing!",
    );
  });

  test('basic embed', () {
    final d1 = embedding.embedSingle(prompts[0]);
    _compareList(d1, values[0]);
    final d2 = embedding.embedSingle(prompts[1]);
    _compareList(d2, values[1]);
  });

  test('batch embed', () {
    final result = embedding.embedBatch(prompts);
    expect(result.length, values.length);
    for (final r in result.indexed) {
      final (i, d) = r;
      _compareList(d, values[i]);
    }
  });

  tearDownAll(() {
    embedding.dispose();
  });
}
