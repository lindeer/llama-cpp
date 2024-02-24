import 'dart:convert' show json;
import 'dart:io';

import 'package:llama_cpp/embedding.dart';

import 'chroma.dart';

Future<Chroma> setupChroma(Map<String, dynamic> config) async {
  final embeddingPath = config['embedding_model'] as String;
  final embedding = Embedding(embeddingPath);
  final chroma = await Chroma.create(
    baseUrl: config['db_server_url'] as String,
    database: config['database'] as String,
    collection: config['collection'] as String,
    embedding: embedding,
  );
  return chroma;
}

Map<String, dynamic> get appConfig {
  final uri = Directory.current.uri;
  final f1 = File.fromUri(uri.resolve('_config.json'));
  final f2 = File.fromUri(uri.resolve('config.json'));
  if (!f1.existsSync() || !f2.existsSync()) {
    print("We need '_config.json' and 'config.json' files");
    return {};
  }
  final config = json.decode(f1.readAsStringSync()) as Map<String, dynamic>;
  config.addAll(json.decode(f2.readAsStringSync()));
  return config;
}
