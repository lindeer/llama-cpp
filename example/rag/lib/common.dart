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
