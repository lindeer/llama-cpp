import 'package:chromadb/chromadb.dart' as db;
import 'package:llama_cpp/embedding.dart';
import 'package:uuid/uuid.dart' show Uuid;

final class ChromaDoc {
  final String content;
  final String source;

  const ChromaDoc(this.content, this.source);

  @override
  String toString() => "('$source':'$content')";
}

final class ChromaItem {
  final String? id;
  final String doc;
  final Map<String, dynamic>? metadata;

  const ChromaItem._(this.id, this.doc, this.metadata);

  @override
  String toString() => "ChromaItem(id:$id, doc:'$doc', meta:$metadata)";
}

class Chroma {
  final db.ChromaClient client;
  final db.Collection collection;

  /// Not use `EmbeddingFunction` because of its type
  final Embedding embed;

  Chroma._(this.client, this.collection, this.embed);

  static Future<Chroma> create({
    required String baseUrl,
    required String database,
    required collection,
    required Embedding embedding,
  }) async {
    final client = db.ChromaClient(
      baseUrl: baseUrl,
      database: database,
    );
    final c = await client.getOrCreateCollection(name: collection);
    return Chroma._(client, c, embedding);
  }

  Future<void> add(List<ChromaDoc> docs) async {
    final len = docs.length;
    final uuid = Uuid();
    final ids = List.generate(len, (i) => uuid.v1());
    final docList = docs.map((e) => e.content).toList(growable: false);
    final embeddings = embed.embedBatch(docList);
    final metadatas = docs.map((e) => {'source': e.source}).toList();
    await collection.add(
      ids: ids,
      documents: docList,
      embeddings: embeddings,
      metadatas: metadatas,
    );
  }

  Future<List<ChromaItem>> get allItems async {
    final res = await collection.get();
    return res.ids.indexed.map((r) {
      final (i, id) = r;
      final doc = res.documents?[i] ?? '';
      final metadata = res.metadatas?[i];
      return ChromaItem._(id, doc, metadata);
    }).toList(growable: false);
  }

  Future<List<ChromaItem>> query(
    String doc, {
    final int nResults = 4,
  }) async {
    final embeddings = embed.embedSingle(doc);
    final result = await collection.query(
      queryEmbeddings: [embeddings],
      nResults: nResults,
    );
    final (ids, docs, metadatas) = (
      result.ids.first,
      result.documents?.first,
      result.metadatas?.first,
    );
    return ids.indexed.map((r) {
      final (i, id) = r;
      final doc = docs?[i] ?? '';
      final metadata = metadatas?[i];
      return ChromaItem._(id, doc, metadata);
    }).toList(growable: false);
  }

  void dispose() {
    embed.dispose();
  }
}
