import 'dart:convert' show json, utf8;

import 'package:http/http.dart' as http;
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
    await _add(
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
    final result = await _query(embeddings);
    final (ids, docs, metadatas) = (
      result.ids.first,
      result.documents?.first,
      result.metadatas?.first,
    );
    return ids.indexed.map((r) {
      final (i, id) = r;
      final doc = docs?[i] ?? '';
      final metadata = metadatas?[i];
      return ChromaItem._(id, utf8.decode(doc.codeUnits), metadata);
    }).toList(growable: false);
  }

  void dispose() {
    embed.dispose();
  }

  Future<String> _add({
    required final List<String> ids,
    final List<List<double>>? embeddings,
    final List<Map<String, dynamic>>? metadatas,
    final List<String>? documents,
  }) async {
    final id = collection.id;
    final body = <String, dynamic>{
      "embeddings": embeddings,
      "metadatas": metadatas,
      "documents": documents,
      "ids": ids,
      "increment_index": true
    };
    final res = await http.post(
      Uri.parse('http://0.0.0.0:8000/api/v1/collections/$id/add'),
      headers: {
        'Content-Type': 'application/json',
      },
      body: json.encode(body),
    );
    return res.body;
  }

  Future<db.QueryResponse> _query(List<double> embedding) async {
    final id = collection.id;
    final body = <String, dynamic>{
      "where": {},
      "where_document": {},
      "query_embeddings": [embedding],
      "n_results": 2,
      "include": [
        "metadatas",
        "documents",
        "distances",
      ],
    };
    final res = await http.post(
      Uri.parse('http://0.0.0.0:8000/api/v1/collections/$id/query'),
      headers: {
        'Content-Type': 'application/json',
      },
      body: json.encode(body),
    );
    final obj = json.decode(res.body) as Map<String, dynamic>;
    return db.QueryResponse.fromJson(obj);
  }

  /*
  static Future<db.Collection> _fetchCollection({
    required final db.ChromaClient client,
    required final String name,
  }) async {
    final body = {
      "name": name,
      "get_or_create": true,
    };
    final res = await http.post(
      Uri.parse('http://0.0.0.0:8000/api/v1/collections'),
      headers: {
        'Content-Type': 'application/json',
      },
      body: json.encode(body),
    );
    final obj = json.decode(res.body);
    return db.Collection(
      name: obj['name']!,
      id: obj['id']!,
      metadata: obj['metadata'],
      tenant: client.tenant,
      database: client.database,
      api: client.api,
    );
  }
  */
}
