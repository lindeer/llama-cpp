A Dart binding for popular LLM inference framework [llama.cpp](https://github.com/ggerganov/llama.cpp), to bring AI to Dart world!

## Note

`8854044` of `llama.cpp` is latest version supporting single shared library. After that version, `libllama.so` and `libggml.so` were created, but currently dart native-assets not support loading shared libraries at the same time.

## Overview

- Text generation in a separated Dart isolate.
- Stream based output in Dart style.
- Integtate with `native_assets_cli`.
- Extremely simple usage.
- Support both LLM and embedding models.

## Trying examples

```
git clone https://github.com/lindeer/llama-cpp.git
cd llama-cpp
git submodule init --recursive
dart pub get
```

Just run in console:
```
dart --enable-experiment=native-assets run example/main.dart "/path/to/your/LLM.gguf" "your prompt"
```

or run a simple http server:
```
dart --enable-experiment=native-assets run example/server.dart "/path/to/your/LLM.gguf"
```

or run a embedding model:
```
dart --enable-experiment=native-assets run example/embedding.dart "/path/to/your/embedding.gguf" "your text line1
your text line2"
```

Also a minimal RAG example in `example/rag/` with all completely local data and model, inspired by [privateGPT](https://github.com/imartinez/privateGPT):

0. setup a chroma server:
```
pip install chromadb
uvicorn chromadb.app:app --reload --workers 1 --host 0.0.0.0 --port 8000
```

1. `cd example/rag` and creat a `config.json` and config your local models:
```json
{
  "gpt_model": "/your/local/gpt/model",
  "embedding_model": "/your/local/embedding/model"
}

```

3. save documents in `corpus/` to vector database (only txt files currently):
```
dart --enable-experiment=native-assets run bin/ingest.dart
```

4. chat with GPT in console, certainly could replace it with your beatiful GUI with flutter:
```
dart --enable-experiment=native-assets run bin/rag.dart
```

## Getting started

Ask LLM to answer with type writing effect:

```dart
  import 'package:llama_cpp/llama_cpp.dart';

  final path = '/path/to/your/LLM.gguf';
  final llama = await LlamaCpp.load(path, verbose: true);

  await for (final text in llama.answer(prompt)) {
    stdout.write(text);
  }
  stdout.writeln();

  await llama.dispose();
```
or if you want a full answer:
```
final answer = await llama.answer(prompt).join('');
```

More examples could be found at `example/`.

## Notes

native_assets_cli has beaking chanings since >0.1.0, and is not compatible with Dart 3.2, however, it could run with Dart 3.1.5.
