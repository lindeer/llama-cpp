A Dart binding for popular LLM inference framework [llama.cpp](https://github.com/ggerganov/llama.cpp), to bring AI to Dart world!

## Overview

- Text generation in a separated Dart isolate.
- Stream based output in Dart style.
- Integtate with `native_assets_cli`.
- Extremely simple usage.

## Trying examples

Just run in console:
```
dart --enable-experiment=native-assets run bin/main.dart "/path/to/your/LLM.gguf" "your prompt"
```

or run a simple http server:
```
dart --enable-experiment=native-assets run bin/server.dart "/path/to/your/LLM.gguf"
```

## Getting started

Ask LLM to answer with type writing effect:

```dart
  import 'package:llama_cpp/llama_cpp.dart';

  final path = '/path/to/your/LLM.gguf';
  final llama = await LlamaCpp.load(path, verbose: true);

  await for (final text in llama.answer('{"prompt":"$prompt"}')) {
    stdout.write(text);
  }
  stdout.writeln();

  await llama.dispose();
```
or if you want a full answer:
```
final answer = await llama.answer('{"prompt":"$prompt"}').join('');
```

More examples could be found at `example/`.

## Notes

native_assets_cli has beaking chanings since >0.1.0, and is not compatible with Dart >=3.2.
