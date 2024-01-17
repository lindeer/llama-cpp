import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'dart:io' show Platform;

import 'native_llama_cpp.dart' as llama_cpp;

const _libBaseName = 'llama_cpp';

final _libPath = Platform.isMacOS ? 'lib$_libBaseName.dylib'
    : Platform.isWindows ? '$_libBaseName.dll'
    : 'lib$_libBaseName.so';

// final dylib = ffi.DynamicLibrary.open(_libPath);

void loadLlama(String path) {
  llama_cpp.llama_backend_init(false);
  final charPointer = path.toNativeUtf8().cast<ffi.Char>();
  final params = llama_cpp.llama_model_default_params();
  final model = llama_cpp.llama_load_model_from_file(charPointer, params);
  final ctxParams = llama_cpp.llama_context_default_params();
  final ctx = llama_cpp.llama_new_context_with_model(model, ctxParams);

  final contextSize = ctxParams.n_ctx;
  final batchSize = ctxParams.n_batch;

  llama_cpp.llama_free(ctx);
  llama_cpp.llama_free_model(model);
  llama_cpp.llama_backend_free();
}
