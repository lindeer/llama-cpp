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
  final charPointer = path.toNativeUtf8().cast<ffi.Char>();
  final params = llama_cpp.llama_model_default_params();
  final _model = llama_cpp.llama_load_model_from_file(charPointer, params);
}
