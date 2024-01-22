import 'dart:convert' show utf8;
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:llama_cpp/native_llama_cpp.dart' as llama_cpp;

extension NativeStringExt on String {

  ffi.Pointer<ffi.Char> into(NativeString native) {
    final units = utf8.encode(this);
    final size = units.length + 1;
    final len = size - 1;
    native._resize(size);
    final pointer = native._buf.cast<ffi.Uint8>();
    final nativeString = pointer.asTypedList(size);
    nativeString.setAll(0, units);
    nativeString[len] = 0;
    native._len = len;
    return native._buf;
  }
}

final class NativeString {
  int _size = 0;
  int _len = 0;
  ffi.Pointer<ffi.Char> _buf;

  NativeString({int size = 1024})
    : _size = size
    , _len = 0
    , _buf = calloc.allocate<ffi.Char>(size * ffi.sizeOf<ffi.Char>());

  int get length => _len;

  ffi.Pointer<ffi.Char> get pointer => _buf;

  String get dartString => _buf.cast<Utf8>().toDartString(length: _len);

  bool _resize(int size) {
    if (size <= _size) {
      return false;
    }
    dispose();
    _buf = calloc.allocate<ffi.Char>(size * ffi.sizeOf<ffi.Char>());
    _size = size;
    // copy existing elements?
    _len = 0;
    return true;
  }

  String fromToken(ffi.Pointer<llama_cpp.llama_model> model, int token) {
    int len = llama_cpp.llama_token_to_piece(model, token, _buf, _size);
    if (len < 0) {
      _resize(-len);
      _len = -len;
    } else {
      _len = len;
    }
    return dartString;
  }

  void dispose() {
    calloc.free(_buf);
    _len = 0;
    _size = 0;
  }
}

final class TokenArray {
  int _size;
  int _len;
  ffi.Pointer<llama_cpp.llama_token> _buf;

  TokenArray({int size = 512})
    : _size = size
    , _len = 0
    , _buf = calloc.allocate<llama_cpp.llama_token>(size * ffi.sizeOf<llama_cpp.llama_token>());

  int get length => _len;

  int operator [](int pos) => _buf[pos];

  void pavedBy(ffi.Pointer<llama_cpp.llama_model> model, NativeString text) {
    final size = text.length + 1;
    _resize(size);
    final len = llama_cpp.llama_tokenize(
      model,
      text.pointer, text.length,
      _buf, _size,
      false, true,
    );
    if (len < 0) {
      throw Exception("tokenize '${text.dartString}' failed!");
    }
    _len = len;
  }

  bool _resize(int size) {
    if (size <= _size) {
      return false;
    }
    dispose();
    _buf = calloc.allocate<llama_cpp.llama_token>(size * ffi.sizeOf<llama_cpp.llama_token>());
    _size = size;
    // copy existing elements?
    _len = 0;
    return true;
  }

  void dispose() {
    calloc.free(_buf);
    _len = 0;
    _size = 0;
  }
}

final class TokenDataArray {
  int _size;
  int _len;
  ffi.Pointer<llama_cpp.llama_token_data> _buf;
  final pointer = calloc.allocate<llama_cpp.llama_token_data_array>(ffi.sizeOf<llama_cpp.llama_token_data>());

  TokenDataArray(int size)
    : _size = size
    , _len = 0
    , _buf = calloc.allocate<llama_cpp.llama_token_data>(size * ffi.sizeOf<llama_cpp.llama_token_data>());

  int get length => _len;

  void pavedBy(ffi.Pointer<ffi.Float> logits, int size) {
    _resize(size);
    for (var id = 0; id < size; id++) {
      _buf[id]
        ..id = id
        ..logit = logits[id]
        ..p = 0;
    }
    pointer.ref
      ..data = _buf
      ..size = size
      ..sorted = false;
  }

  bool _resize(int size) {
    if (size <= _size) {
      return false;
    }
    dispose();
    _buf = calloc.allocate<llama_cpp.llama_token_data>(size * ffi.sizeOf<llama_cpp.llama_token_data>());
    _size = size;
    // copy existing elements?
    _len = 0;
    return true;
  }

  void dispose() {
    calloc.free(_buf);
    calloc.free(pointer);
    _len = 0;
    _size = 0;
  }
}
