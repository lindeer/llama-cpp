import 'dart:convert' show utf8;
import 'dart:ffi' as ffi;
import 'dart:typed_data' show Uint8List;

import 'package:ffi/ffi.dart' show Utf8, Utf8Pointer, calloc;

import 'lib_llama_cpp.dart' as llama_cpp;

CharArray _fillChars(String str, CharArray Function(int size) getter) {
  final units = utf8.encode(str);
  final size = units.length + 1;
  final len = size - 1;
  final buf = getter(size);
  final pointer = buf._buf.cast<ffi.Uint8>();
  final raw = pointer.asTypedList(size);
  raw.setAll(0, units);
  raw[len] = 0;
  buf._len = len;
  return buf;
}

/// Util class for data conversion between Dart `String` and C `const char *`.
/// From Dart `String` to C `const char *`:
/// ```dart
/// final cStr = CharArray.from('some thing as string');
/// final p = cStr.pointer;
/// call_some_C_function(p, cStr.length);
/// cStr.dispose();
/// ```
/// To reuse an existing `CharArray`:
/// ```dart
/// CharArray cStr;
/// final p = cStr.pavedBy('some thing as string');
/// call_some_C_function(p, cStr.length);
/// cStr.dispose();
/// ```
///
/// From C `const char *` to Dart `String`:
/// ```dart
/// final p = call_some_C_function();
/// final str = CharArray.fromNative(p);
/// ```
final class CharArray {
  int _size;
  int _len;
  ffi.Pointer<ffi.Char> _buf;

  CharArray({int size = 32})
      : _size = size,
        _len = 0,
        _buf = calloc.allocate<ffi.Char>(size * ffi.sizeOf<ffi.Char>());

  /// Create newly a buffer for an existing Dart string.
  factory CharArray.from(String str) {
    final buf = _fillChars(str, (size) => CharArray(size: size));
    return buf;
  }

  /// A helper function that converts the given Dart String to `const char *`
  /// with an existing `CharArray`.
  /// The capacity is expanded automatically.
  ffi.Pointer<ffi.Char> pavedBy(String str) {
    _fillChars(str, (size) => this.._resize(size));
    return _buf;
  }

  int get length => _len;

  ffi.Pointer<ffi.Char> get pointer => _buf;

  /// Convert to Dart string with data in current buf and specified length.
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

  /// Convert to Dart string with extern CString pointer without length.
  static String toDartString(ffi.Pointer<ffi.Char> pointer) =>
      pointer.cast<Utf8>().toDartString();

  /// A string representation for a token.
  /// In some model, one token would not return a full utf8 string.
  String tokenString(ffi.Pointer<llama_cpp.llama_model> model, int token) {
    final bytes = tokenBytes(model, token);
    try {
      return dartString;
    } on Exception catch (_) {
      return bytes.toString();
    }
  }

  /// Return a raw bytes with a given token Id.
  /// We need convert assigned int to unassigned, or else
  /// `FormatException: Invalid UTF-8 byte` would be thrown.
  List<int> tokenBytes(ffi.Pointer<llama_cpp.llama_model> model, int token) {
    final len = llama_cpp.llama_token_to_piece(
      model,
      token,
      _buf,
      _size,
      false,
    );
    if (len < 0) {
      _resize(-len);
      _len = llama_cpp.llama_token_to_piece(model, token, _buf, _size, false);
    } else {
      _len = len;
    }
    return Uint8List.fromList(List<int>.generate(_len, (i) => _buf[i]));
  }

  /// Release native resources.
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
      : _size = size,
        _len = 0,
        _buf = calloc.allocate<llama_cpp.llama_token>(
            size * ffi.sizeOf<llama_cpp.llama_token>());

  int get length => _len;

  int get capacity => _size;

  int operator [](int pos) => _buf[pos];

  ffi.Pointer<llama_cpp.llama_token> pointerAt(int pos) => _buf + pos;

  void pavedBy(
    ffi.Pointer<llama_cpp.llama_model> model,
    CharArray text, {
    bool addBos = false,
  }) {
    final size = text.length + 1;
    _resize(size);
    final len = llama_cpp.llama_tokenize(
      model,
      text.pointer,
      text.length,
      _buf,
      _size,
      addBos,
      false,
    );
    if (len < 0) {
      throw Exception("tokenize '${text.dartString}' failed!");
    }
    _len = len;
  }

  int clear() {
    var n = _len;
    _len = 0;
    return n;
  }

  void add(int token) {
    _resize(_len + 1);
    _buf[_len++] = token;
  }

  bool _resize(int size) {
    if (size <= _size) {
      return false;
    }
    dispose();
    _buf = calloc.allocate<llama_cpp.llama_token>(
        size * ffi.sizeOf<llama_cpp.llama_token>());
    _size = size;
    // copy existing elements?
    _len = 0;
    return true;
  }

  List<int> toList() => List.generate(_len, (i) => _buf[i]);

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
  final pointer = calloc.allocate<llama_cpp.llama_token_data_array>(
      ffi.sizeOf<llama_cpp.llama_token_data>());

  TokenDataArray(int size)
      : _size = size,
        _len = 0,
        _buf = calloc.allocate<llama_cpp.llama_token_data>(
            size * ffi.sizeOf<llama_cpp.llama_token_data>());

  int get length => _len;

  llama_cpp.llama_token_data operator [](int pos) => _buf[pos];

  void setLogit(int pos, double value) {
    (_buf + pos).ref.logit = value;
  }

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
    _release();
    _buf = calloc.allocate<llama_cpp.llama_token_data>(
        size * ffi.sizeOf<llama_cpp.llama_token_data>());
    _size = size;
    // copy existing elements?
    _len = 0;
    return true;
  }

  // not free array pointer
  void _release() {
    calloc.free(_buf);
    _len = 0;
    _size = 0;
  }

  void dispose() {
    _release();
    calloc.free(pointer);
  }
}
