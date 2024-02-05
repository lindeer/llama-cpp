import 'dart:convert' show utf8;
import 'dart:ffi' as ffi;
import 'dart:typed_data' show Uint8List;

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

  NativeString({int size = 32})
      : _size = size,
        _len = 0,
        _buf = calloc.allocate<ffi.Char>(size * ffi.sizeOf<ffi.Char>());

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

  static String fromNative(ffi.Pointer<ffi.Char> pointer) =>
      pointer.cast<Utf8>().toDartString();

  /// A string representation for a token.
  /// In some model, one token would not return a full utf8 string.
  String tokenString(ffi.Pointer<llama_cpp.llama_model> model, int token) {
    if (token == 0 || token == 1 || token == 2) {
      return '';
    }
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
    final len = llama_cpp.llama_token_to_piece(model, token, _buf, _size);
    if (len < 0) {
      _resize(-len);
      _len = llama_cpp.llama_token_to_piece(model, token, _buf, _size);
    } else {
      _len = len;
    }
    return Uint8List.fromList(List<int>.generate(_len, (i) => _buf[i]));
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
      : _size = size,
        _len = 0,
        _buf = calloc.allocate<llama_cpp.llama_token>(
            size * ffi.sizeOf<llama_cpp.llama_token>());

  int get length => _len;

  int get capacity => _size;

  int operator [](int pos) => _buf[pos];

  ffi.Pointer<llama_cpp.llama_token> pointerAt(int pos) => _buf.elementAt(pos);

  void pavedBy(ffi.Pointer<llama_cpp.llama_model> model, NativeString text) {
    final size = text.length + 1;
    _resize(size);
    final len = llama_cpp.llama_tokenize(
      model,
      text.pointer,
      text.length,
      _buf,
      _size,
      false,
      true,
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
    _buf.elementAt(pos).ref.logit = value;
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
