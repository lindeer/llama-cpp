// Copyright (c) 2023, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

import 'dart:io' show File, Platform, Process, exit, stdout;
import 'package:path/path.dart' as p;
import 'package:native_assets_cli/native_assets_cli.dart';
import 'package:ffigen/ffigen.dart' as fg;

const packageName = 'llama_cpp';
const _repoLibName = 'libllama.so';

/// Implements the protocol from `package:native_assets_cli` by building
/// the C code in `src/` and reporting what native assets it built.
void main(List<String> args) async {
  // Parse the build configuration passed to this CLI from Dart or Flutter.
  final buildConfig = await BuildConfig.fromArgs(args);
  final ffiConfig = fg.Config.fromFile(
    File.fromUri(buildConfig.packageRoot.resolve('ffigen.yaml')));
  final ffiLib = fg.parse(ffiConfig);
  ffiLib.generateFile(File.fromUri(Uri.parse(ffiConfig.output)));

  final env = Platform.environment;
  final cublas = env['LLAMA_CUBLAS'];
  final proResult = await Process.run(
    'make',
    [
      _repoLibName,
      if (cublas != null)
        'LLAMA_CUBLAS=$cublas',
    ],
    workingDirectory: 'src',
  );
  print(proResult.stdout);
  print(proResult.stderr);
  if (proResult.exitCode != 0) {
    exit(-1);
  }

  final linkMode = buildConfig.linkModePreference.preferredLinkMode;
  final libName = buildConfig.targetOs.libraryFileName(packageName, linkMode);
  final libUri = buildConfig.outDir.resolve(libName);
  print('libUri=$libUri');
  File(p.join('src', _repoLibName)).renameSync(libUri.path);
}
