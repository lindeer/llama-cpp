// Copyright (c) 2023, the Dart project authors.  Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

import 'dart:io' show File, Platform, Process, exit, stderr, stdout;
import 'package:path/path.dart' as p;
import 'package:native_assets_cli/native_assets_cli.dart';

const packageName = 'llama_cpp';
const _repoLibName = 'libllama.so';

Future<String> _commandPath(String cmd) async {
  final proc = await Process.run('which', [cmd]);
  stderr.write(proc.stderr);
  return proc.exitCode == 0 ? proc.stdout.toString() : '';
}

/// Implements the protocol from `package:native_assets_cli` by building
/// the C code in `src/` and reporting what native assets it built.
void main(List<String> args) async {
  // Parse the build configuration passed to this CLI from Dart or Flutter.
  final buildConfig = await BuildConfig.fromArgs(args);

  final env = Platform.environment;
  final nvcc = env['LLAMA_CUDA_NVCC'] ?? await _commandPath('nvcc');
  final arch = env['CUDA_DOCKER_ARCH'] ?? 'compute_75';
  final pkgRoot = buildConfig.packageRoot;
  final srcDir = pkgRoot.resolve('src');
  final proc = await Process.start(
    'make',
    [
      '-j',
      _repoLibName,
      if (nvcc.isNotEmpty) ...['LLAMA_CUBLAS=1', 'CUDA_DOCKER_ARCH=$arch'],
    ],
    workingDirectory: srcDir.path,
  );
  stdout.addStream(proc.stdout);
  stderr.addStream(proc.stderr);
  final code = await proc.exitCode;
  if (code != 0) {
    final p = await Process.run('gcc', ['--version']);
    if (p.exitCode == 0) {
      final gccVer = p.stdout.toString();
      stderr.writeln("Build failed, make sure 'gcc>=9.5.0':\n$gccVer");
    } else {
      stderr.writeln("GCC not exists!");
    }
    exit(code);
  }

  final linkMode = buildConfig.linkModePreference.preferredLinkMode;
  final libName = buildConfig.targetOs.libraryFileName(packageName, linkMode);
  final libUri = buildConfig.outDir.resolve(libName);
  File(p.join(srcDir.path, _repoLibName)).renameSync(libUri.path);

  final buildOutput = BuildOutput();
  buildOutput.assets.add(Asset(
    id: 'package:$packageName/src/lib_$packageName.dart',
    linkMode: linkMode,
    target: buildConfig.target,
    path: AssetAbsolutePath(libUri),
  ));
  final src = [
    'src/llama.cpp',
    'src/ggml.c',
    'src/ggml-alloc.c',
    'src/ggml-backend.c',
    'src/ggml-quants.c',
  ];

  buildOutput.dependencies.dependencies.addAll([
    ...src.map((s) => pkgRoot.resolve(s)),
    pkgRoot.resolve('build.dart'),
  ]);
  await buildOutput.writeToFile(outDir: buildConfig.outDir);
}
