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
  await build(args, _builder);
}

Future<void> _builder(BuildConfig buildConfig, BuildOutput buildOutput) async {
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

  final linkMode = _linkMode(buildConfig.linkModePreference);
  final libName = buildConfig.targetOS.libraryFileName(packageName, linkMode);
  final libUri = buildConfig.outputDirectory.resolve(libName);
  final uri = pkgRoot.resolve(p.join('src', _repoLibName));
  final file = File.fromUri(uri).resolveSymbolicLinksSync();
  File(file).renameSync(libUri.path);

  buildOutput.addAsset(NativeCodeAsset(
    package: packageName,
    name: 'src/lib_$packageName.dart',
    linkMode: linkMode,
    os: buildConfig.targetOS,
    file: libUri,
    architecture: buildConfig.targetArchitecture,
  ));
  final src = [
    'src/llama.cpp',
    'src/ggml.c',
    'src/ggml-alloc.c',
    'src/ggml-backend.c',
    'src/ggml-quants.c',
  ];

  buildOutput.addDependencies([
    ...src.map((s) => pkgRoot.resolve(s)),
    pkgRoot.resolve('build.dart'),
  ]);
}

LinkMode _linkMode(LinkModePreference preference) {
  if (preference == LinkModePreference.dynamic ||
      preference == LinkModePreference.preferDynamic) {
    return DynamicLoadingBundled();
  }
  assert(preference == LinkModePreference.static ||
      preference == LinkModePreference.preferStatic);
  return StaticLinking();
}
