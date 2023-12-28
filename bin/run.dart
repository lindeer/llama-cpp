import 'dart:io';
import 'dart:isolate' show Isolate;
import 'package:native_assets_cli/native_assets_cli.dart';

Future<Uri?> fsLocation([String path = '/']) async {
  final uri = Uri.parse('package:native_assets_cli$path');
  return await Isolate.resolvePackageUri(uri);
}

String unparseKey(String key) => key.replaceAll('.', '__').toUpperCase();

final Uri? envScript = Platform
    .environment[unparseKey(CCompilerConfig.envScriptConfigKeyFull)]
    ?.asFileUri();

/// Arguments for [envScript] provided by environment.
final List<String>? envScriptArgs = Platform
    .environment[unparseKey(CCompilerConfig.envScriptArgsConfigKeyFull)]
    ?.split(' ');

extension on String {
  Uri asFileUri() => Uri.file(this);
}

Future<void> main() async {
  const name = 'llama_cpp';
  final tempUri = (await Directory('/home/wesley/Work/projects/native-dart/pkgs/native_assets_cli/hehe').create()).uri;

  final packageUri = await fsLocation();
  final testPackageUri = packageUri?.resolve('../example/$name/') ?? Uri.parse('');
  final dartUri = Uri.file(Platform.resolvedExecutable);
  const dryRun = false;

  final Uri? cc = Platform
    .environment[unparseKey(CCompilerConfig.ccConfigKeyFull)]
    ?.asFileUri();

  final processResult = await Process.run(
    dartUri.toFilePath(),
    [
      'build.dart',
      '-Dout_dir=${tempUri.toFilePath()}',
      '-Dpackage_name=$name',
      '-Dpackage_root=${testPackageUri.toFilePath()}',
      '-Dtarget_os=${OS.current}',
      '-Dversion=${BuildConfig.version}',
      '-Dlink_mode_preference=dynamic',
      '-Ddry_run=$dryRun',
      if (!dryRun) ...[
        '-Dtarget_architecture=${Architecture.current}',
        '-Dbuild_mode=debug',
        if (cc != null) '-Dcc=${cc.toFilePath()}',
        if (envScript != null)
          '-D${CCompilerConfig.envScriptConfigKeyFull}='
              '${envScript!.toFilePath()}',
        if (envScriptArgs != null)
          '-D${CCompilerConfig.envScriptArgsConfigKeyFull}='
              '${envScriptArgs!.join(' ')}',
      ],
    ],
    workingDirectory: testPackageUri.toFilePath(),
  );
  if (processResult.exitCode != 0) {
    print(processResult.stdout);
    print(processResult.stderr);
    print(processResult.exitCode);
  }
}
