import 'dart:io';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:uuid/uuid.dart';

class LoggingContext {
  static String? sessionId;
  static String? deviceModel;
  static String? osVersion;
  static String? appVersion;
  static String? userRole;

  static Future<void> initialize() async {
    // Generate new session ID for each app session
    sessionId = const Uuid().v4();

    // Get device model and OS version
    final deviceInfo = DeviceInfoPlugin();
    if (Platform.isAndroid) {
      final info = await deviceInfo.androidInfo;
      deviceModel = info.model;
      osVersion = 'Android ${info.version.release}';
    } else if (Platform.isIOS) {
      final info = await deviceInfo.iosInfo;
      deviceModel = info.utsname.machine;
      osVersion = 'iOS ${info.systemVersion}';
    } else {
      deviceModel = 'Unknown';
      osVersion = 'Unknown';
    }

    // Get app version
    final packageInfo = await PackageInfo.fromPlatform();
    appVersion = packageInfo.version;

    // User role lookup deferred until profile-based roles are implemented
    userRole = null;
  }
}
