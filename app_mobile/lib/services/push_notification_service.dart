import 'package:flutter/foundation.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class PushNotificationService {
  static String? _currentProfileId;

  /// Initialize push notifications after the user's profile is fully loaded.
  /// Do NOT call before onboarding is complete.
  static Future<void> initialize(String profileId) async {
    _currentProfileId = profileId;

    final messaging = FirebaseMessaging.instance;

    // Request permission
    final settings = await messaging.requestPermission(
      alert: true,
      badge: true,
      sound: true,
    );

    if (settings.authorizationStatus == AuthorizationStatus.denied) {
      debugPrint('PushNotificationService: permission denied');
      return;
    }

    // Get and register FCM token
    final token = await messaging.getToken();
    if (token != null) {
      await _upsertToken(profileId, token);
    }

    // Handle token refresh
    messaging.onTokenRefresh.listen((newToken) async {
      final pid = _currentProfileId;
      if (pid != null) {
        await _upsertToken(pid, newToken);
      }
    });

    // Foreground message handler
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      debugPrint(
        'PushNotificationService: foreground message: '
        '${message.notification?.title} — ${message.notification?.body}',
      );
    });
  }

  /// Upsert device token into device_tokens table.
  static Future<void> _upsertToken(String profileId, String token) async {
    try {
      await Supabase.instance.client.from('device_tokens').upsert(
        {
          'profile_id': profileId,
          'token': token,
          'platform': 'android',
          'updated_at': DateTime.now().toUtc().toIso8601String(),
        },
        onConflict: 'profile_id,token',
      );
      debugPrint('PushNotificationService: token registered');
    } catch (e) {
      debugPrint('PushNotificationService: token upsert failed: $e');
    }
  }

  /// Remove device token on logout.
  static Future<void> removeToken() async {
    try {
      final messaging = FirebaseMessaging.instance;
      final token = await messaging.getToken();
      if (token != null) {
        await Supabase.instance.client
            .from('device_tokens')
            .delete()
            .eq('token', token);
        debugPrint('PushNotificationService: token removed');
      }
    } catch (e) {
      debugPrint('PushNotificationService: token removal failed: $e');
    }
    _currentProfileId = null;
  }
}
