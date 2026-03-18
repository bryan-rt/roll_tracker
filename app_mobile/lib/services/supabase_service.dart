import 'package:supabase_flutter/supabase_flutter.dart';
import 'dart:io';
import 'package:device_info_plus/device_info_plus.dart';
import '../supabase_config.dart';

Future<Map<String, String>> getDeviceMetadata() async {
  final deviceInfo = DeviceInfoPlugin();

  if (Platform.isAndroid) {
    final androidInfo = await deviceInfo.androidInfo;
    return {
      'device_model': androidInfo.model,
      'os_version': 'Android ${androidInfo.version.release}',
    };
  } else if (Platform.isIOS) {
    final iosInfo = await deviceInfo.iosInfo;
    return {
      'device_model': iosInfo.utsname.machine,
      'os_version': 'iOS ${iosInfo.systemVersion}',
    };
  }

  return {
    'device_model': 'unknown',
    'os_version': 'unknown',
  };
}

/// Represents clip metadata from the Supabase `clips` table.
class ClipMetadata {
  final String id;
  final String storageObjectPath;
  final int durationSeconds;
  final String? signedUrl;

  ClipMetadata({
    required this.id,
    required this.storageObjectPath,
    required this.durationSeconds,
    this.signedUrl,
  });

  factory ClipMetadata.fromJson(Map<String, dynamic> json) {
    return ClipMetadata(
      id: json['id'],
      storageObjectPath: json['storage_object_path'] ?? '',
      durationSeconds: (json['duration_seconds'] as num?)?.toInt() ?? 0,
    );
  }
}

class SupabaseService {
  /// Get current authenticated profile
  Future<Map<String, dynamic>?> fetchCurrentProfile() async {
    final user = Supabase.instance.client.auth.currentUser;
    if (user == null) return null;
    return await Supabase.instance.client
        .from('profiles')
        .select()
        .eq('auth_user_id', user.id)
        .maybeSingle();
  }

  /// Insert profile row after signup
  Future<void> insertProfile({
    required String authUserId,
    required String email,
    required String displayName,
  }) async {
    await Supabase.instance.client.from('profiles').insert({
      'auth_user_id': authUserId,
      'email': email,
      'display_name': displayName,
    });
  }

  /// Set home gym on profile
  Future<void> setHomeGym(String profileId, String gymId) async {
    await Supabase.instance.client
        .from('profiles')
        .update({'home_gym_id': gymId})
        .eq('id', profileId);
  }

  /// Fetch clips for current athlete
  Future<List<ClipMetadata>> fetchMyClips(String profileId) async {
    final response = await Supabase.instance.client
        .from('clips')
        .select('id, storage_object_path, duration_seconds')
        .or('fighter_a_profile_id.eq.$profileId,'
            'fighter_b_profile_id.eq.$profileId')
        .order('created_at', ascending: false);
    return (response as List)
        .map((e) => ClipMetadata.fromJson(e))
        .toList();
  }

  /// Generate signed URL for a clip.
  /// Replaces 127.0.0.1 with the configured Supabase URL host so the
  /// phone can reach local Supabase over LAN.
  Future<String> getSignedUrl(String storageObjectPath) async {
    final response = await Supabase.instance.client.storage
        .from('match-clips')
        .createSignedUrl(storageObjectPath, 3600);
    final configuredHost = Uri.parse(supabaseUrl).host;
    return response.replaceAll('127.0.0.1', configuredHost);
  }

  /// Search gyms by name for signup flow
  Future<List<Map<String, dynamic>>> searchGyms(String query) async {
    return await Supabase.instance.client
        .from('gyms')
        .select('id, name, address')
        .ilike('name', '%$query%')
        .limit(20);
  }

  /// Submit gym interest signal
  Future<void> submitGymInterest({
    required String profileId,
    required String gymNameEntered,
  }) async {
    await Supabase.instance.client.from('gym_interest_signals').insert({
      'profile_id': profileId,
      'gym_name_entered': gymNameEntered,
    });
  }

  /// Insert a structured log event into the log_events table.
  /// Column mapping: category→event_type, severity→event_level,
  /// app_version stays top-level, everything else goes into details jsonb.
  /// event_time defaults to now() in the DB — not sent by the client.
  Future<void> insertLogEvent({
    required String category,
    required String message,
    String? severity,
    String? emailOverride,
    Map<String, dynamic>? extraFields,
  }) async {
    final email = emailOverride ??
        Supabase.instance.client.auth.currentUser?.email;

    final data = {
      'event_type': category,
      'event_level': severity ?? 'info',
      'message': message,
      'app_version': extraFields?['app_version'],
      'details': {
        'email': email ?? 'anonymous',
        'session_id': extraFields?['session_id'],
        'source': extraFields?['source'],
        'device_model': extraFields?['device_model'],
        'os_version': extraFields?['os_version'],
        'user_role': extraFields?['user_role'],
        'context': extraFields?['event_context'],
      },
    };

    try {
      await Supabase.instance.client.from('log_events').insert(data);
    } catch (e) {
      print('Failed to insert log event: $e');
    }
  }
}
