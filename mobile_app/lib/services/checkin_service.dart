import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:geolocator/geolocator.dart';
import 'package:network_info_plus/network_info_plus.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class CheckinService {
  static final _connectivity = Connectivity();
  static final _networkInfo = NetworkInfo();
  static StreamSubscription? _subscription;

  /// Start listening for WiFi connection changes and do an immediate check.
  /// Call once from main() after Supabase.initialize().
  static void startListening() {
    _subscription = _connectivity.onConnectivityChanged.listen(
      (results) async {
        // onConnectivityChanged returns List<ConnectivityResult> in v6+
        final connected = results.contains(ConnectivityResult.wifi);
        if (!connected) return;
        await _handleWifiConnection();
      },
    );

    // Immediate check for current WiFi (the listener only fires on changes)
    checkCurrentWifi();
  }

  /// Check current WiFi and attempt auto check-in.
  /// Requests location permission if needed (required to read SSID on Android).
  /// Public so it can be called after auth state changes.
  static Future<void> checkCurrentWifi() async {
    try {
      // Ensure location permission is granted (required for SSID on Android 10+)
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        debugPrint('CheckinService: location permission denied, skipping WiFi check');
        return;
      }

      await _handleWifiConnection();
    } catch (e) {
      debugPrint('CheckinService: WiFi check failed: $e');
    }
  }

  static Future<void> _handleWifiConnection() async {
    try {
      final user = Supabase.instance.client.auth.currentUser;
      if (user == null) return;

      // Read current WiFi SSID and BSSID
      // On Android requires ACCESS_FINE_LOCATION permission
      final ssid = await _networkInfo.getWifiName();
      final bssid = await _networkInfo.getWifiBSSID();

      if (ssid == null || ssid.isEmpty) return;

      // Strip surrounding quotes iOS adds to SSID
      final cleanSsid = ssid.replaceAll('"', '');

      debugPrint('CheckinService: detected WiFi SSID=$cleanSsid');

      // Look up profile for current user
      final profile = await Supabase.instance.client
          .from('profiles')
          .select('id')
          .eq('auth_user_id', user.id)
          .maybeSingle();

      if (profile == null) return;

      // Query gyms table for matching SSID (and optionally BSSID)
      final query = Supabase.instance.client
          .from('gyms')
          .select('id')
          .eq('wifi_ssid', cleanSsid);

      // If BSSID available, use it to narrow match
      final gyms = bssid != null && bssid.isNotEmpty
          ? await query.eq('wifi_bssid', bssid)
          : await query;

      if ((gyms as List).isEmpty) return;

      final gymId = gyms.first['id'];
      final profileId = profile['id'];

      debugPrint('CheckinService: auto check-in at gym=$gymId for profile=$profileId');

      // Write check-in record (3hr TTL enforced by DB trigger)
      await Supabase.instance.client.from('gym_checkins').insert({
        'profile_id': profileId,
        'gym_id': gymId,
        'checked_in_at': DateTime.now().toUtc().toIso8601String(),
        'is_active': true,
        'source': 'wifi_auto',
      });
    } catch (e) {
      // Silent failure — check-in is best-effort, never blocks the user
      debugPrint('CheckinService: check-in failed: $e');
    }
  }

  /// Manual check-in (e.g. from Find a Gym screen).
  /// auto_expires_at is handled by the DB trigger — do not pass it.
  static Future<void> manualCheckIn(String profileId, String gymId) async {
    await Supabase.instance.client.from('gym_checkins').insert({
      'profile_id': profileId,
      'gym_id': gymId,
      'checked_in_at': DateTime.now().toUtc().toIso8601String(),
      'is_active': true,
      'source': 'manual',
    });
  }

  /// Stop listening (call on app dispose if needed)
  static void stopListening() {
    _subscription?.cancel();
    _subscription = null;
  }
}
