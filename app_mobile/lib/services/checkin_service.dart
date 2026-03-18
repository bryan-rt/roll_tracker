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
  static Timer? _refreshTimer;
  static String? _currentGymId;
  static String? _currentProfileId;

  /// Start listening for WiFi connection changes and do an immediate check.
  /// Call once from main() after Supabase.initialize().
  static void startListening() {
    _subscription = _connectivity.onConnectivityChanged.listen(
      (results) async {
        // onConnectivityChanged returns List<ConnectivityResult> in v6+
        final connected = results.contains(ConnectivityResult.wifi);
        if (!connected) {
          _cancelRefreshTimer();
          return;
        }
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

      if (ssid == null || ssid.isEmpty) {
        _cancelRefreshTimer();
        return;
      }

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

      // Query gyms table for matching SSID
      // BSSID match is best-effort: only filter if gym has one stored
      final gyms = await Supabase.instance.client
          .from('gyms')
          .select('id, wifi_bssid')
          .eq('wifi_ssid', cleanSsid);

      if ((gyms as List).isEmpty) {
        _cancelRefreshTimer();
        return;
      }

      // Prefer BSSID-matched gym if available, otherwise take first SSID match
      var matched = gyms.first;
      if (bssid != null && bssid.isNotEmpty) {
        final bssidMatch = (gyms as List).where(
          (g) => g['wifi_bssid'] == bssid,
        );
        if (bssidMatch.isNotEmpty) matched = bssidMatch.first;
      }

      final gymId = matched['id'];
      final profileId = profile['id'];

      debugPrint('CheckinService: auto check-in at gym=$gymId for profile=$profileId');

      await _upsertCheckin(profileId, gymId, 'wifi_auto');

      // Start periodic refresh timer (slides TTL every 60 minutes)
      _currentGymId = gymId;
      _currentProfileId = profileId;
      _startRefreshTimer();
    } catch (e) {
      // Silent failure — check-in is best-effort, never blocks the user
      debugPrint('CheckinService: check-in failed: $e');
    }
  }

  /// Upsert a check-in row. The DB trigger computes auto_expires_at = checked_in_at + 3h.
  /// On conflict (profile_id, gym_id), the row is updated — sliding the TTL forward.
  static Future<void> _upsertCheckin(String profileId, String gymId, String source) async {
    await Supabase.instance.client.from('gym_checkins').upsert(
      {
        'profile_id': profileId,
        'gym_id': gymId,
        'checked_in_at': DateTime.now().toUtc().toIso8601String(),
        'is_active': true,
        'source': source,
      },
      onConflict: 'profile_id,gym_id',
    );
  }

  /// Manual check-in (e.g. from Find a Gym screen).
  /// auto_expires_at is handled by the DB trigger — do not pass it.
  static Future<void> manualCheckIn(String profileId, String gymId) async {
    await _upsertCheckin(profileId, gymId, 'manual');
  }

  /// Start a periodic timer that re-upserts the check-in every 60 minutes
  /// while the athlete remains on the gym's WiFi.
  static void _startRefreshTimer() {
    _cancelRefreshTimer();
    _refreshTimer = Timer.periodic(const Duration(hours: 1), (_) async {
      final gymId = _currentGymId;
      final profileId = _currentProfileId;
      if (gymId == null || profileId == null) {
        _cancelRefreshTimer();
        return;
      }
      try {
        debugPrint('CheckinService: refreshing check-in TTL for gym=$gymId');
        await _upsertCheckin(profileId, gymId, 'wifi_auto');
      } catch (e) {
        debugPrint('CheckinService: TTL refresh failed: $e');
      }
    });
  }

  /// Cancel the periodic refresh timer and clear tracked gym/profile.
  static void _cancelRefreshTimer() {
    _refreshTimer?.cancel();
    _refreshTimer = null;
    _currentGymId = null;
    _currentProfileId = null;
  }

  /// Stop listening (call on app dispose if needed)
  static void stopListening() {
    _subscription?.cancel();
    _subscription = null;
    _cancelRefreshTimer();
  }
}
