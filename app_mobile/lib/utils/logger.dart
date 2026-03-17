import 'package:supabase_flutter/supabase_flutter.dart';
import '../services/supabase_service.dart';
import '../utils/logging_context.dart';

class AppLogger {
  final SupabaseService _supabaseService;

  AppLogger(this._supabaseService);

  /// Logs a structured event to the Supabase `log_events` table.
  Future<void> logEvent(String category,
                        String message,
                        {String? severity,
                        Map<String, dynamic>? context}) async {
    try {
      final resolvedSeverity = severity ?? _autoDetermineSeverity(message);
      final userEmail = Supabase.instance.client.auth.currentUser?.email;

    await _supabaseService.insertLogEvent(
      category: category,
      message: message,
      severity: resolvedSeverity,
      extraFields: {
        'email': userEmail,
        'session_id': LoggingContext.sessionId,
        'source': 'mobile_app',
        'device_model': LoggingContext.deviceModel,
        'os_version': LoggingContext.osVersion,
        'user_role': LoggingContext.userRole,
        'app_version': LoggingContext.appVersion,
        'event_context': context,
      },
    );
  } catch (e) {
    print('❌ Logging error: $e');
  }
  }

  /// Infers severity from keywords in the message
  String _autoDetermineSeverity(String message) {
    final lower = message.toLowerCase();
    if (lower.contains('fail') || lower.contains('error') || lower.contains('exception')) return 'error';
    if (lower.contains('warn') || lower.contains('slow') || lower.contains('retry')) return 'warn';
    if (lower.contains('debug') || lower.contains('trace')) return 'debug';
    return 'info';
  }
}
