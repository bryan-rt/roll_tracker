import 'package:supabase_flutter/supabase_flutter.dart';
import 'supabase_service.dart';

class AuthService {
  final SupabaseService _supabaseService = SupabaseService();

  Stream<AuthState> get authStateChanges =>
      Supabase.instance.client.auth.onAuthStateChange;

  User? get currentUser => Supabase.instance.client.auth.currentUser;

  Future<AuthResponse> signUp({
    required String email,
    required String password,
    required String displayName,
    String? gymId,
  }) async {
    final response = await Supabase.instance.client.auth.signUp(
      email: email,
      password: password,
    );

    if (response.user != null) {
      await _supabaseService.insertProfile(
        authUserId: response.user!.id,
        email: email,
        displayName: displayName,
      );

      if (gymId != null) {
        final profile = await _supabaseService.fetchCurrentProfile();
        if (profile != null) {
          await _supabaseService.setHomeGym(profile['id'], gymId);
        }
      }
    }

    return response;
  }

  Future<AuthResponse> signIn({
    required String email,
    required String password,
  }) async {
    return await Supabase.instance.client.auth.signInWithPassword(
      email: email,
      password: password,
    );
  }

  Future<void> signOut() async {
    await Supabase.instance.client.auth.signOut();
  }
}
