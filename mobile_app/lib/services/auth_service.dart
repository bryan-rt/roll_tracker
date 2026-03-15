import 'package:supabase_flutter/supabase_flutter.dart';

class AuthService {
  Stream<AuthState> get authStateChanges =>
      Supabase.instance.client.auth.onAuthStateChange;

  User? get currentUser => Supabase.instance.client.auth.currentUser;

  /// Sign up with email + password only.
  /// The DB auth trigger auto-creates the profiles row.
  /// Onboarding screens handle display_name and home_gym_id.
  Future<AuthResponse> signUp({
    required String email,
    required String password,
  }) async {
    return await Supabase.instance.client.auth.signUp(
      email: email,
      password: password,
    );
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
