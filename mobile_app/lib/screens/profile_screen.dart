// lib/screens/profile_screen.dart
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../services/supabase_service.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final _formKey = GlobalKey<FormState>();
  final TextEditingController displayNameController = TextEditingController();
  final TextEditingController emailController = TextEditingController();
  final TextEditingController newPasswordController = TextEditingController();

  bool isLoading = true;
  String? profileId;
  final SupabaseService supabaseService = SupabaseService();

  @override
  void initState() {
    super.initState();
    loadProfile();
  }

  Future<void> loadProfile() async {
    final profile = await supabaseService.fetchCurrentProfile();
    if (profile != null) {
      profileId = profile['id'];
      displayNameController.text = profile['display_name'] ?? '';
      emailController.text = profile['email'] ?? '';
    }
    setState(() => isLoading = false);
  }

  Future<void> saveProfile() async {
    final newDisplayName = displayNameController.text.trim();
    final newPassword = newPasswordController.text.trim();

    try {
      // Update display_name in profiles table
      if (profileId != null) {
        await Supabase.instance.client
            .from('profiles')
            .update({'display_name': newDisplayName})
            .eq('id', profileId!);
      }

      // Update Supabase Auth password if provided
      if (newPassword.isNotEmpty) {
        await Supabase.instance.client.auth.updateUser(
          UserAttributes(password: newPassword),
        );
      }

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Profile updated successfully')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Profile')),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
              padding: const EdgeInsets.all(16),
              child: Form(
                key: _formKey,
                child: ListView(
                  children: [
                    TextFormField(
                      controller: emailController,
                      enabled: false,
                      decoration: const InputDecoration(labelText: 'Email'),
                    ),
                    TextFormField(
                      controller: displayNameController,
                      decoration: const InputDecoration(labelText: 'Display Name'),
                    ),
                    TextFormField(
                      controller: newPasswordController,
                      obscureText: true,
                      decoration: const InputDecoration(labelText: 'New Password (optional)'),
                    ),
                    const SizedBox(height: 16),
                    ElevatedButton(
                      onPressed: saveProfile,
                      child: const Text('Save Changes'),
                    )
                  ],
                ),
              ),
            ),
    );
  }
}
