// lib/widgets/drawer_widget.dart
import 'package:flutter/material.dart';
import '../screens/profile_screen.dart';
import '../screens/settings_screen.dart';
import '../screens/find_gym_screen.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../utils/secure_storage.dart';
import '../main.dart';

class AppDrawer extends StatelessWidget {
  const AppDrawer({super.key});

  static Future<void> globalSignOut(BuildContext context) async {
    await Supabase.instance.client.auth.signOut();
    await SecureStorage.clearCredentials();

    if (context.mounted) {
      Navigator.pushAndRemoveUntil(
        context,
        MaterialPageRoute(builder: (_) => const AuthScreen()),
        (_) => false,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          const DrawerHeader(
            decoration: BoxDecoration(color: Colors.blue),
            child: Text('Navigation Menu', style: TextStyle(color: Colors.white)),
          ),
          ListTile(
            leading: const Icon(Icons.video_library),
            title: const Text('My Clips'),
            onTap: () {
              Navigator.pop(context); // Just close the drawer
            },
          ),
          ListTile(
            leading: const Icon(Icons.location_on),
            title: const Text('Find a Gym'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const FindGymScreen()),
              );
            },
          ),
          ListTile(
            leading: const Icon(Icons.person),
            title: const Text('Profile'),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const ProfileScreen()),
              );
            },
          ),
          ListTile(
            leading: const Icon(Icons.logout),
            title: const Text('Sign Out'),
            onTap: () => globalSignOut(context),
          ),
          ListTile(
            leading: const Icon(Icons.settings),
            title: const Text('Settings'),
            onTap: () {
              Navigator.of(context).push(MaterialPageRoute(
                builder: (context) => const SettingsScreen(),
              ));
            },
          ),
        ],
      ),
    );
  }
}
