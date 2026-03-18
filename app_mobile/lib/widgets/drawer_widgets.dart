// lib/widgets/drawer_widget.dart
import 'package:flutter/material.dart';
import '../screens/profile_screen.dart';
import '../screens/settings_screen.dart';
import '../screens/find_gym_screen.dart';
import '../screens/unlinked_clips_screen.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../utils/secure_storage.dart';
import '../main.dart';

class AppDrawer extends StatefulWidget {
  const AppDrawer({super.key});

  static Future<void> globalSignOut(BuildContext context) async {
    await Supabase.instance.client.auth.signOut();
    await SecureStorage.clearCredentials();

    if (context.mounted) {
      Navigator.pushAndRemoveUntil(
        context,
        MaterialPageRoute(builder: (_) => const AuthGate()),
        (_) => false,
      );
    }
  }

  @override
  State<AppDrawer> createState() => _AppDrawerState();
}

class _AppDrawerState extends State<AppDrawer> {
  int _unlinkedCount = 0;

  @override
  void initState() {
    super.initState();
    _loadUnlinkedCount();
  }

  Future<void> _loadUnlinkedCount() async {
    final count = await UnlinkedClipsScreen.fetchClaimableCount();
    if (mounted) {
      setState(() => _unlinkedCount = count);
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
            leading: const Icon(Icons.link_off),
            title: const Text('Unlinked Clips'),
            trailing: _unlinkedCount > 0
                ? Badge(
                    label: Text('$_unlinkedCount'),
                    child: const SizedBox.shrink(),
                  )
                : null,
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const UnlinkedClipsScreen()),
              ).then((_) => _loadUnlinkedCount());
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
            onTap: () => AppDrawer.globalSignOut(context),
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
