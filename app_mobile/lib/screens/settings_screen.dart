import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({Key? key}) : super(key: key);

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool useFingerprint = false;
  bool staySignedIn = false;

  @override
  void initState() {
    super.initState();
    _loadPreferences();
  }

  Future<void> _loadPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      useFingerprint = prefs.getBool('useFingerprint') ?? false;
      staySignedIn = prefs.getBool('staySignedIn') ?? false;
    });
  }

  Future<void> _updatePreference(String key, bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(key, value);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        children: [
          SwitchListTile(
            title: const Text('Use Fingerprint Login'),
            value: useFingerprint,
            onChanged: (value) {
              setState(() {
                useFingerprint = value;
              });
              _updatePreference('useFingerprint', value);
            },
          ),
          SwitchListTile(
            title: const Text('Stay Signed In'),
            value: staySignedIn,
            onChanged: (value) {
              setState(() {
                staySignedIn = value;
              });
              _updatePreference('staySignedIn', value);
            },
          ),
        ],
      ),
    );
  }
}
