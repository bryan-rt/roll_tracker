import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'invite_gym_screen.dart';
import '../../main.dart';

class GymSelectScreen extends StatefulWidget {
  const GymSelectScreen({super.key});

  @override
  State<GymSelectScreen> createState() => _GymSelectScreenState();
}

class _GymSelectScreenState extends State<GymSelectScreen> {
  final _searchController = TextEditingController();
  List<Map<String, dynamic>> _results = [];
  bool _saving = false;

  Future<void> _search(String query) async {
    if (query.length < 2) {
      setState(() => _results = []);
      return;
    }
    try {
      final data = await Supabase.instance.client
          .from('gyms')
          .select('id, name, address')
          .ilike('name', '%$query%')
          .limit(10);
      setState(() => _results = List<Map<String, dynamic>>.from(data));
    } catch (_) {
      setState(() => _results = []);
    }
  }

  Future<void> _selectGym(String gymId) async {
    setState(() => _saving = true);
    try {
      final user = Supabase.instance.client.auth.currentUser;
      if (user == null) return;

      await Supabase.instance.client
          .from('profiles')
          .update({'home_gym_id': gymId})
          .eq('auth_user_id', user.id);

      if (mounted) {
        Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (_) => const ClipListScreen()),
          (_) => false,
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error selecting gym: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Select Your Gym')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(
              controller: _searchController,
              decoration: const InputDecoration(
                labelText: 'Search gyms',
                prefixIcon: Icon(Icons.search),
              ),
              onChanged: _search,
            ),
            const SizedBox(height: 8),
            if (_saving)
              const Center(child: CircularProgressIndicator())
            else
              Expanded(
                child: ListView.builder(
                  itemCount: _results.length,
                  itemBuilder: (context, index) {
                    final gym = _results[index];
                    return ListTile(
                      title: Text(gym['name'] ?? ''),
                      subtitle: Text(gym['address'] ?? ''),
                      onTap: () => _selectGym(gym['id']),
                    );
                  },
                ),
              ),
            const SizedBox(height: 8),
            TextButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const InviteGymScreen()),
                );
              },
              child: const Text("My gym isn't listed"),
            ),
          ],
        ),
      ),
    );
  }
}
