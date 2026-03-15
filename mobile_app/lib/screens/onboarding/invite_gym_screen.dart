import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../../main.dart';

class InviteGymScreen extends StatefulWidget {
  const InviteGymScreen({super.key});

  @override
  State<InviteGymScreen> createState() => _InviteGymScreenState();
}

class _InviteGymScreenState extends State<InviteGymScreen> {
  final _gymNameController = TextEditingController();
  final _emailController = TextEditingController();
  bool _submitting = false;

  Future<void> _submit() async {
    final gymName = _gymNameController.text.trim();
    if (gymName.isEmpty) return;

    setState(() => _submitting = true);
    try {
      final profile = await supabaseService.fetchCurrentProfile();
      if (profile == null) return;

      await Supabase.instance.client.from('gym_interest_signals').insert({
        'profile_id': profile['id'],
        'gym_name_entered': gymName,
        'owner_email': _emailController.text.trim().isNotEmpty
            ? _emailController.text.trim()
            : null,
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Thanks! We\'ll let them know.')),
        );
        Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (_) => const ClipListScreen()),
          (_) => false,
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error submitting: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Invite Your Gym')),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Know the owner or head coach? We\'ll let them know you\'re waiting for them.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 24),
            TextField(
              controller: _gymNameController,
              decoration: const InputDecoration(labelText: 'Gym Name'),
              textCapitalization: TextCapitalization.words,
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _emailController,
              decoration: const InputDecoration(
                labelText: 'Owner / Coach Email (optional)',
              ),
              keyboardType: TextInputType.emailAddress,
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: _submitting ? null : _submit,
              child: _submitting
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Submit'),
            ),
          ],
        ),
      ),
    );
  }
}
