import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class UnlinkedClipsScreen extends StatefulWidget {
  const UnlinkedClipsScreen({super.key});

  @override
  State<UnlinkedClipsScreen> createState() => _UnlinkedClipsScreenState();

  /// Static helper to fetch the count of claimable clips for badge display.
  static Future<int> fetchClaimableCount() async {
    try {
      final user = Supabase.instance.client.auth.currentUser;
      if (user == null) return 0;

      final profile = await Supabase.instance.client
          .from('profiles')
          .select('tag_id, home_gym_id')
          .eq('auth_user_id', user.id)
          .maybeSingle();

      if (profile == null || profile['tag_id'] == null || profile['home_gym_id'] == null) {
        return 0;
      }

      final result = await Supabase.instance.client.rpc(
        'get_claimable_clips',
        params: {
          'p_tag_id': profile['tag_id'],
          'p_gym_id': profile['home_gym_id'],
          'p_window_hours': 72,
        },
      );

      return (result as List).length;
    } catch (_) {
      return 0;
    }
  }
}

class _UnlinkedClipsScreenState extends State<UnlinkedClipsScreen> {
  List<Map<String, dynamic>> _clips = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _fetchClaimableClips();
  }

  Future<void> _fetchClaimableClips() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final user = Supabase.instance.client.auth.currentUser;
      if (user == null) {
        setState(() {
          _loading = false;
          _error = 'Not authenticated';
        });
        return;
      }

      final profile = await Supabase.instance.client
          .from('profiles')
          .select('id, tag_id, home_gym_id')
          .eq('auth_user_id', user.id)
          .maybeSingle();

      if (profile == null || profile['tag_id'] == null || profile['home_gym_id'] == null) {
        setState(() {
          _loading = false;
          _clips = [];
        });
        return;
      }

      final result = await Supabase.instance.client.rpc(
        'get_claimable_clips',
        params: {
          'p_tag_id': profile['tag_id'],
          'p_gym_id': profile['home_gym_id'],
          'p_window_hours': 72,
        },
      );

      setState(() {
        _clips = List<Map<String, dynamic>>.from(result as List);
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _loading = false;
        _error = 'Failed to load clips: $e';
      });
    }
  }

  Future<void> _claimClip(Map<String, dynamic> clip) async {
    try {
      await Supabase.instance.client.rpc(
        'claim_clip',
        params: {
          'p_clip_id': clip['clip_id'],
          'p_fighter_side': clip['fighter_side'],
        },
      );
      setState(() {
        _clips.removeWhere((c) => c['clip_id'] == clip['clip_id']);
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Clip claimed!')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to claim clip: $e')),
        );
      }
    }
  }

  void _dismissClip(Map<String, dynamic> clip) {
    setState(() {
      _clips.removeWhere((c) => c['clip_id'] == clip['clip_id']);
    });
  }

  String _formatDuration(Map<String, dynamic> clip) {
    final start = (clip['start_seconds'] as num?)?.toDouble() ?? 0.0;
    final end = (clip['end_seconds'] as num?)?.toDouble() ?? 0.0;
    final duration = (end - start).clamp(0.0, double.infinity);
    final minutes = duration ~/ 60;
    final seconds = (duration % 60).toInt();
    return '${minutes}m ${seconds}s';
  }

  String _formatDate(Map<String, dynamic> clip) {
    final raw = clip['created_at'];
    if (raw == null) return '';
    try {
      final dt = DateTime.parse(raw.toString()).toLocal();
      return '${dt.month}/${dt.day} ${dt.hour}:${dt.minute.toString().padLeft(2, '0')}';
    } catch (_) {
      return raw.toString();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Unlinked Clips')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(child: Text(_error!))
              : _clips.isEmpty
                  ? const Center(child: Text('No unlinked clips'))
                  : RefreshIndicator(
                      onRefresh: _fetchClaimableClips,
                      child: ListView.builder(
                        itemCount: _clips.length,
                        itemBuilder: (context, index) {
                          final clip = _clips[index];
                          return Card(
                            margin: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 6,
                            ),
                            child: Padding(
                              padding: const EdgeInsets.all(12),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'Match: ${_formatDuration(clip)}',
                                    style: Theme.of(context).textTheme.titleMedium,
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    _formatDate(clip),
                                    style: Theme.of(context).textTheme.bodySmall,
                                  ),
                                  const SizedBox(height: 8),
                                  Row(
                                    mainAxisAlignment: MainAxisAlignment.end,
                                    children: [
                                      TextButton(
                                        onPressed: () => _dismissClip(clip),
                                        child: const Text('Not me'),
                                      ),
                                      const SizedBox(width: 8),
                                      ElevatedButton(
                                        onPressed: () => _claimClip(clip),
                                        child: const Text('This is me'),
                                      ),
                                    ],
                                  ),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
                    ),
    );
  }
}
